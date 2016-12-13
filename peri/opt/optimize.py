import os
import sys
import time
import tempfile
import pickle
import gc

import numpy as np
from numpy.random import randint
from scipy.optimize import newton, minimize_scalar

from peri.util import Tile
from peri import states
from peri import models as mdl
from peri.logger import log
CLOG = log.getChild('opt')

"""
If the LMEngine gets 'stuck' on the first loop attempt, since _last_vals ==
param_vals the LM will check completion and terminate. Leaving as is since I've
only got this to happen when it's at the minimum...
TODO:
    1. burn -- 2 loops of globals is only good for the first few loops; after
            that it's overkill. Best to make a 3rd mode now, since the
            psf and zscale move around without a set of burns.
    2. better optimization? things like run_3 with eigenvalue blocks

To add:
1. AugmentedState: ILM scale options? You'd need a way to get an overall scale
    block, which would probably need to come from the ILM itself.
6. With opt using big regions for particles, globals, it makes sense to
    put stuff back on the card again....

To fix:
1.  In the engine, make do_run_1() and do_run_2() play nicer with each other.
2.  Right now, when marquardt_damping=False (the default, which works nicely),
    the correct damping parameter scales with the image size. For each element
    of J is O(1), so JTJ[i,j]~1^2 * N ~ N where N is the number of residuals
    pixels. But the damping matrix only matters in its overall ratio to J.
    So, changing max_mem or changing the image size will affect what a
    reasonable damping is. One way to do this is to scale the damping by
    the size of the residuals..........................................

LM Algorithm is:
1. Evaluate J_ia = df(xi,mu)/dmu_a
2. Solve the for delta:
    (J^T*J + l*Diag(J^T*J))*delta = J^T(y-f(xi,mu))     (1)
3. Update mu -> mu + delta

To solve eq. (1), we need to:
1. Construct the matrix JTJ = J^T*J
2. Construct the matrix A=JTJ + l*Diag(JTJ)
3. Construct err= y-f(x,beta)
4. np.linalg.leastsq(A,err, rcond=min_eigval) to avoid near-zero eigenvalues

My only change to this is, instead of calculating J_ia, we calculate
J_ia for a small subset (say 1%) of the pixels in the image randomly selected,
rather than for all the pixels (in addition to leastsq solution instead of
linalg.solve
"""

def get_rand_Japprox(s, params, num_inds=1000, **kwargs):
    """
    Calculates a random approximation to J by returning J only at a
    set of random pixel/voxel locations.

    Parameters
    ----------
        s : :class:`peri.states.State`
            The state to calculate J for.
        params : List
            The list of parameter names to calculate the gradient of.
        num_inds : Int, optional.
            The number of pix/voxels at which to calculate the random
            approximation to J. Default is 1000.

    **kwargs Parameters
    -------------------
        All kwargs parameters get passed to s.gradmodel only.

    Returns
    -------
        J : numpy.ndarray
            [d, num_inds] array of J, at the given indices.

        return_inds : numpy.ndarray or slice
            [num_inds] element array or slice(0, None) of the model
            indices at which J was evaluated.
    """
    start_time = time.time()
    tot_pix = s.residuals.size
    if num_inds < tot_pix:
        inds = np.random.choice(tot_pix, size=num_inds, replace=False)
        slicer = None
        return_inds = inds
    else:
        inds = None
        return_inds = slice(0, None)
        slicer = [slice(0, None)]*len(s.residuals.shape)
    #J = d/dx( residuals ) = d/dx( data - model) = -d/dx( model)
    J = -s.gradmodel(params=params, inds=inds, slicer=slicer, flat=False, **kwargs)
    CLOG.debug('JTJ:\t%f' % (time.time()-start_time))
    return J, return_inds

def name_globals(s, remove_params=None):
    """
    Returns a list of the global parameter names.

    Parameters
    ----------
        s : :class:`peri.states.ImageState`
            The state to name the globals of.
        remove_params : Set or None
            A set of unique additional parameters to remove from the globals
            list.

    Returns
    -------
        all_params : list
            The list of the global parameter names, with each of
            remove_params removed.
    """
    all_params = s.params
    for p in s.param_positions():  #FIXME s.param_particle()??
        all_params.remove(p)
    for p in s.param_radii():
        all_params.remove(p)
    if remove_params is not None:
        for p in set(remove_params):
            all_params.remove(p)
    return all_params

def get_num_px_jtj(s, nparams, decimate=1, max_mem=1e9, min_redundant=20):
    """
    Calculates the number of pixels to use for J at a given memory usage.

    Tries to pick a number of pixels as (size of image / `decimate`).
    However, clips this to a maximum size and minimum size to ensure that
    (1) too much memory isn't used and (2) J has enough elements so that
    the inverse of JTJ will be well-conditioned.

    Parameters
    ----------
        s : :class:`peri.states.State`
            The state on which to calculate J.
        nparams : Int
            The number of parameters that will be included in J.
        decimate : Int, optional
            The amount to decimate the number of pixels in the image by,
            i.e. tries to pick num_px = size of image / decimate.
            Default is 1
        max_mem : Numeric, optional
            The maximum allowed memory, in bytes, for J to occupy at
            double-precision. Default is 1e9.
        min_redundant : Int, optional
            The number of pixels must be at least `min_redundant` *
            `nparams`. If not, an error is raised. Default is 20

    Returns
    -------
        num_px : Int
            The number of pixels at which to calcualte J.
    """
    #1. Max for a given max_mem:
    px_mem = int(max_mem / 8 / nparams) #1 float = 8 bytes
    #2. num_pix for a given redundancy
    px_red = min_redundant*nparams
    #3. And # desired for decimation
    px_dec = s.residuals.size/decimate

    if px_red > px_mem:
        raise RuntimeError('Insufficient max_mem for desired redundancy.')
    num_px = np.clip(px_dec, px_red, px_mem).astype('int')
    return num_px

def vectorize_damping(params, damping=1.0, increase_list=[['psf-', 1e4]]):
    """
    Returns a non-constant damping vector, allowing certain parameters to be
    more strongly damped than others.

    Parameters
    ----------
        params : List
            The list of parameter names, in order.
        damping : Float
            The default value of the damping.
        increase_list: List
            A nested 2-element list of the params to increase and their
            scale factors. All parameters containing the string
            increase_list[i][0] are increased by a factor increase_list[i][1].
    Returns
    -------
        damp_vec : np.ndarray
            The damping vector to use.
    """
    damp_vec = np.ones(len(params)) * damping
    for nm, fctr in increase_list:
        for a in xrange(damp_vec.size):
            if nm in params[a]:
                damp_vec[a] *= fctr
    return damp_vec

def halve_randomly(blk):
    """
    Randomly halves a block.

    Given an array blk of bools, returns two arrays `blk1`, `blk2` such
    that `blk1` | `blk2` = `blk`, `blk1` & `blk2` = 0, and `blk1` and
    `blk2` have an equal amount of True's.

    Parameters
    ----------
        blk : numpy.ndarray
            1D ndarray of booleans

    Returns
    -------
        blk1 : numpy.ndarray
            The first boolean block.
        blk2 : numpy.ndarray
            The second boolean block.
    """
    inds = np.nonzero(blk)[0]
    np.random.shuffle(inds)  #in-place shuffling
    blk1 = np.zeros_like(blk, dtype='bool')
    blk2 = np.zeros_like(blk, dtype='bool')
    blk1[inds[:inds.size/2]] = True
    blk2[inds[inds.size/2:]] = True
    return [blk1,blk2]

#=============================================================================#
#               ~~~~~  Particle Optimization stuff  ~~~~~
#=============================================================================#
def find_particles_in_tile(state, tile):
    """
    Finds the particles in a tile, as numpy.ndarray of ints.

    Parameters
    ----------
        state : :class:`peri.states.ImageState`
            The state to locate particles in.
        tile : :class:`peri.util.Tile` instance
            Tile of the region inside which to check for particles.

    Returns
    -------
        numpy.ndarray, int
            The indices of the particles in the tile.
    """
    bools = tile.contains(state.obj_get_positions())
    return np.arange(bools.size)[bools]

def separate_particles_into_groups(s, region_size=40, bounds=None):
    """
    Separates particles into convenient groups for optimization.

    Given a state, returns a list of groups of particles. Each group of
    particles are located near each other in the image. Every particle
    located in the desired region is contained in exactly 1 group.

    Parameters
    ----------
    s : :class:`peri.states.ImageState`
        The peri state to find particles in.
    region_size : Int or 3-element list-like of ints, optional
        The size of the box. Groups particles into boxes of shape
        (region_size[0], region_size[1], region_size[2]). If region_size
        is a scalar, the box is a cube of length region_size.
        Default is 40.
    bounds : 2-element list-like of 3-element lists, optional
        The sub-region of the image over which to look for particles.
            bounds[0]: The lower-left  corner of the image region.
            bounds[1]: The upper-right corner of the image region.
        Default (None -> ([0,0,0], s.oshape.shape)) is a box of the entire
        image size, i.e. the default places every particle in the image
        somewhere in the groups.

    Returns
    -------
    particle_groups : List
        Each element of particle_groups is an int numpy.ndarray of the
        group of nearby particles. Only contains groups with a nonzero
        number of particles, so the elements don't necessarily correspond
        to a given image region.
    """
    bounding_tile = (s.oshape.translate(-s.pad) if bounds is None else
            Tile(bounds[0], bounds[1]))
    rs = (np.array([region_size, region_size, region_size]).ravel() if
            np.size(region_size) == 1 else np.array(region_size))

    n_translate = np.ceil(bounding_tile.shape.astype('float')/rs).astype('int')
    particle_groups = []
    tile = Tile(left=bounding_tile.l, right=bounding_tile.l + rs)
    d0s, d1s, d2s = np.meshgrid(*[np.arange(i) for i in n_translate])

    groups = map(lambda d0, d1, d2: find_particles_in_tile(s, tile.translate(
            np.array([d0,d1,d2]) * rs)), d0s.ravel(), d1s.ravel(), d2s.ravel())
    for i in xrange(len(groups)-1, -1, -1):
        if groups[i].size == 0:
            groups.pop(i)

    return groups

def calc_particle_group_region_size(s, region_size=40, max_mem=1e9, **kwargs):
    """
    Finds the biggest region size for LM particle optimization with a
    given memory constraint.

    Input Parameters
    ----------------
        s : :class:`peri.states.ImageState`
            The state with the particles
        region_size : Int or 3-element list-like of ints, optional.
            The initial guess for the region size. Default is 40
        max_mem : Numeric, optional
            The maximum memory for the optimizer to take. Default is 1e9

    **kwargs
    --------
        bounds: 2-element list-like of 3-element lists.
            The sub-region of the image over which to look for particles.
                bounds[0]: The lower-left  corner of the image region.
                bounds[1]: The upper-right corner of the image region.
            Default (None -> ([0,0,0], s.oshape.shape)) is a box of the entire
            image size, i.e. the default places every particle in the image
            somewhere in the groups.
    Returns
    -------
        region_size : numpy.ndarray of ints of the region size.
    """
    region_size = np.array(region_size).astype('int')

    def calc_mem_usage(region_size):
        rs = np.array(region_size)
        particle_groups = separate_particles_into_groups(s, region_size=
                rs.tolist(), **kwargs)
        #The actual max_mem is np.max(map(f, p_groups) where
        # f = lambda g: get_slicered_difference(s, get_tile_from_multiple_
        #   particle_change(s, g).slicer, s.image_mask[" " " " .slicer] == 1)
        #   .nbytes * g.size * 4
        #However this is _way_ too slow. A better approximation is
        # d = s.residuals
        # max_mem = np.max(map(lambda g: d[get_tile_from_multiple_particle_change(
                # s, g).slicer].nbytes * g.size * 4, particle_groups))
        # return max_mem
        ##But this is still too slow (like 1 min vs 250 ms). So instead --
        num_particles = np.max(map(np.size, particle_groups))
        psf_shape = s.get('psf').get_padding_size(s.ishape).shape
        mem_per_part = 32 * np.prod(rs + (psf_shape + np.median(s.obj_get_radii())))
        return num_particles * mem_per_part

    im_shape = s.oshape.shape
    if calc_mem_usage(region_size) > max_mem:
        while ((calc_mem_usage(region_size) > max_mem) and
                np.any(region_size > 2)):
            region_size = np.clip(region_size-1, 2, im_shape)
    else:
        while ((calc_mem_usage(region_size) < max_mem) and
                np.any(region_size < im_shape)):
            region_size = np.clip(region_size+1, 2, im_shape)
        region_size -= 1 #need to be < memory, so we undo 1 iteration

    return region_size

def get_residuals_update_tile(st, padded_tile):
    """
    Translates a tile in the padded image to the unpadded image.

    Given a state and a tile that corresponds to the padded image, returns
    a tile that corresponds to the the corresponding pixels of the difference
    image

    Parameters
    ----------
        st : :class:`peri.states.State`
            The state
        padded_tile : :class:`peri.util.Tile`
            The tile in the padded image.

    Returns
    -------
        :class:`peri.util.Tile`
            The tile corresponding to padded_tile in the unpadded image.
    """
    inner_tile = st.ishape.intersection([st.ishape, padded_tile])
    return inner_tile.translate(-st.pad)

#=============================================================================#
#         ~~~~~        Class/Engine LM minimization Stuff     ~~~~~
#=============================================================================#

def find_best_step(err_vals):
    """
    Returns the index of the lowest of the passed values. Catches nans etc.
    """
    if np.all(np.isnan(err_vals)):
        raise ValueError('All err_vals are nans!')
    return np.nanargmin(err_vals)

class LMEngine(object):
    """
    Levenberg-Marquardt engine with all the options from the M. Transtrum
    J. Sethna 2012 ArXiV paper [1]_.

    Parameters
    ----------
        damping: Float, optional
            The initial damping factor for Levenberg-Marquardt. Adjusted
            internally. Default is 1.
        increase_damp_factor: Float, optional
            The amount to increase damping by when an attempted step
            has failed. Default is 3.
        decrease_damp_factor: Float, optional
            The amount to decrease damping by after a successful step.
            Default is 8. increase_damp_factor and decrease_damp_factor
            must not have all the same factors.

        min_eigval: Float scalar, optional, <<1.
            The minimum eigenvalue to use in inverting the JTJ matrix,
            to avoid degeneracies in the parameter space (i.e. 'rcond'
            in np.linalg.lstsq). Default is 1e-12.
        marquardt_damping: Bool, optional
            Set to False to use Levenberg damping (damping matrix
            proportional to the identiy) instead of Marquardt damping
            (damping matrix proportional to the diagonal terms of JTJ).
            Default is False.
        transtrum_damping: Float or None, optional
            If not None, then clips the Marquardt damping diagonal
            entries to be at least transtrum_damping. Default is None.

        use_accel: Bool, optional
            Set to True to incorporate the geodesic acceleration term
            from M. Transtrum J. Sethna 2012. Default is False.
        max_accel_correction: Float, optional
            Acceleration corrections bigger than max_accel_correction*
            the normal LM step are viewed as bad steps, causing a
            decrease in damping. Default is 1.0. Only applies to the
            do_run_1 method.

        paramtol : Float, optional
            Algorithm has converged when the none of the parameters have
            changed by more than paramtol. Default is 1e-6.
        errtol : Float, optional
            Algorithm has converged when the error has changed by less
            than errtol after 1 step. Default is 1e-6.
        exptol : Float, optional
            Algorithm has converged when the expected change in error is
            less than exptol. Default is 1e-3.
        fractol : Float, optional
            Algorithm has converged when the error has changed by a
            fractional amount less than fractol after 1 step.
            Default is 1e-6.
        costol : Float, optional
            Algorithm has converged when the cosine of the angle between
            (residuals projected onto the model manifold) and (the
            residuals) is < costol. Default is None, i.e. doesn't check
            the cosine (since it takes a bit of time).
        max_iter : Int, optional
            The maximum number of iterations before the algorithm
            stops iterating. Default is 5.

        update_J_frequency: Int, optional
            The frequency to re-calculate the full Jacobian matrix.
            Default is 2, i.e. every other run.
        broyden_update: Bool, optional
            Set to True to do a Broyden partial update on J after
            each step, updating the projection of J along the
            parameter change direction. Cheap in time cost, but not
            always accurate. Default is False.
        eig_update: Bool, optional
            Set to True to update the projection of J along the most
            stiff eigendirections of JTJ. Slower than broyden but
            more accurate & useful. Default is False.
        num_eig_dirs: Int, optional
            If eig_update == True, the number of eigendirections to
            update when doing the eigen update. Default is 4.
        eig_update_frequency: Int, optional
            If eig_update, the frequency to do this partial update.
            Default is 3.
        broyden_update_frequency: Int, optional
            If broyden_update, the frequency to do this partial update.
            Default is 1.

    Attributes
    ----------
        J : numpy.ndarray
            The calculated J for Levenberg-Marquardt. Starts as `None`
        JTJ : numpy.ndarray
            The approximation to the fit Hessian; np.dot(J, J.T)
        damping : numpy.ndarray
            The current damping vector for the parameters.
        _num_iter : Int
            The number of iterations ran in the current cycle. Don't touch

    Methods
    -------
        reset(new_damping=None)
            Resets all counters etc so a new run can commence.
        do_run_1()
            Method 1 for optimization.
        do_run_2()
            Method 2 for optimization
        do_internal_run()
            Additional, slight optimization once J has been calculated
        do_run_3()
            Experimental... FIXME
        do_stuckJ_run()
            Experimental... FIXME
        find_LM_updates(grad, do_correct_damping=True, subblock=None)
            Returns the Levenberg-Marquardt step.
        increase_damping()
            Increases damping
        decrease_damping(undo_decrease=False)
            Decreases damping or undoes a previous decrease.
        update_param_vals(new_vals, incremental=False)
            Updates the current set of parameter values and previous
            values, sets a flag to re-calculate J.
        calc_model_cosine(decimate=None)
            Calculates the cosine of the residuals with the current
            model tangent space J
        get_termination_stats(get_cos=True)
            Returns a dict of termination statistics
        check_completion()
            Checks if the algorithm has found a satisfactory minimum
        check_termination()
            Checks if the algorithm should terminate
        update_J()
            Updates J, JTJ
        calc_grad()
            Calculates the gradient of the cost w.r.t. the parameters.
        update_Broyden_J()
            Execute a Broyden update of J
        update_eig_J()
            Execute an eigen update of J
        calc_accel_correction(damped_JTJ, delta0)
            Calculates the geodesic acceleration correction to the
            standard Levenberg-Marquardt step.
        update_select_J(blk)
            Updates J only for certain parameters, described by the
            boolean mask `blk`.

    See Also
    --------
        LMFunction
        LMGlobals
        LMParticles
        LMParticleGroupCollection
        LMAugmentedState
        LMOptObj

    Notes
    -----
    There are 3 different options for optimizing:
        do_run_1():
            Checks to calculate full, Broyden, and eigen J, then tries a step.
            If the step is accepted, decreases damping; if not, increases.
        do_run_2():
            Checks to calculate full, Broyden, and eigen J, then tries a
            step with the current damping and with a decreased damping,
            accepting whichever is lower. Decreases damping iff the lower
            damping is better. It then calls do_internal_run() (see below).
            Rejected steps result in increased damping until a step is
            accepted. Checks for full, Broyden, and eigen J updates.
        do_internal_run():
            Checks for Broyden and eigen J updates only, then uses
            pre-calculated J, JTJ, etc to evaluate LM steps. Does
            not change damping during the run. Does not check do update
            the full J, but does check for Broyden, eigen updates.
            Does not work if J has not been evaluated yet.
    Whether to update the full J is controlled by update_J_frequency only,
    which only counts iterations of do_run_1() and do_run_2().
    Partial updates are controlled by *_update_frequency, which
    counts internal runs in do_internal_run and full runs in do_run_1.

    So, if you want a partial update every other run, full J the remaining,
    this would be:
        do_run_1(): update_J_frequency=2, partial_update_frequency=1
        do_run_2(): update_J_frequency=1, partial_update_frequency=1,
                    run_length=2

    Partial Updates:
    Broyden update  : an update to J (and then JTJ) by approximating the
        derivative of the model as the finite difference of the last
        step. (rank-1)
    Eigen update    : a rank-num_eig_dirs update to J (and then JTJ) by
        finite-differencing with eig_dl along the highest num_eig_dirs
        eigendirections.

    Damping:
    marquardt : Damp proportional to the diagonal elements of JTJ
    transtrum : Marquardt damping, clipped to be at least a certain number
    default   : (levenberg) : Damp using something proportional to the
            identity

    References
    ----------
        ..[1] M. Transtrum and J. Sethna, "Improvements to the Levenberg-
            Marquardt algorithm for nonlinear least-squares minimization,"
            ArXiV preprint arXiv:1201.5885 (2012)
    """
    def __init__(self, damping=1., increase_damp_factor=3., decrease_damp_factor=8.,
                min_eigval=1e-13, marquardt_damping=False, transtrum_damping=None,
                use_accel=False, max_accel_correction=1., paramtol=1e-6,
                errtol=1e-5, exptol=1e-3, fractol=1e-6, costol=None,
                max_iter=5, run_length=5, update_J_frequency=1,
                broyden_update=True, eig_update=False, eig_update_frequency=3,
                num_eig_dirs=8, eig_dl=1e-5, broyden_update_frequency=1):
        self.increase_damp_factor = float(increase_damp_factor)
        self.decrease_damp_factor = float(decrease_damp_factor)
        self.min_eigval = min_eigval
        self.marquardt_damping = marquardt_damping
        self.transtrum_damping = transtrum_damping

        self.use_accel = use_accel
        self.max_accel_correction = max_accel_correction

        self.paramtol = paramtol
        self.errtol = errtol
        self.exptol = exptol
        self.fractol = fractol
        self.costol = costol
        self.max_iter = max_iter

        self.update_J_frequency = update_J_frequency
        self.broyden_update = broyden_update
        self.eig_update = eig_update
        self.num_eig_dirs = num_eig_dirs
        self.run_length = run_length
        self._inner_run_counter = 0
        self.eig_update_frequency = eig_update_frequency
        self.broyden_update_frequency = broyden_update_frequency

        #Initializing counters etc for the first loop:
        self._num_iter = 0
        self._exp_err = 10*self.exptol

        #We want to start updating JTJ
        self.J = None
        self._J_update_counter = update_J_frequency
        self._fresh_JTJ = False

        #the max # of times trying to decrease damping before giving up
        self._max_inner_loop = 15

        #Finally we set the error and parameter values:
        self._set_err_paramvals()
        self.damping = np.ones(self.param_vals.size, dtype='float')
        self.damping[:] = np.array(damping)  #keeping the damping always a vector
        self._has_run = False
        self.eig_dl = eig_dl

    def reset(self, new_damping=None):
        """
        Keeps all user supplied options the same, but resets counters etc.
        """
        self._num_iter = 0
        self._inner_run_counter = 0
        self._J_update_counter = self.update_J_frequency
        self._fresh_JTJ = False
        self._has_run = False
        if new_damping is not None:
            self.damping = np.array(new_damping).astype('float')
        self._set_err_paramvals()

    def _set_err_paramvals(self):
        """
        Must update:
            self.error, self._last_error, self.param_vals, self._last_vals
        """
        raise NotImplementedError('implement in subclass')

    def calc_J(self):
        """Updates self.J, returns nothing"""
        raise NotImplementedError('implement in subclass')

    def calc_residuals(self):
        """returns residuals = data - model."""
        raise NotImplementedError('implement in subclass')

    def update_function(self, param_vals):
        """Takes an array param_vals, updates function, returns the new error"""
        raise NotImplementedError('implement in subclass')

    def do_run_1(self):
        """
        LM run, evaluating 1 step at a time.

        Broyden or eigendirection updates replace full-J updates until
        a full-J update occurs. Does not run with the calculated J (no
        internal run).
        """
        while not self.check_terminate():
            self._has_run = True
            self._run1()
            self._num_iter += 1; self._inner_run_counter += 1

    def _run1(self):
        """workhorse for do_run_1"""
        if self.check_update_J():
            self.update_J()
        else:
            if self.check_Broyden_J():
                self.update_Broyden_J()
            if self.check_update_eig_J():
                self.update_eig_J()

        #1. Assuming that J starts updated:
        delta_vals = self.find_LM_updates(self.calc_grad())

        #2. Increase damping until we get a good step:
        er1 = self.update_function(self.param_vals + delta_vals)
        good_step = (find_best_step([self.error, er1]) == 1)
        if not good_step:
            er0 = self.update_function(self.param_vals)
            if np.abs(er0 -self.error)/er0 > 1e-7:
                raise RuntimeError('Function updates are not exact.')
            CLOG.debug('Bad step, increasing damping')
            CLOG.debug('\t\t%f\t%f' % (self.error, er1))
            grad = self.calc_grad()
            for _try in xrange(self._max_inner_loop):
                self.increase_damping()
                delta_vals = self.find_LM_updates(grad)
                er1 = self.update_function(self.param_vals + delta_vals)
                good_step = (find_best_step([self.error, er1]) == 1)
                if good_step:
                    break
            else:
                er0 = self.update_function(self.param_vals)
                CLOG.warn('Stuck!')
                if np.abs(er0 -self.error)/er0 > 1e-7:
                    raise RuntimeError('Function updates are not exact.')

        #state is updated, now params:
        if good_step:
            self._last_error = self.error
            self.error = er1
            CLOG.debug('Good step\t%f\t%f' % (self._last_error, self.error))
            self.update_param_vals(delta_vals, incremental=True)
            self.decrease_damping()

    def do_run_2(self):
        """
        LM run evaluating 2 steps (damped and not) and choosing the best.

        After finding the best of 2 steps, runs with that damping + Broyden
        or eigendirection updates, until deciding to do a full-J update.
        Only changes damping after full-J updates.
        """
        while not self.check_terminate():
            self._has_run = True
            self._run2()
            self._num_iter += 1

    def _run2(self):
        """Workhorse for do_run_2"""
        if self.check_update_J():
            self.update_J()
        else:
            if self.check_Broyden_J():
                self.update_Broyden_J()
            if self.check_update_eig_J():
                self.update_eig_J()

        #0. Find _last_residuals, _last_error, etc:
        _last_residuals = self.calc_residuals().copy()
        _last_error = 1*self.error
        _last_vals = self.param_vals.copy()

        #1. Calculate 2 possible steps
        delta_params_1 = self.find_LM_updates(self.calc_grad(),
                do_correct_damping=False)
        self.decrease_damping()
        delta_params_2 = self.find_LM_updates(self.calc_grad(),
                do_correct_damping=False)
        self.decrease_damping(undo_decrease=True)

        #2. Check which step is best:
        er1 = self.update_function(self.param_vals + delta_params_1)
        er2 = self.update_function(self.param_vals + delta_params_2)

        triplet = (self.error, er1, er2)
        best_step = find_best_step(triplet)
        if best_step == 0:
            #Both bad steps, put back & increase damping:
            _ = self.update_function(self.param_vals.copy())
            grad = self.calc_grad()
            CLOG.debug('Bad step, increasing damping')
            CLOG.debug('%f\t%f\t%f' % triplet)
            for _try in xrange(self._max_inner_loop):
                self.increase_damping()
                delta_vals = self.find_LM_updates(grad)
                er_new = self.update_function(self.param_vals + delta_vals)
                good_step = er_new < self.error
                if good_step:
                    #Update params, error, break:
                    self.update_param_vals(delta_vals, incremental=True)
                    self.error = er_new
                    CLOG.debug('Sufficiently increased damping')
                    CLOG.debug('%f\t%f' % (triplet[0], self.error))
                    break
            else: #for-break-else
                #Throw a warning, put back the parameters
                CLOG.warn('Stuck!')
                self.error = self.update_function(self.param_vals.copy())

        elif best_step == 1:
            #er1 <= er2:
            good_step = True
            CLOG.debug('Good step, same damping')
            CLOG.debug('%f\t%f\t%f' % triplet)
            #Update to er1 params:
            er1_1 = self.update_function(self.param_vals + delta_params_1)
            if np.abs(er1_1 - er1) > 1e-6:
                raise RuntimeError('Function updates are not exact.')
            self.update_param_vals(delta_params_1, incremental=True)
            self.error = er1

        elif best_step == 2:
            #er2 < er1:
            good_step = True
            self.error = er2
            CLOG.debug('Good step, decreasing damping')
            CLOG.debug('%f\t%f\t%f' % triplet)
            #-we're already at the correct parameters
            self.update_param_vals(delta_params_2, incremental=True)
            self.decrease_damping()

        #3. Run with current J, damping; update what we need to::
        if good_step:
            self._last_residuals = _last_residuals
            self._last_error = _last_error
            self._last_vals = _last_vals
            self.error
            self.do_internal_run(initial_count=1)

    def do_internal_run(self, initial_count=0, subblock=None, update_derr=True):
        """
        Takes more steps without calculating J again.

        Given a fixed damping, J, JTJ, iterates calculating steps, with
        optional Broyden or eigendirection updates. Iterates either until
        a bad step is taken or for self.run_length times.
        Called internally by do_run_2() but is also useful on its own.

        Parameters
        ----------
            initial_count : Int, optional
                The initial count of the run. Default is 0. Increasing from
                0 effectively temporarily decreases run_length.
            subblock : None or np.ndarray of bools, optional
                If not None, a boolean mask which determines which sub-
                block of parameters to run over. Default is None, i.e.
                all the parameters.
            update_derr : Bool, optional
                Set to False to not update the variable that determines
                delta_err, preventing premature termination through errtol.

        Notes
        -----
        It might be good to do something similar to update_derr with the
        parameter values, but this is trickier because of Broyden updates
        and _fresh_J.
        """
        self._inner_run_counter = initial_count; good_step = True
        n_good_steps = 0
        CLOG.debug('Running...')

        _last_residuals = self.calc_residuals().copy()
        while ((self._inner_run_counter < self.run_length) & good_step &
                (not self.check_terminate())):
            #1. Checking if we update J
            if self.check_Broyden_J() and self._inner_run_counter != 0:
                self.update_Broyden_J()
            if self.check_update_eig_J() and self._inner_run_counter != 0:
                self.update_eig_J()

            #2. Getting parameters, error
            er0 = 1*self.error
            delta_vals = self.find_LM_updates(self.calc_grad(),
                    do_correct_damping=False, subblock=subblock)
            er1 = self.update_function(self.param_vals + delta_vals)
            good_step = er1 < er0

            if good_step:
                n_good_steps += 1
                CLOG.debug('%f\t%f' % (er0, er1))
                #Updating:
                self.update_param_vals(delta_vals, incremental=True)
                self._last_residuals = _last_residuals.copy()
                if update_derr:
                    self._last_error = er0
                self.error = er1

                _last_residuals = self.calc_residuals().copy()
            else:
                er0_0 = self.update_function(self.param_vals)
                CLOG.debug('Bad step!')
                if np.abs(er0 - er0_0) > 1e-6:
                    raise RuntimeError('Function updates are not exact.')

            self._inner_run_counter += 1
        return n_good_steps

    def do_run_3(self, max_bad=None, max_updates=3, stop_halving=1):
        """
        I need better names for these functions
        Runs 1 step of run2, then runs with stuckJ until the # of bad
        parameter blocks is at least max_bad. Counts both run2 and stuckJ
        runs towards max_iter counter.
        Parameters
        ----------
            max_bad : Int or None
                The maximum number of bad sub-blocks of J before giving
                up and re-calculating an entire J. Default is None, which
                uses self.param_vals.size / 4

            max_updates : Int
                The maximum number of times to attempt the local ln(N) updates
                of J. Default is 3.

            stop_halving : Int
                When the size of a bad block is < stop_halving, the entire
                block is counted as bad and updated, rather than chasing
                down to the very last bad block. Default is 1.

        Maybe you should ensure that the run2() gave a bad J first? But I
        also don't want to end up in a situation where the optimizer moves
        by 1e-10 for 100 steps... -- should be OK because of errtol.
        One other issue with this is that if you move around in 1 parameter
        you can get a small but positive change in the error, which can
        create a premature stoppage through self.check_terminate()
        """
        n_bad = 0
        max_bad = self.param_vals.size / 4 if max_bad is None else max_bad
        while not self.check_terminate():
            self._has_run = True
            self._run2()
            self._num_iter += 1
            for _ in xrange(max_updates):
                n_bad = self.do_stuckJ_run(stop_halving=stop_halving)
                self._num_iter += 1  #Maybe we should could full J updates, maybe not
                if (self.check_terminate()) or (n_bad >= max_bad):
                    break

    def do_stuckJ_run(self, stop_halving=1):
        """
        Optimization when a J is calculated, but trying to step with a full
        J results in a bad step.
            stop_halving : Int
                When the # of elements in a bad block is < stop_halving,
                just call the whole sub-block bad. Default is 1.
        """
        #0. Initialize an active list of sub-blocks of J
        full_blk = np.ones(self.J.shape[0], dtype='bool')
        active_list = halve_randomly(full_blk)
        good_params = np.zeros_like(full_blk, dtype='bool')
        bad_params = np.zeros_like(full_blk, dtype='bool')

        while len(active_list) > 0:
            #1. Pop one active sub-block
            # cur_block = active_list.pop(np.random.choice(len(active_list)))
            cur_block = active_list.pop(0)  #largest rather than random block.

            #2. Take a step with the sub-block
            CLOG.debug('Run with {}-element subblock:'.format(cur_block.sum()))
            n_steps = self.do_internal_run(subblock=cur_block, update_derr=False)

            #3. IF subblock is good or bad, flow:
            if n_steps > 0:  #at least 1 good step == good block
                good_params |= cur_block
            elif cur_block.sum() > stop_halving:  #bad block, can be halved
                active_list.extend(halve_randomly(cur_block))
            else:
                bad_params |= cur_block
            #4. run while len(active_list) > 0

        #5. Try a step with the good parameters
        ngood = good_params.sum()
        if ngood > 0:
            CLOG.debug('Run with all {} good parameters.'.format(ngood))
            _ = self.do_internal_run(subblock=good_params)

        if bad_params.sum() > 0:
            #6. Update the bad parameters
            self.update_select_J(bad_params)
            #7. Run with full J
            CLOG.debug('Try run with all parameters.')
            n_full_steps = self.do_internal_run()  #??
            if (n_full_steps == 0):  #bad step:
                CLOG.debug('Run with freshly-updated {} bad parameters.'.format(bad_params.sum()))
                n_bad_steps = self.do_internal_run(subblock=bad_params)

                #If we took a bad step, we know the newly-updated J is correct:
                _try = 0
                while ((n_bad_steps == 0) & (not self.check_terminate()) &
                        (_try < self._max_inner_loop)):
                    _try += 1
                    self.increase_damping()
                    n_bad_steps = self.do_internal_run(subblock=bad_params)
                if _try >= self._max_inner_loop:
                    CLOG.warn('Stuck!')
                else:
                    #successfully ran with updated bad params, now with full J
                    CLOG.debug('Try again with all parameters.')
                    _ = self.do_internal_run()
        #8. Because we might want to keep running this until it stops working,
        #   we return the number of bad parameters to decide whether to
        #   keep going.
        return bad_params.sum()

    def _calc_damped_jtj(self, JTJ, subblock=None):
        if self.marquardt_damping:
            diag_vals = np.diag(JTJ)
        elif self.transtrum_damping is not None:
            diag_vals = np.clip(np.diag(JTJ), self.transtrum_damping, np.inf)
        else:
            diag_vals = np.ones(JTJ.shape[0])

        diag = np.diagflat(diag_vals)
        if subblock is None:
            damped_JTJ = JTJ + self.damping*diag
        else:
            damped_JTJ = JTJ + self.damping[subblock]*diag
        return damped_JTJ

    def find_LM_updates(self, grad, do_correct_damping=True, subblock=None):
        """
        Calculates LM updates, with or without the acceleration correction.

        Paramters
        ---------
            grad : numpy.ndarray
                The gradient of the model cost.
            do_correct_damping : Bool, optional
                If `self.use_accel`, then set to True to correct damping
                if the acceleration correction is too big. Default is True
                Does nothing is `self.use_accel` is False
            subblock : slice, numpy.ndarray, or None, optional
                Set to a slice or a valide numpy.ndarray to use only a
                certain subset of the parameters. Default is None, i.e.
                use all the parameters.

        Returns
        -------
            delta : numpy.ndarray
                The Levenberg-Marquadt step, relative to the old
                parameters. Size is always self.param_vals.size.
        """
        if subblock is not None:
            if (subblock.sum() == 0) or (subblock.size == 0):
                CLOG.fatal('Empty subblock in find_LM_updates')
                raise ValueError('Empty sub-block')
            j = self.J[subblock]
            JTJ = np.dot(j, j.T)
            damped_JTJ = self._calc_damped_jtj(JTJ, subblock=subblock)
            grad = grad[subblock]  #select the subblock of the grad
        else:
            damped_JTJ = self._calc_damped_jtj(self.JTJ, subblock=subblock)

        delta = self._calc_lm_step(damped_JTJ, grad, subblock=subblock)

        if self.use_accel:
            accel_correction = self.calc_accel_correction(damped_JTJ, delta)
            nrm_d0 = np.sqrt(np.sum(delta**2))
            nrm_corr = np.sqrt(np.sum(accel_correction**2))
            CLOG.debug('|correction| / |LM step|\t%e' % (nrm_corr/nrm_d0))
            if nrm_corr/nrm_d0 < self.max_accel_correction:
                delta += accel_correction
            elif do_correct_damping:
                CLOG.debug('Untrustworthy step! Increasing damping...')
                self.increase_damping()
                damped_JTJ = self._calc_damped_jtj(JTJ, subblock=subblock)
                delta = self._calc_lm_step(damped_JTJ, grad, subblock=subblock)

        if np.any(np.isnan(delta)):
            CLOG.fatal('Calculated steps have nans!?')
            raise FloatingPointError('Calculated steps have nans!?')
        return delta

    def _calc_lm_step(self, damped_JTJ, grad, subblock=None):
        """Calculates a Levenberg-Marquard step w/o acceleration"""
        delta0, res, rank, s = np.linalg.lstsq(damped_JTJ, -0.5*grad,
                rcond=self.min_eigval)
        if self._fresh_JTJ:
            CLOG.debug('%d degenerate of %d total directions' % (
                    delta0.size-rank, delta0.size))
        if subblock is not None:
            delta = np.zeros(self.J.shape[0])
            delta[subblock] = delta0
        else:
            delta = delta0.copy()
        return delta

    def increase_damping(self):
        self.damping *= self.increase_damp_factor

    def decrease_damping(self, undo_decrease=False):
        if undo_decrease:
            self.damping *= self.decrease_damp_factor
        else:
            self.damping /= self.decrease_damp_factor

    def update_param_vals(self, new_vals, incremental=False):
        """
        Updates the current set of parameter values and previous values,
        sets a flag to re-calculate J.

        Parameters
        ----------
            new_vals : numpy.ndarray
                The new values to update to
            incremental : Bool, optional
                Set to True to make it an incremental update relative
                to the old parameters. Default is False
        """
        self._last_vals = self.param_vals.copy()
        if incremental:
            self.param_vals += new_vals
        else:
            self.param_vals = new_vals.copy()
        #And we've updated, so JTJ is no longer valid:
        self._fresh_JTJ = False

    def find_expected_error(self, delta_params='calc'):
        """
        Returns the error expected after an update if the model were linear.

        Parameters
        ----------
            delta_params : {numpy.ndarray, 'calc', or 'perfect'}, optional
                The relative change in parameters. If 'calc', uses update
                calculated from the current damping, J, etc; if 'perfect',
                uses the update calculated with zero damping.

        Returns
        -------
            numpy.float64
                The expected error after the update with `delta_params`
        """
        grad = self.calc_grad()
        if list(delta_params) in [list('calc'), list('perfect')]:
            jtj = (self.JTJ if delta_params == 'perfect' else
                    self._calc_damped_jtj(self.JTJ))
            delta_params = self._calc_lm_step(jtj, self.calc_grad())
        #If the model were linear, then the cost would be quadratic,
        #with Hessian 2*`self.JTJ` and gradient `grad`
        expected_error = (self.error + np.dot(grad, delta_params) +
                np.dot(np.dot(self.JTJ, delta_params), delta_params))
        return expected_error

    def calc_model_cosine(self, decimate=None, mode='err'):
        """
        Calculates the cosine of the residuals with the model.

        Parameters
        ----------
            decimate : Int or None, optional
                Decimate the residuals by `decimate` pixels. If None, no
                decimation is used. Valid only with mode='svd'. Default
                is None
            mode : {'svd', 'err'}
                Which mode to use; see Notes section. Default is 'err'.

        Returns
        -------
            abs_cos : numpy.float64
                The absolute value of the model cosine.

        Notes
        -----
        The model cosine is defined in terms of the geometric view of
        curve-fitting, as a model manifold embedded in a high-dimensional
        space. The model cosine is the cosine of the residuals vector
        with its projection on the tangent space: cos(phi) = |P^T r|/|r|
        where P^T is the projection operator onto the model manifold and r
        the residuals. This can be calculated two ways: By calculating
        the projection operator P directly with SVD (mode=`svd`), or by
        using the expected error if the model were linear to calculate a
        model sine first (mode=`err`). Since the SVD of a large matrix is
        slow, mode=`err` is faster.

        `Decimate' allows for every nth pixel only to be counted in the
        SVD matrix of J for speed. While this is n x faster, it is
        considerably less accurate, so the default is no decimation.
        """
        if mode == 'svd':
            slicer = slice(0, None, decimate)

            #1. Calculate projection term
            u, sig, v = np.linalg.svd(self.J[:,slicer], full_matrices=False) #slow part
            # p = np.dot(v.T, v) - memory error, so term-by-term
            r = self.calc_residuals()[slicer]
            abs_r = np.sqrt((r*r).sum())

            v_r = np.dot(v,r/abs_r)
            projected = np.dot(v.T, v_r)

            abs_cos = np.sqrt((projected*projected).sum())
        elif mode == 'err':
            expected_error = self.find_expected_error(delta_params='perfect')
            model_sine_2 = expected_error / self.error  #error = distance^2
            abs_cos = np.sqrt(1 - model_sine_2)
        else:
            raise ValueError('mode must be one of `svd`, `err`')
        return abs_cos

    def get_termination_stats(self, get_cos=True):
        """
        Returns a dict of termination statistics

        Parameters
        ----------
            get_cos : Bool, optional
                Whether or not to calcualte the cosine of the residuals
                with the tangent plane of the model using the current J.
                The calculation may take some time. Default is True

        Returns
        -------
            dict
                Has keys
                    delta_vals  : The last change in parameter values.
                    delta_err   : The last change in the error.
                    exp_err     : The expected (last) change in the error.
                    frac_err    : The fractional change in the error.
                    num_iter    : The number of iterations completed.
                    error       : The current error.
        """
        delta_vals = self._last_vals - self.param_vals
        delta_err = self._last_error - self.error
        frac_err = delta_err / self.error
        to_return = {'delta_vals':delta_vals, 'delta_err':delta_err,
                'num_iter':1*self._num_iter, 'frac_err':frac_err,
                'error':self.error, 'exp_err':self._exp_err}
        if get_cos:
            model_cosine = self.calc_model_cosine()
            to_return.update({'model_cosine':model_cosine})
        return to_return

    def check_completion(self):
        """
        Returns a Bool of whether the algorithm has found a satisfactory minimum
        """
        terminate = False
        term_dict = self.get_termination_stats(get_cos=self.costol is not None)
        terminate |= np.all(np.abs(term_dict['delta_vals']) < self.paramtol)
        terminate |= (term_dict['delta_err'] < self.errtol)
        terminate |= (term_dict['exp_err'] < self.exptol)
        terminate |= (term_dict['frac_err'] < self.fractol)
        if self.costol is not None:
            terminate |= (curcos < term_dict['model_cosine'])

        return terminate

    def check_terminate(self):
        """
        Returns a Bool of whether to terminate.

        Checks whether a satisfactory minimum has been found or whether
        too many iterations have occurred.
        """
        if not self._has_run:
            return False
        else:
            #1-3. errtol, paramtol, model cosine low enough?
            terminate = self.check_completion()

            #4. too many iterations??
            terminate |= (self._num_iter >= self.max_iter)
            return terminate

    def check_update_J(self):
        """
        Checks if the full J should be updated.

        Right now, just updates after update_J_frequency loops
        """
        self._J_update_counter += 1
        update = self._J_update_counter >= self.update_J_frequency
        return update & (not self._fresh_JTJ)

    def update_J(self):
        """Updates J, JTJ, and internal counters."""
        self.calc_J()
        self.JTJ = np.dot(self.J, self.J.T)
        self._fresh_JTJ = True
        self._J_update_counter = 0
        if np.any(np.isnan(self.J)) or np.any(np.isnan(self.JTJ)):
            raise FloatingPointError('J, JTJ have nans.')
        #Update self._exp_err
        self._exp_err = self.error - self.find_expected_error(delta_params='perfect')

    def calc_grad(self):
        """The gradient of the cost w.r.t. the parameters."""
        residuals = self.calc_residuals()
        return 2*np.dot(self.J, residuals)

    def _rank_1_J_update(self, direction, values):
        """
        Does J += np.outer(direction, new_values - old_values) without
        using lots of memory
        """
        vals_to_sub = np.dot(direction, self.J)
        delta_vals = values - vals_to_sub
        for a in xrange(direction.size):
            self.J[a] += direction[a] * delta_vals

    def check_Broyden_J(self):
        do_update = (self.broyden_update & (not self._fresh_JTJ) &
                ((self._inner_run_counter % self.broyden_update_frequency) == 0))
        return do_update

    def update_Broyden_J(self):
        """Execute a Broyden update of J"""
        CLOG.debug('Broyden update.')
        delta_vals = self.param_vals - self._last_vals
        delta_residuals = self.calc_residuals() - self._last_residuals
        nrm = np.sqrt(np.dot(delta_vals, delta_vals))
        direction = delta_vals / nrm
        vals = delta_residuals / nrm
        self._rank_1_J_update(direction, vals)
        self.JTJ = np.dot(self.J, self.J.T)

    def check_update_eig_J(self):
        do_update = (self.eig_update & (not self._fresh_JTJ) &
                ((self._inner_run_counter % self.eig_update_frequency) == 0))
        return do_update

    def update_eig_J(self):
        """Execute an eigen update of J"""
        CLOG.debug('Eigen update.')
        vls, vcs = np.linalg.eigh(self.JTJ)
        res0 = self.calc_residuals()
        for a in xrange(min([self.num_eig_dirs, vls.size])):
            #1. Finding stiff directions
            stif_dir = vcs[-(a+1)] #already normalized

            #2. Evaluating derivative along that direction, we'll use dl=5e-4:
            dl = self.eig_dl #1e-5
            _ = self.update_function(self.param_vals + dl*stif_dir)
            res1 = self.calc_residuals()

            #3. Updating
            grad_stif = (res1-res0)/dl
            self._rank_1_J_update(stif_dir, grad_stif)

        self.JTJ = np.dot(self.J, self.J.T)
        #Putting the parameters back:
        _ = self.update_function(self.param_vals)

    def calc_accel_correction(self, damped_JTJ, delta0):
        """
        Geodesic acceleration correction to the LM step.

        Parameters
        ----------
            damped_JTJ : numpy.ndarray
                The damped JTJ used to calculate the initial step.
            delta0 : numpy.ndarray
                The initial LM step.

        Returns
        -------
            corr : numpy.ndarray
                The correction to the original LM step.
        """
        #Get the derivative:
        _ = self.update_function(self.param_vals)
        rm0 = self.calc_residuals().copy()
        _ = self.update_function(self.param_vals + delta0)
        rm1 = self.calc_residuals().copy()
        _ = self.update_function(self.param_vals - delta0)
        rm2 = self.calc_residuals().copy()
        der2 = (rm2 + rm1 - 2*rm0)

        corr, res, rank, s = np.linalg.lstsq(damped_JTJ, np.dot(self.J, der2),
                rcond=self.min_eigval)
        corr *= -0.5
        return corr

    def update_select_J(self, blk):
        """
        Updates J only for certain parameters, described by the boolean
        mask `blk`.
        """
        p0 = self.param_vals.copy()
        self.update_function(p0)  #in case things are not put back...
        r0 = self.calc_residuals().copy()
        dl = np.zeros(p0.size, dtype='float')
        blk_J = []
        for i in np.nonzero(blk)[0]:
            dl *= 0; dl[i] = self.eig_dl
            self.update_function(p0 + dl)
            r1 = self.calc_residuals().copy()
            blk_J.append((r1-r0)/self.eig_dl)
        self.J[blk] = np.array(blk_J)
        self.update_function(p0)
        #Then we also need to update JTJ:
        self.JTJ = np.dot(self.J, self.J.T)
        if np.any(np.isnan(self.J)) or np.any(np.isnan(self.JTJ)):
            raise FloatingPointError('J, JTJ have nans.')

class LMFunction(LMEngine):
    """
    Levenberg-Marquardt optimization for a user-supplied function.

    Contains alll the options from the M. Transtrum J. Sethna 2012 ArXiV
    paper. See LMEngine for further documentation.

    Parameters
    -------
        data : N-element numpy.ndarray
            The measured data to fit.
        func : Function
            The function to evaluate. Syntax must be
            func(param_values, *func_args, **func_kwargs), and return a
            numpy.ndarray of the same shape as data
        p0 : P-elemnet numpy.ndarray
            Float array of the initial parameter guess.
        dl : Float or P-element numpy.ndarray, optional
            The dl used for finite-difference derivatives, i.e.
            (f(x+dl[i])) - f(x)) / (dl[i]) in each direction. If dl is
            a scalar, it is transformed internally to a list. Default is
            1e-8.
        func_args : List-like, optional
            Extra *args to pass to the function. Default is ()
        func_kargs : Dictionary, optional
            Extra **kwargs to pass to the function. Default is {}
        **kwargs : Any keyword args passed to LMEngine.

    Attributes
    ----------
        param_vals : numpy.ndarray
            The current best-fit parameter values of the function.
    """
    def __init__(self, data, func, p0, func_args=(), func_kwargs={}, dl=1e-8,
            **kwargs):
        self.data = data
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.param_vals = p0.astype('float')
        if np.size(dl) == 1:
            self.dl = np.ones_like(self.param_vals) * dl
        else:
            self.dl = npj.array(dl)
        super(LMFunction, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        """
        Must update:
            self.error, self._last_error, self.param_vals, self._last_vals
        """
        # self.param_vals = p0 #sloppy...
        self._last_vals = self.param_vals.copy()
        self.error = self.update_function(self.param_vals)
        self._last_error = (1 + 2*self.fractol) * self.error

    def calc_J(self):
        """Updates self.J, returns nothing"""
        del self.J
        self.J = np.zeros([self.param_vals.size, self.data.size])
        dp = np.zeros_like(self.param_vals)
        f0 = self.model.copy()
        for a in xrange(self.param_vals.size):
            dp *= 0
            dp[a] = self.dl[a]
            f1 = self.func(self.param_vals + dp, *self.func_args, **self.func_kwargs)
            grad_func = (f1 - f0) / dp[a]
            #J = grad(residuals) = -grad(model)
            self.J[a] = -grad_func

    def calc_residuals(self):
        return self.data - self.model

    def update_function(self, param_vals):
        """Takes an array param_vals, updates function, returns the new error"""
        self.model = self.func(param_vals, *self.func_args, **self.func_kwargs)
        d = self.calc_residuals()
        return np.dot(d.flat, d.flat) #faster for large arrays than (d*d).sum()

class LMOptObj(LMEngine):
    """
    Levenberg-Marquardt optimization on an OptObj instance.

    Parameters
    ----------
        opt_obj : OptObj or daughter instance
            The OptObj to optimize.
    **kwargs Parameters
    -------------------
        Any kwargs for LMEngine.

    Methods
    -------
        _set_err_paramvals()
            Helps initialize the LMEngine.
        calc_J()
        calc_residuals()
        update_function()
            Updates the opt_obj, returns new error.

    See Also
    --------
        LMEngine
        OptObj
        OptState

    Notes
    -----
    Uses an OptObj instance.... should be the syntax for all LMEngine objects?
    """
    def __init__(self, opt_obj, **kwargs):
        self.opt_obj = opt_obj
        super(LMOptObj, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        self.param_vals = self.opt_obj.param_vals.copy()
        self._last_vals = self.param_vals.copy()
        self.error = self.opt_obj.get_error()
        self._last_error = (1 + 2*self.fractol) * self.error + 2*self.errtol

    def calc_J(self):
        del self.J
        self.J = self.opt_obj.calc_J()

    def calc_residuals(self):
        return self.opt_obj.calc_residuals()

    def update_function(self, param_vals):
        """Updates the opt_obj, returns new error."""
        self.opt_obj.update_function(param_vals)
        return self.opt_obj.get_error()

class OptObj(object):
    """Basically an empty class; just laying out the structure for any daughters."""
    def __init__(self, param_vals):
        self.param_vals = param_vals
        pass

    def calc_J(self):
        pass
    def calc_residuals(self):
        pass

    def get_error(self): #@property?
        pass

    def update_function(self, param_vals):
        pass
        return None

class OptState(OptObj):
    """
    A wrapper for a :class:`peri.states.State` instance which allows for
    optimization along any set of directions.

    Parameters
    ----------
        state : :class:`peri.states.State`
            The state to optimize
        directions : numpy.ndarray
            [M,N] element array of the M tangent vectors determining
            the plane, each with N dimensions.
        p0 : numpy.ndarray or None, optional
            The optimization is done on a hyperplane spanned by the
            tangent vectors `directions` and passing through the point
            `p0`. If `p0` is `None`, then `p0` is set to the current
            state paramters. Default is None
        dl : Float, optional
            The step size for finite-differencing the derivative. Default
            is 1e-7

    Attributes
    ----------
        param_vals : numpy.ndarray
            The parameter values of the distance moved from `p0` along
            each of the tangent vectors.

    Methods
    -------
        update_function(param_vals)
            Update the state to `param_vals` on the hyperplane
        get_error()
            Returns self.state.error
        calc_residuals()
            Returns self.state.residuals.ravel()
        calc_J()
            Calculates J for the state as parameterized by the directions.

    See Also
    --------
        LMOptObj
        do_levmarq_n_directions
    """
    #FIXME should have an allowed set of params, and then R(z) etc
    #parameterizations should be incorporated into this to eliminate
    #AugmentedState (or turn it into a subclass)
    def __init__(self, state, directions, p0=None, dl=1e-7):
        self.state = state
        self.dl = dl
        if p0 is None:
            self.p0 = np.array(state.state[state.params]).copy()
        else:
            self.p0 = p0.copy()
            if p0.size != np.size(state.state[state.params]):
                raise ValueError('direction must have same # of elements as state.size')
        self.directions = np.array(directions)
        self.param_vals = np.zeros(self.directions.shape[0])

    def update_function(self, param_vals):
        """Updates with param_vals[i] = distance from self.p0 along self.direction[i]."""
        dp = np.zeros(self.p0.size)
        for a in xrange(param_vals.size):
            dp += param_vals[a] * self.directions[a]
        self.state.update(self.state.params, self.p0 + dp)
        self.param_vals[:] = param_vals
        return None

    def get_error(self):
        return self.state.error

    def calc_residuals(self):
        """
        See also
        --------
        :func:`peri.state.States.residuals`
        """
        return self.state.residuals.ravel().copy()

    def calc_J(self):
        """Calculates J along the direction."""
        r0 = self.state.residuals.copy().ravel()
        dl = np.zeros(self.param_vals.size)
        p0 = self.param_vals.copy()
        J = []
        for a in xrange(self.param_vals.size):
            dl *= 0
            dl[a] += self.dl
            self.update_function(p0 + dl)
            r1 = self.state.residuals.copy().ravel()
            J.append( (r1-r0)/self.dl)
        self.update_function(p0)
        return np.array(J)

class LMGlobals(LMEngine):
    """
    Levenberg-Marquardt, optimized for state globals.

    Contains alll the options from the M. Transtrum J. Sethna 2012 ArXiV
    paper. See LMEngine for further documentation.

    Parameters
    ----------
        state : :class:`peri.states.State`
            The state to optimize
        param_names : List
            List of the parameter names (strings) to optimize over
        max_mem : Numeric, optional
            The maximum memory to use for the optimization; controls pixel
            decimation. Default is 1e9.
        opt_kwargs : Dict, optional
            Dict of **kwargs for opt implementation. Right now only for
            *.get_num_px_jtj, i.e. keys of 'decimate', 'min_redundant'.
            Default is `{}`

    Attributes
    ----------
        state : :class:`peri.states.State`
            The state to optimize.
        opt_kwargs : dict
            A **kwargs-like dictionary for optimization implementation.
        max_mem : Float or int
            The max memory occupied by J.
        num_pix : Int
            The number of pixels of the residuals used to calculate J.
        param_names : List
            The list of the parameter names being optimized.

    Methods
    -------
        set_params(new_param_names, new_damping=None)
            Change the parameter names to optimize.
        reset(new_damping=None)
            Resets counters etc to zero, allowing more runs to commence.

    Other Parameters, Attributes, and Methods
    -----------------------------------------
        See LMEngine

    See Also
    --------
        LMParticles
        LMAugmnetedState
        do_levmarq
        do_levmarq_particles
    """
    def __init__(self, state, param_names, max_mem=1e9, opt_kwargs={}, **kwargs):
        self.state = state
        self.opt_kwargs = opt_kwargs
        self.max_mem = max_mem
        self.num_pix = get_num_px_jtj(state, len(param_names), max_mem=max_mem,
                **self.opt_kwargs)
        self.param_names = param_names
        super(LMGlobals, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        self.error = self.state.error
        self._last_error = (1 + 2*self.fractol) * self.state.error
        self.param_vals = np.ravel(self.state.state[self.param_names])
        self._last_vals = self.param_vals.copy()

    def calc_J(self):
        del self.J
        self.J, self._inds = get_rand_Japprox(self.state,
                self.param_names, num_inds=self.num_pix)

    def calc_residuals(self):
        return self.state.residuals.ravel()[self._inds].copy()

    def update_function(self, values):
        self.state.update(self.param_names, values)
        if np.any(np.isnan(self.state.residuals)):
            raise FloatingPointError('state update caused nans in residuals')
        return self.state.error

    def set_params(self, new_param_names, new_damping=None):
        self.param_names = new_param_names
        self._set_err_paramvals()
        self.reset(new_damping=new_damping)

    def update_select_J(self, blk):
        """
        Updates J only for certain parameters, described by the boolean
        mask blk.
        """
        self.update_function(self.param_vals)
        params = np.array(self.param_names)[blk].tolist()
        blk_J = -self.state.gradmodel(params=params, inds=self._inds, flat=False)
        self.J[blk] = blk_J
        #Then we also need to update JTJ:
        self.JTJ = np.dot(self.J, self.J.T)
        if np.any(np.isnan(self.J)) or np.any(np.isnan(self.JTJ)):
            raise FloatingPointError('J, JTJ have nans.')

class LMParticles(LMEngine):
    """
    Levenberg-Marquardt, optimized for state globals.

    Contains alll the options from the M. Transtrum J. Sethna 2012 ArXiV
    paper. See LMEngine for further documentation.

    Parameters
    ----------
        state : :class:`peri.states.ImageState`
            The state to optimize
        particles : numpy.ndarray
            Array of the particle indices to optimize over.
        include_rad : Bool, optional
            Whether or not to include the particle radii in the
            optimization. Default is True

    Attributes
    ----------
        state : :class:`peri.states.ImageState`
            The state to optimize.
        particles : dict
            A **kwargs-like dictionary for optimization implementation.
        param_names : List
            The list of the parameter names being optimized.

    Methods
    -------
        set_particles(new_particles, new_damping=None)
            Change the particle to optimize.
        reset(new_damping=None)
            Resets counters etc to zero, allowing more runs to commence.

    Other Parameters, Attributes, and Methods
    -----------------------------------------
        See LMEngine

    See Also
    --------
        LMParticles
        LMAugmnetedState
        do_levmarq
        do_levmarq_particles

    Notes
    -----
        To prevent the state updates from breaking, this clips the
        particle rads to [self._MINRAD, self._MAXRAD] and the positions
        to at least self._MINDIST from the edge of the padded image.
        These are:
            _MINRAD  : 1e-3
            _MAXRAD  : 2e2
            _MINDIST : 1e-3
        For extremely large particles (e.g. larger than _MAXRAD or larger
        than the pad and barely overlapping the image) these numbers might
        be insufficient.
    """
    def __init__(self, state, particles, include_rad=True, **kwargs):
        self.state = state
        if len(particles) == 0:
            raise ValueError('Empty list of particle indices')
        self.particles = particles
        self.param_names = (state.param_particle(particles) if include_rad
                else state.param_particle_pos(particles))
        self._dif_tile = self._get_diftile()
        #Max, min rads, distance from edge for allowed updates
        self._MINRAD = 1e-3
        self._MAXRAD = 2e2
        self._MINDIST= 1e-3

        #is_rad, is_pos masks:
        rad_nms = self.state.param_radii()
        self._is_rad = np.array(map(lambda x: x in rad_nms, self.param_names))
        pos_nms = self.state.param_positions()
        self._is_pos = []
        for a in xrange(3):
            self._is_pos.append(np.array(map(lambda x: (x in pos_nms) &
                    (x[-1] == 'zyx'[a]), self.param_names)))
        super(LMParticles, self).__init__(**kwargs)

    def _get_diftile(self):
        vals = np.ravel(self.state.state[self.param_names])
        itile = self.state.get_update_io_tiles(self.param_names, vals)[1]
        return get_residuals_update_tile(self.state, itile)

    def _set_err_paramvals(self):
        self.error = self.state.error
        self._last_error = (1 + 2*self.fractol) * self.state.error
        self.param_vals = np.ravel(self.state.state[self.param_names])
        self._last_vals = self.param_vals.copy()

    def calc_J(self):
        self._dif_tile = self._get_diftile()
        del self.J
        #J = grad(residuals) = -grad(model)
        if self._dif_tile.volume > 0:
            self.J = -self.state.gradmodel(params=self.param_names, rts=True,
                slicer=self._dif_tile.slicer)
        else:
            self.J = np.zeros([len(self.param_names), 1])

    def calc_residuals(self):
        if self._dif_tile.volume > 0:
            return self.state.residuals[self._dif_tile.slicer].ravel().copy()
        else:
            return np.zeros(1)

    def update_function(self, values):
        #1. Clipping values:
        values[self._is_rad] = np.clip(values[self._is_rad], self._MINRAD,
                self._MAXRAD)
        pd = self.state.pad
        for a in xrange(3):
            values[self._is_pos[a]] = np.clip(values[self._is_pos[a]],
                    self._MINDIST - pd[a], self.state.ishape.shape[a] +
                    pd[a] - self._MINDIST)

        self.state.update(self.param_names, values)
        if np.any(np.isnan(self.state.residuals)):
            raise FloatingPointError('state update caused nans in residuals')
        return self.state.error

    def set_particles(self, new_particles, new_damping=None):
        self.particles = new_particles
        self.param_names = (state.param_particle(particles) if include_rad
                else state.param_particle_pos(particles))
        self._dif_tile = self._get_diftile()
        self._set_err_paramvals()
        self.reset(new_damping=new_damping)

class LMParticleGroupCollection(object):
    """
    Levenberg-Marquardt on all particles in a state.

    Convenience wrapper for LMParticles. Generates a separate instance
    for the particle groups each time and optimizes with that, since
    storing J for the particles is too large.

    Parameters
    ----------
        state : :class:`peri.states.ImageState`
            The state to optimize
        region_size : Int or 3-element list-like of ints, optional
            The region size for sub-blocking particles. Default is 40
        do_calc_size : Bool, optional
            If True, calculates the region size internally based on
            the maximum allowed memory. Default is True
        max_mem : Numeric, optional
            The maximum allowed memory for J to occupy. Default is 1e9
        get_cos : Bool, optional
            Set to True to include the model cosine in the statistics
            on each individual group's run, using `LMEngine`
            get_termination_stats(). Stored in self.stats. Default is
            False
        save_J : Bool
            Set to True to create a series of temp files that save J
            for each group of particles. Needed for do_internal_run().
            Default is False.
        **kwargs :
            Pass any kwargs that would be passed to LMParticles.
            Stored in self._kwargs for reference.

    Attributes
    ----------
        stats : List
            A list of the termination stats for each sub-block of particles
        particle_groups : List
            A list of particle groups. Element [i] in the list is a
            numpy.ndarray of the indices in group [i].
        region_size : Int or 3-element list-like
            The region size of the tiles. If `do_calc_size` is True,
            region_size will be the calculated value, which may differ
            from the input value.
        _kwargs : Dict
            The **kwargs passed to LMParticles.

    Methods
    -------
        reset()
            Re-calculate all the groups
        do_run_1()
            Run do_run_1 for every group of particles
        do_run_2()
            Run do_run_2 for every group of particles
        do_internal_run()
            Run do_internal_run for every group of particles

    See Also
    -------
        LMParticles

    Notes
    -----
    Since storing J for many particles can require a huge amount of
    memory, this object proceeds by re-initializing a separate LMParticles
    instance for each group of particles, calculating and then discarding
    J each time. The calculated J's can be kept by setting `save_J` to
    True, which saves each J for each group in a separate tempfile,
    located in the current directory. The J's can then be loaded again to
    attempt a second step without re-calculating J. However, for a big
    image this can attempt to store _a_lot_ of temp files, which might be
    more than the operating system limit (as temp files are always open),
    which will raise an error. So use with caution. Deleting any
    references to the temp files by deleting the LMParticleGroupCollection
    instance will close and remove the temporary files.
    """
    def __init__(self, state, region_size=40, do_calc_size=True, max_mem=1e9,
            get_cos=False, save_J=False, **kwargs):
        self.state = state
        self._kwargs = kwargs
        self.region_size = region_size
        self.get_cos = get_cos
        self.save_J = save_J
        self.max_mem = max_mem

        self.reset(do_calc_size=do_calc_size)

    def reset(self, new_region_size=None, do_calc_size=True, new_damping=None,
            new_max_mem=None):
        """
        Resets the particle groups and optionally the region size and damping.

        Parameters
        ----------
            new_region_size : : Int or 3-element list-like of ints, optional
                The region size for sub-blocking particles. Default is 40
            do_calc_size : Bool, optional
                If True, calculates the region size internally based on
                the maximum allowed memory. Default is True
            new_damping : Float or None, optional
                The new damping of the optimizer. Set to None to leave
                as the default for LMParticles. Default is None.
            new_max_mem : Numeric, optional
                The maximum allowed memory for J to occupy. Default is 1e9
        """
        if new_region_size is not None:
            self.region_size = new_region_size
        if new_max_mem != None:
            self.max_mem = new_max_mem
        if do_calc_size:
            self.region_size = calc_particle_group_region_size(self.state,
                    region_size=self.region_size, max_mem=self.max_mem)
        self.stats = []
        self.particle_groups = separate_particles_into_groups(self.state,
                self.region_size)
        if new_damping is not None:
            self._kwargs.update({'damping':new_damping})
        if self.save_J:
            if len(self.particle_groups) > 90:
                CLOG.warn('Attempting to create many open files. Consider increasing max_mem and/or region_size to avoid crashes.')
            self._tempfiles = []
            self._has_saved_J = []
            for a in xrange(len(self.particle_groups)):
                #TemporaryFile is automatically deleted
                for _ in ['j','tile']:
                    self._tempfiles.append(tempfile.TemporaryFile(dir=os.getcwd()))
                self._has_saved_J.append(False)

    def _get_tmpfiles(self, group_index):
        j_file = self._tempfiles[2*group_index]
        tile_file = self._tempfiles[2*group_index+1]
        #And we rewind before we return:
        j_file.seek(0)
        tile_file.seek(0)
        return j_file, tile_file

    def _dump_j_diftile(self, group_index, j, tile):
        j_file, tile_file = self._get_tmpfiles(group_index)
        np.save(j_file, j)
        pickle.dump(tile, tile_file)

    def _load_j_diftile(self, group_index):
        j_file, tile_file = self._get_tmpfiles(group_index)
        J = np.load(j_file)
        tile = pickle.load(tile_file)
        JTJ = np.dot(J, J.T)
        return J, JTJ, tile

    def _do_run(self, mode='1'):
        """workhorse for the self.do_run_xx methods."""
        for a in xrange(len(self.particle_groups)):
            group = self.particle_groups[a]
            lp = LMParticles(self.state, group, **self._kwargs)
            if mode == 'internal':
                lp.J, lp.JTJ, lp._dif_tile = self._load_j_diftile(a)

            if mode == '1':
                lp.do_run_1()
            if mode == '2':
                lp.do_run_2()
            if mode == 'internal':
                lp.do_internal_run()

            self.stats.append(lp.get_termination_stats(get_cos=self.get_cos))
            if self.save_J and (mode != 'internal'):
                self._dump_j_diftile(a, lp.J, lp._dif_tile)
                self._has_saved_J[a] = True

    def do_run_1(self):
        """Calls LMParticles.do_run_1 for each group of particles."""
        self._do_run(mode='1')

    def do_run_2(self):
        """Calls LMParticles.do_run_2 for each group of particles."""
        self._do_run(mode='2')

    def do_internal_run(self):
        """Calls LMParticles.do_internal_run for each group of particles."""
        if not self.save_J:
            raise RuntimeError('self.save_J=True required for do_internal_run()')
        if not np.all(self._has_saved_J):
            raise RuntimeError('J, JTJ have not been pre-computed. Call do_run_1 or do_run_2')
        self._do_run(mode='internal')

class AugmentedState(object):
    """
    Augments a state with a set of radii(z) parameters.

    Operates by updating the radii as R -> R0*np.exp(legval(zp)), where
    zp is a rescaled z coordinate. The order of the Legendre polynomial
    for the rescaling is set by rz_order

    Paramters
    ---------
        state : :class:`peri.states.ImageState`
            The state to augment.
        param_names : list
            The list of the parameter names to include in the augmented
            state. Can contain any parameters except the particle radii
        rz_order : Int, optional
            The order of the Legendre polynomial used for rescaling.
            Default is 3.

    Methods
    -------
        reset()
            Resets the augmented state by resetting the initial positions
            and radii used for updating the particles. Use if any pos or
            rad has been updated outside of the augmented state.
        update(param_vals)
            Updates the augmented state


    See Also
    --------
        LMAugmentedState

    Notes
    -----
        This could be extended to do xyzshift(xyz), radii(xyz)
    """
    def __init__(self, state, param_names, rz_order=3):
        rad_nms = state.param_radii()
        has_rad = map(lambda x: x in param_names, rad_nms)
        if np.any(has_rad):
            raise ValueError('param_names must not contain any radii.')

        self.state = state
        self.param_names = param_names
        self.n_st_params = len(param_names)
        self.rz_order = rz_order

        #Controling which params are globals, which are r(xyz) parameters
        globals_mask = np.zeros(self.n_st_params + rz_order, dtype='bool')
        globals_mask[:self.n_st_params] = True
        rscale_mask = -globals_mask
        self.globals_mask = globals_mask
        self.rscale_mask = rscale_mask

        param_vals = np.zeros(globals_mask.size, dtype='float')
        param_vals[:self.n_st_params] = np.copy(self.state.state[param_names])
        self.param_vals = param_vals
        self.reset()

    def reset(self):
        """
        Resets the initial radii used for updating the particles. Call
        if any of the particle radii or positions have been changed
        external to the augmented state.
        """
        inds = range(self.state.obj_get_radii().size)
        self._rad_nms = self.state.param_particle_rad(inds)
        self._pos_nms = self.state.param_particle_pos(inds)
        self._initial_rad = np.copy(self.state.state[self._rad_nms])
        self._initial_pos = np.copy(self.state.state[self._pos_nms]).reshape((-1,3))
        self.param_vals[self.rscale_mask] = 0

    def rad_func(self, pos):
        """Right now exp(self._poly(z))"""
        return np.exp(self._poly(pos[:,2]))

    def _poly(self, z):
        """Right now legval(z)"""
        shp = self.state.oshape.shape
        zmax = float(shp[0])
        zmin = 0.0
        zmid = zmax * 0.5

        coeffs = self.param_vals[self.rscale_mask].copy()
        if coeffs.size == 0:
            ans = 0*z
        else:
            ans = np.polynomial.legendre.legval((z-zmid)/zmid,
                    self.param_vals[self.rscale_mask])
        return ans

    def update(self, param_vals):
        """Updates all the parameters of the state + rscale(z)"""
        self.update_rscl_x_params(param_vals[self.rscale_mask], do_reset=False)
        self.state.update(self.param_names, param_vals[self.globals_mask])
        self.param_vals[:] = param_vals.copy()
        if np.any(np.isnan(self.state.residuals)):
            raise FloatingPointError('state update caused nans in residuals')

    def update_rscl_x_params(self, new_rscl_params, do_reset=True):
        #1. What to change:
        p = self._initial_pos

        #2. New, old values:
        self.param_vals[self.rscale_mask] = new_rscl_params
        new_scale = self.rad_func(p)

        rnew = self._initial_rad * new_scale
        if do_reset:
            self.state.update(self._rad_nms, rnew)
        else:
            #FIXME you can do this without the extra convolution if you pass
            #all at once... right now don't worry about it
            self.state.update(self._rad_nms, rnew)

class LMAugmentedState(LMEngine):
    """
    Levenberg-Marquardt on an augmented state.

    Contains alll the options from the M. Transtrum J. Sethna 2012 ArXiV
    paper. See LMEngine for further documentation.

    Parameters
    ----------
        aug_state : AugmentedState
            The state to optimize
        max_mem : Numeric, optional
            The maximum memory to use for the optimization; controls pixel
            decimation. Default is 1e9.
        opt_kwargs : Dict, optional
            Dict of **kwargs for opt implementation. Right now only for
            *.get_num_px_jtj, i.e. keys of 'decimate', min_redundant'.
            Default is `{}`

    Attributes
    ----------
        aug_state : AugmentedState
            The augmented state to optimize.
        opt_kwargs : dict
            A **kwargs-like dictionary for optimization implementation.
        max_mem : Float or int
            The max memory occupied by J.
        num_pix : Int
            The number of pixels of the residuals used to calculate J.

    Methods
    -------
        reset(new_damping=None)
            Resets the augmented state, counters, etc to zero, allowing
            more runs to commence.

    Other Parameters, Attributes, and Methods
    -----------------------------------------
        See LMEngine

    See Also
    --------
        LMParticles
        LMGlobals
        AugmentedState
        do_levmarq
        do_levmarq_particles
    """
    def __init__(self, aug_state, max_mem=1e9, opt_kwargs={}, **kwargs):
        self.aug_state = aug_state
        self.opt_kwargs = opt_kwargs
        self.max_mem = max_mem
        self.num_pix = get_num_px_jtj(aug_state.state, aug_state.param_vals.size,
                max_mem=max_mem, **self.opt_kwargs)
        super(LMAugmentedState, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        self.error = self.aug_state.state.error
        self._last_error = (1 + 2*self.fractol) * self.aug_state.state.error
        self.param_vals = self.aug_state.param_vals.copy()
        self._last_vals = self.param_vals.copy()

    def calc_J(self):
        #0.
        del self.J

        #1. J for the state:
        s = self.aug_state.state
        sa = self.aug_state
        J_st, inds = get_rand_Japprox(s, self.aug_state.param_names,
                num_inds=self.num_pix, **self.opt_kwargs)
        self._inds = inds

        #2. J for the augmented portion:
        old_aug_vals = sa.param_vals[sa.rscale_mask].copy()
        dl = 1e-6
        J_aug = []
        i0 = s.residuals
        for a in xrange(old_aug_vals.size):
            dx = np.zeros(old_aug_vals.size)
            dx[a] = dl
            sa.update_rscl_x_params(old_aug_vals + dl, do_reset=True)
            i1 = s.residuals
            #J = grad(residuals)
            der = (i1-i0)/dl
            J_aug.append(der.ravel()[self._inds].copy())

        if J_st.size == 0:
            self.J = np.array(J_aug)
        elif old_aug_vals.size == 0:
            self.J = J_st
        else:
            self.J = np.append(J_st, np.array(J_aug), axis=0)

    def calc_residuals(self):
        return self.aug_state.state.residuals.ravel()[self._inds]

    def update_function(self, params):
        self.aug_state.update(params)
        return self.aug_state.state.error

    def reset(self, **kwargs):
        """Resets the aug_state and the LMEngine"""
        self.aug_state.reset()
        super(LMAugmentedState, self).reset(**kwargs)

#=============================================================================#
#         ~~~~~             Convenience Functions             ~~~~~
#=============================================================================#
def do_levmarq(s, param_names, damping=0.1, decrease_damp_factor=10.,
        run_length=6, eig_update=True, collect_stats=False, rz_order=0,
        run_type=2, **kwargs):
    """
    Runs Levenberg-Marquardt optimization on a state.

    Convenience wrapper for LMGlobals. Same keyword args, but the defaults
    have been set to useful values for optimizing globals.
    See LMGlobals and LMEngine for documentation.

    See Also
    --------
        do_levmarq_particles : Levenberg-Marquardt optimization of a
            specified set of particles.

        do_levmarq_all_particle_groups : Levenberg-Marquardt optimization
            of all the particles in the state.

        LMGlobals : Optimizer object; the workhorse of do_levmarq.

        LMEngine : Engine superclass for all the optimizers.
    """
    if rz_order > 0:
        aug = AugmentedState(s, param_names, rz_order=rz_order)
        lm = LMAugmentedState(aug, damping=damping, run_length=run_length,
                decrease_damp_factor=decrease_damp_factor, eig_update=
                eig_update, **kwargs)
    else:
        lm = LMGlobals(s, param_names, damping=damping, run_length=run_length,
                decrease_damp_factor=decrease_damp_factor, eig_update=
                eig_update, **kwargs)
    if run_type == 2:
        lm.do_run_2()
    elif run_type == 1:
        lm.do_run_1()
    else:
        raise ValueError('run_type=1,2 only')
    if collect_stats:
        return lm.get_termination_stats()

def do_levmarq_particles(s, particles, damping=1.0, decrease_damp_factor=10.,
        run_length=4, collect_stats=False, max_iter=2, **kwargs):
    """
    Levenberg-Marquardt optimization on a set of particles.

    Convenience wrapper for LMParticles. Same keyword args, but the
    defaults have been set to useful values for optimizing particles.
    See LMParticles and LMEngine for documentation.

    See Also
    --------
        do_levmarq_all_particle_groups : Levenberg-Marquardt optimization
            of all the particles in the state.

        do_levmarq : Levenberg-Marquardt optimization of the entire state;
            useful for optimizing global parameters.

        LMParticles : Optimizer object; the workhorse of do_levmarq_particles.

        LMEngine : Engine superclass for all the optimizers.
    """
    lp = LMParticles(s, particles, damping=damping, run_length=run_length,
            decrease_damp_factor=decrease_damp_factor, max_iter=max_iter,
            **kwargs)
    lp.do_run_2()
    if collect_stats:
        return lp.get_termination_stats()

def do_levmarq_all_particle_groups(s, region_size=40, max_iter=2, damping=1.0,
        decrease_damp_factor=10., run_length=4, collect_stats=False, **kwargs):
    """
    Levenberg-Marquardt optimization for every particle in the state.

    Convenience wrapper for LMParticleGroupCollection. Same keyword args,
    but I've set the defaults to what I've found to be useful values for
    optimizing particles. See LMParticleGroupCollection for documentation.

    See Also
    --------
        do_levmarq_particles : Levenberg-Marquardt optimization of a
            specified set of particles.

        do_levmarq : Levenberg-Marquardt optimization of the entire state;
            useful for optimizing global parameters.

        LMParticleGroupCollection : The workhorse of do_levmarq.

        LMEngine : Engine superclass for all the optimizers.
    """
    lp = LMParticleGroupCollection(s, region_size=region_size, damping=damping,
            run_length=run_length, decrease_damp_factor=decrease_damp_factor,
            get_cos=collect_stats, max_iter=max_iter, **kwargs)
    lp.do_run_2()
    if collect_stats:
        return lp.stats

def do_levmarq_n_directions(s, directions, max_iter=2, run_length=2,
        damping=1e-3, collect_stats=False, marquardt_damping=True, **kwargs):
    """
    Optimization of a state along a specific set of directions in parameter
    space.

    Parameters
    ----------
        s : :class:`peri.states.State`
            The state to optimize
        directions : np.ndarray
            [n,d] element numpy.ndarray of the n directions in the d-
            dimensional space to optimize along. `directions` is trans-
            formed to a unit vector internally
        The rest are the same **kwargs in LMEngine.
    """
    # normal = direction / np.sqrt(np.dot(direction, direction))
    normals = np.array([d/np.sqrt(np.dot(d,d)) for d in directions])
    obj = OptState(s, normals)
    lo = LMOptObj(obj, max_iter=max_iter, run_length=run_length, damping=
            damping, marquardt_damping=marquardt_damping, **kwargs)
    lo.do_run_1()
    if collect_stats:
        return lo.get_termination_stats()

def burn(s, n_loop=6, collect_stats=False, desc='', rz_order=0, fractol=1e-7,
        errtol=1e-3, mode='burn', max_mem=1e9, include_rad=True,
        do_line_min='default', partial_log=False):
    """
    Optimizes all the parameters of a state.

    Burns a state through calling LMParticleGroupCollection and LMGlobals/
    LMAugmentedState.

    Parameters
    ----------
        s : :class:`peri.states.ImageState`
            The state to optimize
        n_loop : Int, optional
            The number of times to loop over in the optimizer. Default is 6.
        collect_stats : Bool, optional
            Whether or not to collect information on the optimizer's
            performance. Default is False, because True tends to increase
            the memory usage above max_mem.
        desc : string, optional
            Description to append to the states.save() call every loop.
            Set to None to avoid saving. Default is '', which selects
            one of 'burning', 'polishing', 'doing_positions'
        rz_order: Int, optional
            Set to an int > 0 to optimize with an augmented state (R(z) as
            a global parameter) vs. with the normal global parameters;
            rz_order is the order of the polynomial approximate for R(z).
            Default is 0 (no augmented state).
        fractol : Float, optional
            Fractional change in error at which to terminate. Default 1e-7
        errtol : Float, optional
            Absolute change in error at which to terminate. Default 1e-3
        mode : {'burn', 'do-particles', or 'polish'}, optional
            What mode to optimize with.
                'burn'          : Your state is far from the minimum.
                'do-particles'  : Positions are far from the minimum,
                                  globals are well-fit.
                'polish'        : The state is close to the minimum.
            'burn' is the default. Only `polish` will get to the global
            minimum.
        max_mem : Numeric
            The maximum amount of memory allowed for the optimizers' J's,
            for both particles & globals. Default is 1e9, i.e. 1GB per
            optimizer.
        do_line_min : Bool or 'default'.
            Set to True to do an additional, third optimization per loop
            which optimizes along the subspace spanned by the last 3 steps
            of the burn()'s trajectory. In principle this should signifi-
            cantly speed up the convergence; in practice it sometimes does,
            sometimes doesn't. Default is 'default', which picks by mode:
                'burn'          : False
                'do-particles'  : False
                'polish'        : True

    Notes
    -----
        Proceeds by alternating between one Levenberg-Marquardt step
    optimizing the globals, one optimizing the particles, and repeating
    until termination.
        In addition, if `do_line_min` is True, at the end of each loop
    step an additional optimization is tried along the subspaced spanned
    by the steps taken during the last 3 loops. Ideally, this changes the
    convergence from linear to quadratic, but it doesn't always do much.
        Each of the 3 options proceed by optimizing as follows:
    burn            : lm.do_run_2(), lp.do_run_2(). No psf, 2 loops on lm.
    do-particles    : lp.do_run_2(), scales for ilm, bkg's
    polish          : lm.do_run_2(), lp.do_run_2(). Everything, 1 loop each.
    where lm is a globals LMGlobals instance, and lp a
    LMParticleGroupCollection instance.
        It would be nice if some of these magic #'s (region size,
    num_eig_dirs, etc) were calculated in a good way. FIXME
    """
    mode = mode.lower()
    if mode not in {'burn', 'do-particles', 'polish'}:
        raise ValueError('mode must be one of burn, do-particles, polish')

    #1. Setting Defaults
    if desc is '':
        desc = mode + 'ing' if mode != 'do-particles' else 'doing-particles'

    eig_update = (mode != 'do-particles')
    glbl_run_length = 3 if mode == 'do-particles' else 6
    glbl_mx_itr = 2 if mode == 'burn' else 1
    use_accel = (mode == 'burn')
    rz_order = int(rz_order)
    if do_line_min == 'default':
        do_line_min = (mode == 'polish')

    if mode == 'do-particles':
        glbl_nms = ['ilm-scale', 'offset']  #bkg?
    else:
        remove_params = None if mode == 'polish' else set(
                s.get('psf').params + ['zscale'])
        glbl_nms = name_globals(s, remove_params=remove_params)

    all_lp_stats = []
    all_lm_stats = []
    all_line_stats = []
    all_loop_values = []

    _delta_vals = []  #storing the directions we've moved along for line min
    #2. Optimize
    CLOG.info('Start of loop %d:\t%f' % (0, s.error))
    for a in xrange(n_loop):
        start_err = s.error
        start_params = np.copy(s.state[s.params])
        #2a. Globals
        # glbl_dmp = 0.3 if a == 0 else 3e-2
        ####FIXME we damp degenerate but convenient spaces in the ilm, bkg
        ####manually, but we should do it more betterer.
        BAD_DAMP = 1e7
        BAD_LIST = [['ilm-scale', BAD_DAMP], ['ilm-off', BAD_DAMP], ['ilm-z-0',
                BAD_DAMP], ['bkg-z-0', BAD_DAMP]]
        ####
        glbl_dmp = vectorize_damping(glbl_nms + ['rz']*rz_order, damping=1.0,
                increase_list=[['psf-', 3e1]] + BAD_LIST)
        if a != 0 or mode != 'do-particles':
            if partial_log:
                log.set_level('debug')
            gstats = do_levmarq(s, glbl_nms, max_iter=glbl_mx_itr, run_length=
                    glbl_run_length, eig_update=eig_update, num_eig_dirs=10,
                    eig_update_frequency=3, rz_order=rz_order, damping=
                    glbl_dmp, decrease_damp_factor=10., use_accel=use_accel,
                    collect_stats=collect_stats, fractol=0.1*fractol,
                    max_mem=max_mem)
            if partial_log:
                log.set_level('info')
            all_lm_stats.append(gstats)
        if desc is not None:
            states.save(s, desc=desc)
        CLOG.info('Globals, loop %d:\t%f' % (a, s.error))
        all_loop_values.append(s.values)

        #2b. Particles
        prtl_dmp = 1.0 if a==0 else 1e-2
        #For now, I'm calculating the region size. This might be a bad idea
        #because 1 bad particle can spoil the whole group.
        pstats = do_levmarq_all_particle_groups(s, region_size=40, max_iter=1,
                do_calc_size=True, run_length=4, eig_update=False,
                damping=prtl_dmp, fractol=0.1*fractol, collect_stats=
                collect_stats, max_mem=max_mem, include_rad=include_rad)
        all_lp_stats.append(pstats)
        if desc is not None:
            states.save(s, desc=desc)
        CLOG.info('Particles, loop %d:\t%f' % (a, s.error))
        gc.collect()
        all_loop_values.append(s.values)

        #2c. Line min?
        end_params = np.copy(s.state[s.params])
        _delta_vals.append(start_params - end_params)
        if do_line_min:
            all_line_stats.append(do_levmarq_n_directions(s, _delta_vals[-3:],
                    collect_stats=collect_stats))
            if desc is not None:
                states.save(s, desc=desc)
            CLOG.info('Line min, loop %d:\t%f' % (a, s.error))
            all_loop_values.append(s.values)

        #2d. terminate?
        new_err = s.error
        derr = start_err - new_err
        if (derr/new_err < fractol) or (derr < errtol):
            break

    if collect_stats:
        return all_lp_stats, all_lm_stats, all_line_stats, all_loop_values

def fit_comp(new_comp, old_comp, **kwargs):
    """
    Fits a new component to an old component

    Calls do_levmarq to match the .get() fields of the two objects. The
    parameters of new_comp are modified in place.

    Parameters
    ----------
    new_comp : :class:`peri.comps.comp`
        The new object, whose parameters to update to fit the field of
        `old_comp`. Must have a .get() attribute which returns an ndarray
    old_comp : peri.comp
        The old ilm to match to.

    **kwargs Parameters
    -------------------
        Any keyword arguments to be passed to the optimizer LMGlobals
        through do_levmarq.

    See Also
    --------
    do_levmarq : Levenberg-Marquardt minimization using a random subset
        of the image pixels.
    """
    fake_s = states.ImageState(old_comp.get().copy(), [new_comp], pad=0,
            mdl=mdls.SmoothFieldModel)
    do_levmarq(fake_s, new_comp.params, **kwargs)
