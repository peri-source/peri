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
from peri.comp import psfs, ilms, objs
from peri import states
from peri.logger import log
CLOG = log.getChild('opt')

"""
accel_correction is wrong as checked by a rosenbrock banana function; the
suggested accel steps are (1) always rejected and (2) in the wrong direction.

To fix:
1. opt.burn() -- right now it seems that the globals aren't fully optimized
    but the particles are after a few loops. So you might want to spend 1 more
    iteration updating the globals. Another eig update? More run length?

To add:
1. AugmentedState: ILM scale options? You'd need a way to get an overall scale
    block, which would probably need to come from the ILM itself.
6. With opt using big regions for particles, globals, it makes sense to
    put stuff back on the card again....

To fix:
1. In the engine, make do_run_1() and do_run_2() play nicer with each other.
2. opt.burn() hacks:
    a.  Once the state is mostly optimized, LMGlobals.J doesn't change much
        so you could lmglobals.do_internal_run(); lmparticles.do_internal_run()
        in a loop without recalculating J's (or maybe an eigen update).
        It would be way faster if you could store all the J's for the
        particle group collections. Save LMPartilceGroupCollection's lp's J's
        with numpy.save and a tmp file (standard library).
        -- this is implemented, but it doesn't work too well.

Algorithm is:
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
    Generates
    Would be nice if the following arguments were accepted by state.gradmodel:
        dl
        be_nice (put back updates or not)
        threept (three-point vs two-point stencil, i.e. 2+1 vs 1+1 updates)

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
        slicer = [slice(0, None), slice(0, None), slice(0, None)]
    J = s.gradmodel(params=params, inds=inds, slicer=slicer, flat=False, **kwargs)
    CLOG.debug('JTJ:\t%f' % (time.time()-start_time))
    return J, return_inds

def name_globals(s):
    all_params = s.params
    for p in s.param_positions():
        all_params.remove(p)
    for p in s.param_radii():
        all_params.remove(p)
    return all_params

def get_num_px_jtj(s, nparams, decimate=1, max_mem=2e9, min_redundant=20, **kwargs):
    #1. Max for a given max_mem:
    px_mem = int(max_mem / 8 / nparams) #1 float = 8 bytes
    #2. num_pix for a given redundancy
    px_red = min_redundant*nparams
    #3. And # desired for decimation
    px_dec = s.residuals.size/decimate

    if px_red > px_mem:
        raise RuntimeError('Insufficient max_mem for desired redundancy.')
    num_px = np.clip(px_dec, px_red, px_mem)
    return num_px

#=============================================================================#
#               ~~~~~  Particle Optimization stuff  ~~~~~
#=============================================================================#
def find_particles_in_tile(state, tile):
    """Finds the particles in a tile, as numpy.ndarray of ints."""
    bools = tile.contains(state.obj_get_positions())
    return np.arange(bools.size)[bools]

def separate_particles_into_groups(s, region_size=40, bounds=None, **kwargs):
    """
    Given a state, returns a list of groups of particles. Each group of
    particles are located near each other in the image. Every particle
    located in the desired region is contained in exactly 1 group.

    Parameters:
    -----------
    s : State
        The peri state to find particles in.
    region_size: Int or 3-element list-like of ints.
        The size of the box. Groups particles into boxes of shape
        (region_size[0], region_size[1], region_size[2]). If region_size
        is a scalar, the box is a cube of length region_size.
        Default is 40.
    bounds: 2-element list-like of 3-element lists.
        The sub-region of the image over which to look for particles.
            bounds[0]: The lower-left  corner of the image region.
            bounds[1]: The upper-right corner of the image region.
        Default (None -> ([0,0,0], s.oshape.shape)) is a box of the entire
        image size, i.e. the default places every particle in the image
        somewhere in the groups.

    Returns:
    -----------
    particle_groups: List
        Each element of particle_groups is an int numpy.ndarray of the
        group of nearby particles. Only contains groups with a nonzero
        number of particles, so the elements don't necessarily correspond
        to a given image region.
    """
    bounding_tile = s.oshape if bounds is None else Tile(bounds[0], bounds[1])
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

def calc_particle_group_region_size(s, region_size=40, max_mem=2e9, **kwargs):
    """
    Finds the biggest region size for LM particle optimization with a
    given memory constraint.
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
    Given a state and a tile that corresponds to the padded image, returns
    a tile that corresponds to the the corresponding pixels of the difference
    image
    """
    inner_tile = st.ishape.intersection([st.ishape, padded_tile])
    return inner_tile.translate(-st.pad)
#=============================================================================#
#               ~~~~~           Fit ilm???         ~~~~~
#=============================================================================#
def fit_ilm(new_ilm, old_ilm, **kwargs):
    """
    Fits a new peri.comp.ilms instance to (mostly) match the get_field
    of the old ilm, by creating a fake state with no particles and an
    identity psf and using *.do_levmarq()

    Parameters:
    -----------
    new_ilm : peri.comp.ilms instance
        The new ilm.
    old_ilm : peri.comp.ilms instance
        The old ilm to match to.
    **kwargs: The keyword args passed to the optimizers (LMGlobals through
        do_levmarq).

    See Also
    --------
    do_levmarq: Runs Levenberg-Marquardt minimization using a random
        subset of the image pixels. Works for any fit blocks.
    LMGlobals: Same, but with a cleaner engine instantiation.
    """
    shape = old_ilm.bkg.shape
    psf = psfs.IdentityPSF(params=np.zeros(1), shape=shape)
    obj = objs.SphereCollectionRealSpace(np.zeros([1,3]), np.zeros(1), shape=
            shape, typ=np.zeros(1))
    bkg = ilms.LegendrePoly2P1D(shape=shape, order=(1,1,1))
    bkg.update(bkg.block, np.zeros(bkg.block.size))
    fake_s = states.ConfocalImagePython(old_ilm.bkg.copy(), obj, psf, new_ilm,
            varyn=True, pad=1, bkg=bkg  )

    blk = fake_s.create_block('ilm')
    do_levmarq(fake_s, blk, **kwargs)
    return fake_s.ilm

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
    The engine for running levenberg-marquardt optimization on anything.
    There are 3 different options for optimizing:
        do_run_1():
            Checks to calculate full, Broyden, and eigen J, then tries a step.
            If the step is accepted, decreases damping; if not, increases.
            Checks for full, Broyden, and eigen J updates.
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
    Both partial updates are controlled by partial_update_frequency, which
    counts internal runs in do_internal_run and full runs in do_run_1.

    So, if you want a partial update every other run, full J the remaining,
    this would be:
        do_run_1(): update_J_frequency=2, partial_update_frequency=1
        do_run_2(): update_J_frequency=1, partial_update_frequency=1, run_length=2
    I would like to make this either a little more consistent or totally
    incompatible to be less confusing, especially since do_run_2() with
    update_J_frequency=2 just checks to decrease the damping without either
    partial updates.
    """
    def __init__(self, damping=1., increase_damp_factor=3., decrease_damp_factor=8.,
                min_eigval=1e-13, marquardt_damping=True, transtrum_damping=None,
                use_accel=False, max_accel_correction=1., ptol=1e-6,
                ftol=1e-5, costol=None, max_iter=5, run_length=5,
                update_J_frequency=1, broyden_update=False, eig_update=False,
                partial_update_frequency=3, num_eig_dirs=8):
        """
        Levenberg-Marquardt engine with all the options from the
        M. Transtrum J. Sethna 2012 ArXiV paper.

        Inputs:
        -------
            damping: Float
                The initial damping factor for Levenberg-Marquardt. Adjusted
                internally. Default is 1.
            increase_damp_factor: Float
                The amount to increase damping by when an attempted step
                has failed. Default is 3.
            decrease_damp_factor: Float
                The amount to decrease damping by after a successful step.
                Default is 8. increase_damp_factor and decrease_damp_factor
                must not have all the same factors.

            min_eigval: Float scalar, <<1.
                The minimum eigenvalue to use in inverting the JTJ matrix,
                to avoid degeneracies in the parameter space (i.e. 'rcond'
                in np.linalg.lstsq). Default is 1e-12.
            marquardt_damping: Bool
                Set to False to use Levenberg damping (damping matrix
                proportional to the identiy) instead of Marquardt damping
                (damping matrix proportional to the diagonal terms of JTJ).
                Default is True.
            transtrum_damping: Float or None
                If not None, then clips the Marquardt damping diagonal
                entries to be at least transtrum_damping. Default is None.

            use_accel: Bool
                Set to True to incorporate the geodesic acceleration term
                from M. Transtrum J. Sethna 2012. Default is False.
            max_accel_correction: Float
                Acceleration corrections bigger than max_accel_correction*
                the normal LM step are viewed as bad steps, causing a
                decrease in damping. Default is 1.0. Only applies to the
                do_run_1 method.

            ptol: Float
                Algorithm has converged when the none of the parameters
                have changed by more than ptol. Default is 1e-6.
            ftol: Float
                Algorithm has converged when the error has changed
                by less than ptol after 1 step. Default is 1e-6.
            costol: Float
                Algorithm has converged when the cosine of the angle
                between (residuals projected onto the model manifold)
                and (the residuals) is < costol. Default is None, i.e.
                doesn't check the cosine (since it takes a bit of time).
            max_iter: Int
                The maximum number of iterations before the algorithm
                stops iterating. Default is 5.

            update_J_frequency: Int
                The frequency to re-calculate the full Jacobian matrix.
                Default is 2, i.e. every other run.
            broyden_update: Bool
                Set to True to do a Broyden partial update on J after
                each step, updating the projection of J along the
                parameter change direction. Cheap in time cost, but not
                always accurate. Default is False.
            eig_update: Bool
                Set to True to update the projection of J along the most
                stiff eigendirections of JTJ. Slower than broyden but
                more accurate & useful. Default is False.
            num_eig_dirs: Int
                If eig_update == True, the number of eigendirections to
                update when doing the eigen update. Default is 4.
            partial_update_frequency: Int
                If broyden_update or eig_update, the frequency to do
                either/both of those partial updates. Default is 3.

        Relevant attributes
        -------------------
            do_run_1: Function
                ...what you should set when you use run_1 v run_2 etc
                For instance run_2 might stop prematurely since its
                internal runs update last_error, last_params, and it
                usually just runs until it takes a bad step == small
                param update.
            do_run_2: Function

        """
        # self.damping = float(damping)
        self.damping = np.array(damping).astype('float')
        self.increase_damp_factor = float(increase_damp_factor)
        self.decrease_damp_factor = float(decrease_damp_factor)
        self.min_eigval = min_eigval
        self.marquardt_damping = marquardt_damping
        self.transtrum_damping = transtrum_damping

        self.use_accel = use_accel
        self.max_accel_correction = max_accel_correction

        self.ptol = ptol
        self.ftol = ftol
        self.costol = costol
        self.max_iter = max_iter

        self.update_J_frequency = update_J_frequency
        self.broyden_update = broyden_update
        self.eig_update = eig_update
        self.num_eig_dirs = num_eig_dirs
        self.run_length = run_length
        self._inner_run_counter = 0
        self.partial_update_frequency = partial_update_frequency

        self._num_iter = 0

        #We want to start updating JTJ
        self.J = None
        self._J_update_counter = update_J_frequency
        self._fresh_JTJ = False

        #the max # of times trying to decrease damping before giving up
        self._max_inner_loop = 10

        #Finally we set the error and parameter values:
        self._set_err_paramvals()
        self._has_run = False

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
        """Function that, when called, returns data - model."""
        raise NotImplementedError('implement in subclass')

    def update_function(self, param_vals):
        """Takes an array param_vals, updates function, returns the new error"""
        raise NotImplementedError('implement in subclass')

    def do_run_1(self):
        """
        LM run evaluating 1 step at a time. Broyden or eigendirection
        updates replace full-J updates. No internal runs.
        """
        while not self.check_terminate():
            self._has_run = True
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
                if np.abs(er0 -self.error) > 1e-7:
                    raise RuntimeError('ARG!!!') #FIXME
                CLOG.debug('Bad step, increasing damping')
                CLOG.debug('\t\t%f\t%f' % (self.error, er1))
            _try = 0
            while (_try < self._max_inner_loop) and (not good_step):
                _try += 1
                self.increase_damping()
                delta_vals = self.find_LM_updates(self.calc_grad())
                er1 = self.update_function(self.param_vals + delta_vals)
                good_step = (find_best_step([self.error, er1]) == 1)
                if not good_step:
                    er0 = self.update_function(self.param_vals)
                    if np.abs(er0 -self.error) > 1e-7:
                        raise RuntimeError('ARG!!!') #FIXME
            if _try == (self._max_inner_loop-1):
                CLOG.warn('Stuck!')

            #state is updated, now params:
            if good_step:
                self._last_error = self.error
                self.error = er1
                CLOG.debug('Good step\t%f\t%f' % (self._last_error, self.error))
                self.update_param_vals(delta_vals, incremental=True)
                self.decrease_damping()
            self._num_iter += 1; self._inner_run_counter += 1

    def do_run_2(self):
        """
        LM run evaluating 2 steps (damped and not) and choosing the best.
        Runs with that damping + Broyden or eigendirection updates, until
        deciding to do a full-J update. Only changes damping after full-J
        updates.
        """
        while not self.check_terminate():
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
                _try = 0
                good_step = False
                CLOG.debug('Bad step, increasing damping')
                CLOG.debug('%f\t%f\t%f' % triplet)
                while (_try < self._max_inner_loop) and (not good_step):
                    self.increase_damping()
                    delta_vals = self.find_LM_updates(self.calc_grad())
                    er_new = self.update_function(self.param_vals + delta_vals)
                    good_step = er_new < self.error
                    _try += 1
                if not good_step:
                    #Throw a warning, put back the parameters
                    CLOG.warn('Stuck!')
                    self.error = self.update_function(self.param_vals.copy())
                else:
                    #Good step => Update params, error:
                    self.update_param_vals(delta_vals, incremental=True)
                    self.error = er_new
                    CLOG.debug('Sufficiently increased damping')
                    CLOG.debug('%f\t%f' % (triplet[0], self.error))

            elif best_step == 1:
                #er1 <= er2:
                good_step = True
                CLOG.debug('Good step, same damping')
                CLOG.debug('%f\t%f\t%f' % triplet)
                #Update to er1 params:
                er1_1 = self.update_function(self.param_vals + delta_params_1)
                if np.abs(er1_1 - er1) > 1e-6:
                    raise RuntimeError('GODDAMMIT!') #FIXME
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
                self._has_run = True
                self.do_internal_run()
            #1 loop
            self._num_iter += 1

    def do_internal_run(self):
        """
        Given a fixed damping, J, JTJ, iterates calculating steps, with
        optional Broyden or eigendirection updates.
        Called internally by do_run_2() but might also be useful on its own.
        """
        self._inner_run_counter = 0; good_step = True
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
                    do_correct_damping=False)
            er1 = self.update_function(self.param_vals + delta_vals)
            good_step = er1 < er0

            if good_step:
                CLOG.debug('%f\t%f' % (er0, er1))
                #Updating:
                self.update_param_vals(delta_vals, incremental=True)
                self._last_residuals = _last_residuals.copy()
                self._last_error = er0
                self.error = er1

                _last_residuals = self.calc_residuals().copy()
            else:
                er0_0 = self.update_function(self.param_vals)
                CLOG.debug('Bad step!')
                if np.abs(er0 - er0_0) > 1e-6:
                    raise RuntimeError('GODDAMMIT!') #FIXME

            self._inner_run_counter += 1

    def _calc_damped_jtj(self):
        if self.marquardt_damping:
            diag_vals = np.diag(self.JTJ)
        elif self.transtrum_damping is not None:
            diag_vals = np.clip(np.diag(self.JTJ), self.transtrum_damping, np.inf)
        else:
            diag_vals = np.ones(self.JTJ.shape[0])

        diag = np.diagflat(diag_vals)
        damped_JTJ = self.JTJ + self.damping*diag
        return damped_JTJ

    def find_LM_updates(self, grad, do_correct_damping=True):
        """
        Calculates LM updates, with or without the acceleration correction.
        """
        damped_JTJ = self._calc_damped_jtj()
        delta0, res, rank, s = np.linalg.lstsq(damped_JTJ, -grad, rcond=self.min_eigval)
        if self._fresh_JTJ:
            CLOG.debug('%d degenerate of %d total directions' % (delta0.size-rank, delta0.size))

        if self.use_accel:
            accel_correction = self.calc_accel_correction(damped_JTJ, delta0)
            nrm_d0 = np.sqrt(np.sum(delta0**2))
            nrm_corr = np.sqrt(np.sum(accel_correction**2))
            CLOG.debug('|correction| / |LM step|\t%e' % (nrm_corr/nrm_d0))
            if nrm_corr/nrm_d0 < self.max_accel_correction:
                delta0 += accel_correction
            elif do_correct_damping:
                CLOG.debug('Untrustworthy step! Increasing damping...')
                self.increase_damping()
                damped_JTJ = self._calc_damped_jtj()
                delta0, res, rank, s = np.linalg.lstsq(damped_JTJ, -grad, \
                        rcond=self.min_eigval)

        if np.any(np.isnan(delta0)):
            CLOG.fatal('Calculated steps have nans!?')
            raise RuntimeError('Calculated steps have nans!?')
        return delta0

    def increase_damping(self):
        self.damping *= self.increase_damp_factor

    def decrease_damping(self, undo_decrease=False):
        if undo_decrease:
            self.damping *= self.decrease_damp_factor
        else:
            self.damping /= self.decrease_damp_factor

    def update_param_vals(self, new_vals, incremental=False):
        self._last_vals = self.param_vals.copy()
        if incremental:
            self.param_vals += new_vals
        else:
            self.param_vals = new_vals.copy()
        #And we've updated, so JTJ is no longer valid:
        self._fresh_JTJ = False

    def calc_model_cosine(self, decimate=None):
        """
        Calculates the cosine of the fittable residuals with the actual
        residuals, cos(phi) = |P^T r| / |r| where P^T is the projection
        operator onto the model manifold and r the residuals.

        `Decimate' allows for every nth pixel only to be counted for speed.
        While this is n x faster, it is considerably less accurate, so the
        default is no decimation. (set decimate to an int or None).
        """
        slicer = slice(0,-1,decimate)

        #1. Calculate projection term
        u, sig, v = np.linalg.svd(self.J[:,slicer], full_matrices=False) #slow part
        # p = np.dot(v.T, v) - memory error, so term-by-term
        r = self.calc_residuals()[slicer]
        abs_r = np.sqrt((r*r).sum())

        v_r = np.dot(v,r/abs_r)
        projected = np.dot(v.T, v_r)

        abs_cos = np.sqrt((projected*projected).sum())
        return abs_cos

    def get_termination_stats(self, get_cos=True):
        """
        Returns a dict of termination statistics
        """
        delta_vals = self._last_vals - self.param_vals
        delta_err = self._last_error - self.error
        to_return = {'delta_vals':delta_vals, 'delta_err':delta_err,
                'num_iter':1*self._num_iter}
        if get_cos:
            model_cosine = self.calc_model_cosine()
            to_return.update({'model_cosine':model_cosine})
        return to_return

    def check_completion(self):
        """
        Checks if the algorithm has found a satisfactory minimum
        """
        terminate = False

        #1. change in params small enough?
        delta_vals = self._last_vals - self.param_vals
        terminate |= np.all(np.abs(delta_vals) < self.ptol)

        #2. change in err small enough?
        delta_err = self._last_error - self.error
        terminate |= (delta_err < self.ftol)

        #3. change in cosine small enough?
        if self.costol is not None:
            curcos = self.calc_model_cosine()
            terminate |= (curcos < self.costol)

        return terminate

    def check_terminate(self):
        """
        Termination if ftol, ptol, costol are < a certain amount
        """

        if not self._has_run:
            return False
        else:
            #1-3. ftol, ptol, model cosine low enough?
            terminate = self.check_completion()

            #4. too many iterations??
            terminate |= (self._num_iter >= self.max_iter)
            return terminate

    def check_update_J(self):
        """
        Checks if the full J should be updated. Right now, just updates if
        we've done update_J_frequency loops
        """
        self._J_update_counter += 1
        update = self._J_update_counter >= self.update_J_frequency
        return update & (not self._fresh_JTJ)

    def update_J(self):
        self.calc_J()
        self.JTJ = np.dot(self.J, self.J.T)
        self._fresh_JTJ = True
        self._J_update_counter = 0

    def calc_grad(self):
        residuals = self.calc_residuals()
        return -np.dot(self.J, residuals)

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
                ((self._inner_run_counter % self.partial_update_frequency) == 0))
        return do_update

    def update_Broyden_J(self):
        """
        Broyden update of jacobian.
        """
        delta_vals = self.param_vals - self._last_vals
        delta_residuals = self.calc_residuals() - self._last_residuals
        nrm = np.sqrt(delta_vals*delta_vals)
        direction = delta_vals / nrm
        vals = delta_residuals / nrm
        self._rank_1_J_update(direction, vals)
        self.JTJ = np.dot(self.J, self.J.T)

    def check_update_eig_J(self):
        do_update = (self.eig_update & (not self._fresh_JTJ) &
                ((self._inner_run_counter % self.partial_update_frequency) == 0))
        return do_update

    def update_eig_J(self):
        vls, vcs = np.linalg.eigh(self.JTJ)
        res0 = self.calc_residuals()
        for a in xrange(min([self.num_eig_dirs, vls.size])):
            #1. Finding stiff directions
            stif_dir = vcs[-(a+1)] #already normalized

            #2. Evaluating derivative along that direction, we'll use dl=5e-4:
            dl = 5e-4
            _ = self.update_function(self.param_vals+dl*stif_dir)
            res1 = self.calc_residuals()

            #3. Updating
            grad_stif = (res1-res0)/dl
            self._rank_1_J_update(stif_dir, grad_stif)

        self.JTJ = np.dot(self.J, self.J.T)
        #Putting the parameters back:
        _ = self.update_function(self.param_vals)

    def calc_accel_correction(self, damped_JTJ, delta0):
        """
        This is currently wrong.... I think that there is an error in
        the Transtrum paper or I am interpreting it incorrectly....
        """
        dh = 0.1
        rm0 = self.calc_residuals()
        _ = self.update_function(self.param_vals + delta0*dh)
        rm1 = self.calc_residuals()
        term1 = (rm1 - rm0) / dh
        #and putting back the parameters: - necessary? FIXME
        _ = self.update_function(self.param_vals)

        term2 = np.dot(self.J.T, delta0)
        der2 = 2./dh*(term1 - term2)

        damped_JTJ = self._calc_damped_jtj()
        corr, res, rank, s = np.linalg.lstsq(damped_JTJ, np.dot(self.J, der2),
                rcond=self.min_eigval)
        corr *= -0.5
        return corr

class LMFunction(LMEngine):
    def __init__(self, data, func, p0, func_args=(), func_kwargs={}, dl=1e-8,
            **kwargs):
        """
        Levenberg-Marquardt engine for a user-supplied function with all
        the options from the M. Transtrum J. Sethna 2012 ArXiV paper. See
        LMEngine for documentation.

        Inputs:
        -------
            data : N-element numpy.ndarray
                The measured data to fit.
            func: Function
                The function to evaluate. Syntax must be
                func(param_values, *func_args, **func_kwargs), and return a
                numpy.ndarray of the same shape as data
            p0 : numpy.ndarray
                The initial parameter guess.
            dl : Float
                The fractional amount to use for finite-difference derivatives,
                i.e. (f(x*(1+dl)) - f(x)) / (x*dl) in each direction.
            func_args : List-like
                Extra *args to pass to the function. Optional.
            func_kargs : Dictionary
                Extra **kwargs to pass to the function. Optional.
            **kwargs : Any keyword args passed to LMEngine.
        """
        self.data = data
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.param_vals = p0
        self.dl = dl
        super(LMFunction, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        """
        Must update:
            self.error, self._last_error, self.param_vals, self._last_vals
        """
        # self.param_vals = p0 #sloppy...
        self._last_vals = self.param_vals.copy()
        self.error = self.update_function(self.param_vals)
        self._last_error = 1.001*self.error

    def calc_J(self):
        """Updates self.J, returns nothing"""
        del self.J
        self.J = np.zeros([self.param_vals.size, self.data.size])
        dp = np.zeros_like(self.param_vals)
        f0 = self.model.copy()
        for a in xrange(self.param_vals.size):
            dp *= 0
            dp[a] = (1+self.param_vals[a]) * self.dl
            f1 = self.func(self.param_vals + dp, *self.func_args, **self.func_kwargs)
            self.J[a] = (f1 - f0) / dp[a]

    def calc_residuals(self):
        return self.data - self.model

    def update_function(self, param_vals):
        """Takes an array param_vals, updates function, returns the new error"""
        self.model = self.func(param_vals, *self.func_args, **self.func_kwargs)
        d = self.calc_residuals()
        return np.dot(d.flat, d.flat) #faster for large arrays than (d*d).sum()


class LMGlobals(LMEngine):
    def __init__(self, state, param_names, max_mem=3e9, opt_kwargs={}, **kwargs):
        """
        Levenberg-Marquardt engine for state globals with all the options
        from the M. Transtrum J. Sethna 2012 ArXiV paper. See LMEngine
        for documentation.

        Inputs:
        -------
        state: peri.states.ConfocalImagePython instance
            The state to optimize
        param_names: List of strings(???)
            The parameternames to optimize over
        max_mem: Int
            The maximum memory to use for the optimization; controls pixel
            decimation. Default is 3e9.
        opt_kwargs: Dict
            Dict of **kwargs for opt implementation. Right now only for
            *.get_num_px_jtj, i.e. keys of 'decimate', min_redundant'.
        """
        self.state = state
        self.kwargs = opt_kwargs
        self.max_mem = max_mem
        self.num_pix = get_num_px_jtj(state, len(param_names), max_mem=max_mem,
                **self.kwargs)
        self.param_names = param_names
        super(LMGlobals, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        self.error = self.state.error
        self._last_error = self.state.error
        self.param_vals = np.ravel(self.state.state[self.param_names])
        self._last_values = self.param_vals.copy()

    def calc_J(self):
        del self.J
        self.J, self._inds = get_rand_Japprox(self.state,
                self.param_names, num_inds=self.num_pix, **self.kwargs)

    def calc_residuals(self):
        return self.state.residuals.ravel()[self._inds]

    def update_function(self, values):
        self.state.update(self.param_names, values)
        return self.state.error

    def set_params(self, new_param_names, new_damping=None):
        self.param_names = new_param_names
        self._set_err_paramvals()
        self.reset(new_damping=new_damping)

class LMParticles(LMEngine):
    def __init__(self, state, particles, include_rad=True, **kwargs):
        self.state = state
        self.particles = particles
        self.param_names = (state.param_particle(particles) if include_rad
                else state.param_particle_pos(particles))
        self.error = self.state.error
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
        self._last_error = self.state.error
        self.param_vals = np.ravel(self.state.state[self.param_names])
        self._last_vals = self.param_vals.copy()

    def calc_J(self):
        self._dif_tile = self._get_diftile()
        del self.J
        self.J = self.state.gradmodel(params=self.param_names, rts=False,
            slicer=self._dif_tile.slicer)

    def calc_residuals(self):
        return self.state.residuals[self._dif_tile.slicer].ravel()

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
    Convenience wrapper for LMParticles. This generates a separate instance
    for the particle groups each time and optimizes with that, since storing
    J for the particles is too large.

    Try implementing a way to save the J's via tempfile's. lp.update_J()
    only updates J, JTJ, so you'd only have to save those (or get JTJ from J).


    Methods
    -------
        reset: Re-calculate all the groups
        do_run_1: Run do_run_1 for every group of particles
        do_run_2: Run do_run_2 for every group of particles
    """
    def __init__(self, state, region_size=40, do_calc_size=True, max_mem=2e9,
            get_cos=False, save_J=False, **kwargs):
        """
        Parameters
        ----------
            state: peri.states instance
                The state to optimize
            region_size: Int or 3-element list-like of ints
                The region size for sub-blocking particles. Default is 40
            do_calc_size: Bool
                If True, calculates the region size internally based on
                the maximum allowed memory. Default is True
            get_cos : Bool
                Set to True to include the model cosine in the statistics
                on each individual group's run, using
                LMEngine.get_termination_stats(), stored in self.stats.
                Default is False
            save_J : Bool
                Set to True to create a series of temp files that save J
                for each group of particles. Needed for do_internal_run().
                Default is False.
            **kwargs:
                Pass any kwargs that would be passed to LMParticles.
                Stored in self._kwargs for reference.

        Attributes
        ----------
            stats : List

        """

        self.state = state
        self._kwargs = kwargs
        self.region_size = region_size
        self.get_cos = get_cos
        self.save_J = save_J
        self.max_mem = max_mem

        self.reset(do_calc_size=do_calc_size)

    def reset(self, new_region_size=None, do_calc_size=True, new_damping=None,
            new_max_mem=None):
        """Resets the particle groups and optionally the region size and damping."""
        if new_region_size is not None:
            self.region_size = new_region_size
        if new_max_mem != None:
            self.max_mem = new_max_mem
        if do_calc_size:
            self.region_size = calc_particle_group_region_size(self.state,
                    self.region_size, max_mem=self.max_mem, **self._kwargs)
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
        self._do_run(mode='1')

    def do_run_2(self):
        self._do_run(mode='2')

    def do_internal_run(self):
        if not self.save_J:
            raise RuntimeError('self.save_J=True required for do_internal_run()')
        if not np.all(self._has_saved_J):
            raise RuntimeError('J, JTJ have not been pre-computed. Call do_run_1 or do_run_2')
        self._do_run(mode='internal')

class AugmentedState(object): #FIXME when blocks work....
    """
    A state that, in addition to having normal state update options,
    allows for updating all the particle R, xyz's depending on their
    positions -- basically rscale(x) for everything.
    Right now I'm just doing this with R(z)
    """
    def __init__(self, state, param_names, rz_order=3):
        """
        block can be an array of False, that's OK
        However it cannot have any radii blocks
        """
        #FIXME block -> param_names

        rad_nms = state.param_radii()
        has_rad = map(lambda x: x in param_names, rad_nms)
        if np.any(has_rad):# or np.any(block & state.create_block('rscale')):
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

    def set_block(self, new_block):
        """
        I don't think there is a point to this since the rscale(z) aren't
        actual parameters
        """
        raise NotImplementedError

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
    def __init__(self, aug_state, max_mem=3e9, opt_kwargs={}, **kwargs):
        """
        Levenberg-Marquardt engine for state globals with all the options
        from the M. Transtrum J. Sethna 2012 ArXiV paper. See LMGlobals
        for documentation.

        Inputs:
        -------
        aug_state: opt.AugmentedState instance
            The augmented state to optimize
        max_mem: Int
            The maximum memory to use for the optimization; controls block
            decimation. Default is 3e9.
        opt_kwargs: Dict
            Dict of **kwargs for opt implementation. Right now only for
            opt.get_num_px_jtj, i.e. keys of 'decimate', min_redundant'.
        """
        self.aug_state = aug_state
        self.kwargs = opt_kwargs
        self.max_mem = max_mem
        self.num_pix = get_num_px_jtj(aug_state.state, aug_state.param_vals.size,
                max_mem=max_mem, **self.kwargs)
        super(LMAugmentedState, self).__init__(**kwargs)

    def _set_err_paramvals(self):
        self.error = self.aug_state.state.error
        self._last_error = self.aug_state.state.error
        self.param_vals = self.aug_state.param_vals.copy()
        self._last_values = self.param_vals.copy()

    def calc_J(self):
        #0.
        del self.J

        #1. J for the state:
        s = self.aug_state.state
        sa = self.aug_state
        J_st, inds = get_rand_Japprox(s, self.aug_state.param_names,
                num_inds=self.num_pix, **self.kwargs)
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
            der = (i1-i0)/dl
            J_aug.append(der[self._inds].copy().ravel())

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
def do_levmarq(s, param_names, damping=0.1, decrease_damp_factor=10., run_length=6,
        eig_update=True, collect_stats=False, use_aug=False, run_type=2,
        **kwargs):
    """
    Convenience wrapper for LMGlobals. Same keyword args, but I've set
    the defaults to what I've found to be useful values for optimizing globals.
    See LMGlobals and LMEngine for documentation.
    """
    if use_aug:
        aug = AugmentedState(s, param_names, rz_order=3)
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
        run_length=4, collect_stats=False, **kwargs):
    """
    Convenience wrapper for LMParticles. Same keyword args, but I've set
    the defaults to what I've found to be useful values for optimizing
    particles. See LMParticles and LMEngine for documentation.
    """
    lp = LMParticles(s, particles, damping=damping, run_length=run_length,
            decrease_damp_factor=decrease_damp_factor, **kwargs)
    lp.do_run_2()
    if collect_stats:
        return lp.get_termination_stats()

def do_levmarq_all_particle_groups(s, region_size=40, damping=1.0,
        decrease_damp_factor=10., run_length=4, collect_stats=False, **kwargs):
    """
    Convenience wrapper for LMParticleGroupCollection. Same keyword args,
    but I've set the defaults to what I've found to be useful values for
    optimizing particles. See LMParticleGroupCollection for documentation.
    """
    lp = LMParticleGroupCollection(s, region_size=region_size, damping=damping,
            run_length=run_length, decrease_damp_factor=decrease_damp_factor,
            get_cos=collect_stats, **kwargs)
    lp.do_run_2()
    if collect_stats:
        return lp.stats

def burn(s, n_loop=6, collect_stats=False, desc='', use_aug=False,
        ftol=1e-3, mode='burn', max_mem=3e9, include_rad=True):
    """
    Burns a state through calling LMParticleGroupCollection and LMGlobals/
    LMAugmentedState.

    Parameters
    ----------
        s : peri.states.ConfocalImagePython instance
            The state to optimize

        n_loop : Int
            The number of times to loop over in the optimizer. Default is 6.

        collect_stats : Bool
            Whether or not to collect information on the optimizer's
            performance. Default is False, because True tends to increase
            the memory usage above max_mem.

        desc : string
            Description to append to the states.save() call every loop.
            Set to None to avoid saving. Default is '', which selects
            one of 'burning', 'polishing', 'doing_positions'

        use_aug: Bool
            Set to True to optimize with an augmented state (R(z) as a
            global parameter) vs. with the normal global parameters.
            Default is False (no augmented).

        ftol : Float
            The change in error at which to terminate.

        mode : 'burn' or 'do-particles'
            What mode to optimize with.
                'burn'          : Your state is far from the minimum.
                'do-particles'  : Positions are far from the minimum,
                                  globals are well-fit.
            'burn' is the default and will optimize any scenario, but the
            others will be faster for their specific scenarios.

        max_mem : Numeric
            The maximum amount of memory allowed for the optimizers' J's,
            for both particles & globals. Default is 3e9, i.e. 3GB per
            optimizer.

    Comments
    --------
        - It would be nice if some of these magic #'s (region size, num_eig_dirs,
            etc) were calculated in a good way.

    burn            : lm.do_run_2(), lp.do_run_2()
    do-particles    : lp.do_run_2(), scales for ilm, bkg's
    """
    mode = mode.lower()
    if mode not in {'burn', 'do-particles'}:
        raise ValueError('mode must be one of burn, do-particles')
    if desc is '':
        desc = mode + 'ing' if mode != 'do-particles' else 'doing-particles'

    eig_update = mode != 'do-particles'
    glbl_run_length = 6 if mode != 'do-particles' else 3

    if mode == 'do-particles':
        glbl_nms = ['ilm-scale', 'offset']  #bkg?
    else:
        glbl_nms = name_globals(s)#, include_rscale=(not use_aug), include_sigma=False)

    all_lp_stats = []
    all_lm_stats = []

    #2. Burn.
    CLOG.info('Start of loop %d:\t%f' % (0, s.error))
    for a in xrange(n_loop):
        start_err = s.error
        #2a. Globals
        glbl_dmp = 0.3 if a == 0 else 3e-2
        if a != 0 or mode != 'do-particles':
            gstats = do_levmarq(s, glbl_nms, max_iter=1, run_length=
                    glbl_run_length, eig_update=eig_update, num_eig_dirs=10,
                    partial_update_frequency=3, damping=glbl_dmp,
                    decrease_damp_factor=10., use_aug=use_aug,
                    collect_stats=collect_stats, ftol=0.1*ftol, max_mem=max_mem)
            all_lm_stats.append(gstats)
        if desc is not None:
            states.save(s, desc=desc)
        CLOG.info('Globals, loop %d:\t%f' % (a, s.error))

        #2b. Particles
        prtl_dmp = 1.0 if a==0 else 1e-2
        #For now, I'm calculating the region size. This might be a bad idea
        #because 1 bad particle can spoil the whole group.
        pstats = do_levmarq_all_particle_groups(s, region_size=40, max_iter=1,
                do_calc_size=True, run_length=4, eig_update=False,
                damping=prtl_dmp, ftol=0.1*ftol, collect_stats=collect_stats,
                max_mem=max_mem, include_rad=include_rad)
        all_lp_stats.append(pstats)
        if desc is not None:
            states.save(s, desc=desc)
        CLOG.info('Particles, loop %d:\t%f' % (a, s.error))
        gc.collect()

        #2c. terminate?
        new_err = s.error
        if (start_err - new_err) < ftol:
            break

    if collect_stats:
        return all_lp_stats, all_lm_stats
