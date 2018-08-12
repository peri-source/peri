from builtins import map, zip, range, object

import os
import sys
import time
import tempfile
import pickle
import gc
import itertools

import numpy as np
from numpy.random import randint
from scipy.optimize import newton, minimize_scalar

from peri.util import Tile, Image
from peri import states
from peri import models as mdl
from peri.logger import log
CLOG = log.getChild('opt')
# import peri.opt.optengine as optengine

"""
TODO:
    1. burn -- 2 loops of globals is only good for the first few loops; after
            that it's overkill. Best to make a 3rd mode now, since the
            psf and zscale move around without a set of burns.
    2.  separate_particles_into_groups : Current version makes a lot of
        size-1 groups. A better version would be:
            a. initialize a list of active groups as all particles.
            b. initialize a list of final groups as [].
            c. while len(active list) > 0:
                pop a group from the active list
                if it is small enough for mem < max_mem:
                    append it to the final list
                else:
                    split it in two (randomly? along its long direction?
                    both?) and add both groups back to the active list.
        Algorithm should run in O(ln N) and the groups should all be
        sizeable.

To fix:
2.  Right now, when marquardt_damping=False (the default, which works nicely),
    the correct damping parameter scales with the image size. For each element
    of J is O(1), so JTJ[i,j]~1^2 * N ~ N where N is the number of residuals
    pixels. But the damping matrix only matters in its overall ratio to J.
    So, changing max_mem or changing the image size will affect what a
    reasonable damping is. One way to do this is to scale the damping by
    the size of the residuals..........................................
        np.mean(np.diag(self.JTJ))
"""

def get_rand_Japprox(s, params, num_inds=1000, include_cost=False, **kwargs):
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
        include_cost : Bool, optional
            Set to True to append a finite-difference measure of the full
            cost gradient onto the returned J.

    Other Parameters
    ----------------
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
        return_inds = np.sort(inds)
    else:
        inds = None
        return_inds = slice(0, None)
        slicer = [slice(0, None)]*len(s.residuals.shape)
    if include_cost:
        Jact, ge = s.gradmodel_e(params=params, inds=inds, slicer=slicer,flat=False,
                **kwargs)
        Jact *= -1
        J = [Jact, ge]
    else:
        J = -s.gradmodel(params=params, inds=inds, slicer=slicer, flat=False,
                **kwargs)
    CLOG.debug('J:\t%f' % (time.time()-start_time))
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
    for p in s.param_particle(np.arange(s.obj_get_positions().shape[0])):
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
    px_mem = int(max_mem // 8 // nparams) #1 float = 8 bytes
    #2. num_pix for a given redundancy
    px_red = min_redundant*nparams
    #3. And # desired for decimation
    px_dec = s.residuals.size//decimate

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
        for a in range(damp_vec.size):
            if nm in params[a]:
                damp_vec[a] *= fctr
    return damp_vec


#=============================================================================#
#               ~~~~~  For Groups of particle optimization  ~~~~~
#=============================================================================#


class ParticleGroupCreator(object):
    def __init__(self, state, max_mem=1e9, doshift=False):
        """
        Parameters
        ----------
        state
        max_mem : numeric, optional
        doshift : {True, False, 'rand'}, optional
        """
        self.state = state
        self.max_mem = max_mem

    def find_particles_in_tile(self, tile):
        bools = tile.contains(self.state.obj_get_positions())
        return np.arange(bools.size)[bools]

    def _check_groups(self, groups):
        """Ensures that all particles are included in exactly 1 group"""
        all_indices = [i for group in groups for i in group]
        unique_indices = np.unique(all_indices)
        n_particles = self.state.obj_get_positions().shape[0]
        ok = [unique_indices.size == len(all_indices),
              unique_indices.size == n_particles,
              (np.arange(n_particles) == np.sort(all_indices)).all()]
        return all(ok)

    def calc_group_memory_bytes(self, group):
        param_names = self.state.param_particle(group)
        param_values = self.state.get_values(param_names)
        update_region = self.state.get_update_io_tiles(
            param_names, param_values)[2]
        num_pixels_in_update_region = update_region.volume.astype('int64')
        num_entries_in_J = num_pixels_in_update_region * len(param_names)
        return num_entries_in_J * 8  # 8 for number of bytes in float64

    def separate_particles_into_groups(self):
        raise NotImplementedError('implement in subclass')


class BoxParticleGroupCreator(ParticleGroupCreator):
    def __init__(self, state, max_mem=1e9, doshift=False):
        super(BoxParticleGroupCreator, self).__init__(state, max_mem=max_mem)
        self.doshift = (np.random.choice([True, False]) if doshift == 'rand'
                        else doshift)

    def calc_particle_group_region_size(self, initial_region_size=40):
        im_shape = self.state.oshape.shape
        if np.size(initial_region_size) == 1:
            region_size = np.full(np.ndim(im_shape), initial_region_size,
                                  dtype='int')
        else:
            region_size = np.array(initial_region_size, dtype='int')

        increase_size_amount = 2
        if self._calc_mem_usage(region_size) > self.max_mem:
            while ((self._calc_mem_usage(region_size) > self.max_mem) and
                    np.any(region_size > 2)):
                region_size = np.clip(region_size-1, 2, im_shape)
        else:
            while ((self._calc_mem_usage(region_size) < self.max_mem) and
                    np.any(region_size < im_shape)):
                region_size = np.clip(region_size+1, 2, im_shape)
            # un-doing 1 iteration to ensure the required mem is < max_mem:
            region_size -= increase_size_amount
        return region_size

    def separate_particles_into_groups(self):
        region_size = self.calc_particle_group_region_size()
        groups = self._separate_particles_into_groups(region_size)
        assert self._check_groups(groups)
        return groups

    def _separate_particles_into_groups(self, region_size):
        bounding_tile = self.state.oshape.translate(-self.state.pad)
        n_translate = np.ceil(bounding_tile.shape.astype('float') /
                              region_size).astype('int')
        region_tile = Tile(left=bounding_tile.l,
                           right=bounding_tile.l + region_size)
        if self.doshift:
            shift = region_size // 2
            n_translate += 1
        else:
            shift = 0
        # ~~~ Get the x, y, z shifts of the particles:
        num_tiles_zyx = [[i for i in range(n_translate_xi)]
                         for n_translate_xi in n_translate]
        tile_shift_values = [np.array([i, j, k]) * region_size
                             for i, j, k in itertools.product(*num_tiles_zyx)]
        group_tiles = [region_tile.translate(tile_shift_value)
                       for tile_shift_value in tile_shift_values]
        groups = [self.find_particles_in_tile(group_tile)
                  for group_tile in group_tiles]
        groups = [g for g in groups if len(g) > 0]
        return groups

    def _calc_mem_usage(self, region_size):
        particle_groups = self._separate_particles_into_groups(region_size)
        # The actual mem usage is the max of the memory usage of all the
        # particle groups. However this is too slow. So instead we use the
        # max of the memory of the biggest 5 particle groups:
        num_in_group = [np.size(g) for g in particle_groups]
        biggroups = [particle_groups[i] for i in np.argsort(num_in_group)[-5:]]
        mems = [self.calc_group_memory_bytes(g) for g in biggroups]
        return np.max(mems)


class ListParticleGroupCreator(ParticleGroupCreator):
    def __init__(self, state, max_mem=1e9, split_by=2):
        super(ListParticleGroupCreator, self).__init__(state, max_mem=max_mem)
        self.split_by = split_by
        self.percentiles = 100 * np.arange(split_by) / float(split_by)

    def separate_particles_into_groups(self):
        active_list=[self._get_all_particles()]
        groups = []
        while len(active_list) > 0:
            current_group = active_list.pop()
            if self.calc_group_memory_bytes(current_group) < self.max_mem:
                groups.append(current_group)
            else:
                smaller_groups = self.split_into_smaller_groups(current_group)
                active_list.extend(smaller_groups)
        return groups

    def split_into_smaller_groups(self, current_group):
        x = self.find_amount_along_longest_axis(current_group)
        cut_at = np.percentile(x, self.percentiles)
        group_masks = [(x >= lower) & (x < upper)
                       for lower, upper in zip(cut_at[:-1], cut_at[1:])]
        group_masks.append(x >= cut_at[-1])
        new_groups = [current_group[m] for m in group_masks]
        return new_groups

    def find_amount_along_longest_axis(self, group):
        positions = self.state.obj_get_positions()[group]
        # 1. Find the longest principal axis of the group:
        radius_of_gyration = np.dot(positions.T, positions)
        vals, vecs = np.linalg.eigh(radius_of_gyration)
        direction = vecs[-1]
        # 2. The coordinate along that direction:
        coordinate = positions.dot(direction)
        return coordinate

    def _get_all_particles(self):
        return self.find_particles_in_tile(self.state.oshape)

def separate_particles_into_groups(state, max_mem=1e9, doshift=False):
    """Separates particles into convenient groups for optimization.

    Parameters
    ----------
    s : :class:`peri.states.ImageState`
        The peri state to find particles in.
    max_mem : numeric, optional
        Maximal memory to be used by the optimizer, in bytes.
        Default is 1e9
    doshift : {True, False, `'rand'`}, optional
        Whether or not to shift the tile boxes by half a region size, to
        prevent the same particles to be chosen every time. If `'rand'`,
        randomly chooses either True or False. Default is False

    Returns
    -------
    particle_groups : List
        Each element of particle_groups is a numpy.ndarray of the
        indices of a group of particles, with each particle in one group.
    """
    groupmaker = BoxParticleGroupCreator(
        state, max_mem=max_mem, doshift=doshift)
    return groupmaker.separate_particles_into_groups()


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


class AugmentedState(object):
    """
    Augments a state with a set of radii(z) parameters.

    Operates by updating the radii as ``R`` -> ``R0*np.exp(legval(zp))``,
    where ``zp`` is a rescaled ``z`` coordinate and ``R0`` the initial
    radii. The order of the Legendre polynomial for the rescaling is set
    by ``rz_order``.

    Parameters
    ----------
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
        LMAugmentedState : Levenberg-Marquadt optimization with an
            ``AugmentedState``.

    Notes
    -----
        This could be extended to do xyzshift(xyz), radii(xyz)
    """
    def __init__(self, state, param_names, rz_order=3):
        rad_nms = state.param_radii()
        has_rad = list(map(lambda x: x in param_names, rad_nms))
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
        inds = list(range(self.state.obj_get_positions().shape[0]))
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
        self.update_rscl_x_params(param_vals[self.rscale_mask])
        self.state.update(self.param_names, param_vals[self.globals_mask])
        self.param_vals[:] = param_vals.copy()
        if np.any(np.isnan(self.state.residuals)):
            raise FloatingPointError('state update caused nans in residuals')

    def update_rscl_x_params(self, new_rscl_params):
        #1. What to change:
        p = self._initial_pos

        #2. New, old values:
        self.param_vals[self.rscale_mask] = new_rscl_params
        new_scale = self.rad_func(p)

        rnew = self._initial_rad * new_scale
        #FIXME you can do a full update without the extra convolution
        #... right now don't worry about it
        self.state.update(self._rad_nms, rnew)


#=============================================================================#
#         ~~~~~             Convenience Functions             ~~~~~
#=============================================================================#


def create_state_optimizer(st, param_names, rts=False, dl=3e-6, **kwargs):
    optobj = optengine.OptImageState(st, param_names, rts=rts, dl=dl)
    lm = optengine.LMOptimizer(optobj, **kwargs)
    return lm


def optimize_parameters(st, param_names, **kwargs):
    """Levenberg-Marquardt optimization on a set of parameters."""
    lm = create_state_optimizer(st, param_names, **kwargs)
    lm.optimize()


def optimize_particles(st, particle_indices, include_rad=True, **kwargs):
    """Levenberg-Marquardt optimization on a set of particles."""
    param_names = (st.param_particle(particle_indices) if include_rad
                   else st.param_particle_pos(inds))
    optimize_parameters(st, param_names)


def particles_generator(st, groups, include_rad=True, rts=True, dl=3e-6,
                        **kwargs):
    """does stuff"""
    for g in groups:
        param_names = (st.param_particle(g) if include_rad else
                       st.param_particle_pos(g))
        optobj = OptImageState(st, param_names, dl=dl, rts=rts)
        yield LMOptimizer(optobj, **kwargs)


def create_particle_groups_optimizer(st, groups, **kwargs):
    optimizer_generator = particles_generator(st, groups, **kwargs)
    # param_ranges=limit_particles(st, g) for particles generator?
    return GroupedOptimizer(optimizer_generator)


def optimize_particle_groups(st, groups, **kwargs):
    """Levenberg-Marquardt optimization for groups of particles.

    Parameters
    ---------
    st : `peri.states.ImageState`
    groups : list
        A list of groups of particles. Each

    Other Parameters
    ----------------
    """
    grouped_optimizer = create_particle_groups_optimizer(st, groups, **kwargs)
    grouped_optimizer.optimize()


def optimize_all_particles(st, max_mem=1e9, doshift=False, **kwargs):
    all_groups = separate_particles_into_groups(st, max_mem=max_mem,
                                                doshift=doshift)
    return optimize_particle_groups(st, all_groups, **kwargs)


# TODO Implement with optengine (problem is rz_order)!!
# TODO Deprecate!! needs rz_order
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


# TODO Deprecate!!
# TODO collect_stats , run_length kwarg.
#      -- needs to return stats if collect_stats is True for backwards
#      compatibility.
def do_levmarq_particles(st, particles, damping=1.0, decrease_damp_factor=10.,
        run_length=4, collect_stats=False, max_iter=2, **kwargs):
    """Levenberg-Marquardt optimization on a set of particles.

    See Also
    --------
        do_levmarq_all_particle_groups : Levenberg-Marquardt optimization
            of all the particles in the state.

        do_levmarq : Levenberg-Marquardt optimization of the entire state;
            useful for optimizing global parameters.
    """
    optimize_particles(st, particles, damp=damping, maxiter=max_iter,
                       dampdown=decrease_damp_factor)
    if collect_stats:
        # return lp.get_termination_stats()
        raise NotImplementedError


# TODO Deprecate!!
# TODO collect_stats , run_length kwarg.
#      -- needs to return stats if collect_stats is True for backwards
#      compatibility.
# region_size does not get used below...
def do_levmarq_all_particle_groups(st, region_size=40, max_iter=2, damping=1.0,
        decrease_damp_factor=10., run_length=4, collect_stats=False, **kwargs):
    """Levenberg-Marquardt optimization for every particle in the state.

    See Also
    --------
        do_levmarq_particles : Levenberg-Marquardt optimization of a
            specified set of particles.

        do_levmarq : Levenberg-Marquardt optimization of the entire state;
            useful for optimizing global parameters.
    """
    optimize_all_particles(st, doshift=doshift, damp=damping,
                           damp_down=decrease_damp_factor, maxiter=max_iter)
    if collect_stats:
        # return lp.stats
        raise NotImplementedError


# TODO deprecate these down

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
    Other Parameters
    ----------------
        Any parameters passed to LMEngine.
    """
    # normal = direction / np.sqrt(np.dot(direction, direction))
    normals = np.array([d/np.sqrt(np.dot(d,d)) for d in directions])
    if np.isnan(normals).any():
        raise ValueError('`directions` must not be 0s or contain nan')
    obj = OptState(s, normals)
    lo = LMOptObj(obj, max_iter=max_iter, run_length=run_length, damping=
            damping, marquardt_damping=marquardt_damping, **kwargs)
    lo.do_run_1()
    if collect_stats:
        return lo.get_termination_stats()


def burn(s, n_loop=6, collect_stats=False, desc='', rz_order=0, fractol=1e-4,
        errtol=1e-2, mode='burn', max_mem=1e9, include_rad=True,
        do_line_min='default', partial_log=False, dowarn=True):
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
            performance. Default is False.
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
            Fractional change in error at which to terminate. Default 1e-4
        errtol : Float, optional
            Absolute change in error at which to terminate. Default 1e-2
        mode : {'burn', 'do-particles', or 'polish'}, optional
            What mode to optimize with.
            * 'burn'          : Your state is far from the minimum.
            * 'do-particles'  : Positions far from minimum, globals well-fit.
            * 'polish'        : The state is close to the minimum.
            'burn' is the default. Only `polish` will get to the global
            minimum.
        max_mem : Numeric, optional
            The maximum amount of memory allowed for the optimizers' J's,
            for both particles & globals. Default is 1e9, i.e. 1GB per
            optimizer.
        do_line_min : Bool or 'default', optional
            Set to True to do an additional, third optimization per loop
            which optimizes along the subspace spanned by the last 3 steps
            of the burn()'s trajectory. In principle this should signifi-
            cantly speed up the convergence; in practice it sometimes does,
            sometimes doesn't. Default is 'default', which picks by mode:
            * 'burn'          : False
            * 'do-particles'  : False
            * 'polish'        : True
        dowarn : Bool, optional
            Whether to log a warning if termination results from finishing
            loops rather than from convergence. Default is True.

    Returns
    -------
        dictionary
            Dictionary of convergence information. Contains whether the
            optimization has converged (key ``'converged'``), the values of the
            state after each loop (key ``'all_loop_values'``).
            The values of the state's parameters after each part of the
            loop: globals, particles, linemin. If ``collect_stats`` is set,
            then also contains lists of termination dicts from globals,
            particles, and line minimization (keys ``'global_stats'``,
            ``'particle_stats'``, and ``'line_stats``').

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
    * burn            : lm.do_run_2(), lp.do_run_2(). No psf, 2 loops on lm.
    * do-particles    : lp.do_run_2(), scales for ilm, bkg's
    * polish          : lm.do_run_2(), lp.do_run_2(). Everything, 1 loop each.
    where lm is a globals LMGlobals instance, and lp a
    LMParticleGroupCollection instance.
    """
    # It would be nice if some of these magic #'s (region size,
    # num_eig_dirs, etc) were calculated in a good way. FIXME
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
                s.get('psf').params + ['zscale'])  # FIXME explicit params
        glbl_nms = name_globals(s, remove_params=remove_params)

    all_lp_stats = []
    all_lm_stats = []
    all_line_stats = []
    all_loop_values = []

    _delta_vals = []  # storing the directions we've moved along for line min
    #2. Optimize
    CLOG.info('Start of loop %d:\t%f' % (0, s.error))
    for a in range(n_loop):
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
        CLOG.info('Globals,   loop {}:\t{}'.format(a, s.error))
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
        CLOG.info('Particles, loop {}:\t{}'.format(a, s.error))
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
            CLOG.info('Line min., loop {}:\t{}'.format(a, s.error))
            all_loop_values.append(s.values)

        #2d. terminate?
        new_err = s.error
        derr = start_err - new_err
        dobreak = (derr/new_err < fractol) or (derr < errtol)
        if dobreak:
            break

    if dowarn and (not dobreak):
        CLOG.warn('burn() did not converge; consider re-running')

    d = {'converged':dobreak, 'all_loop_values':all_loop_values}
    if collect_stats:
        d.update({'global_stats':all_lm_stats, 'particle_stats':all_lp_stats,
                'line_stats':all_line_stats})
    return d


def finish(s, desc='finish', n_loop=4, max_mem=1e9, separate_psf=True,
        fractol=1e-7, errtol=1e-3, dowarn=True):
    """
    Crawls slowly to the minimum-cost state.

    Blocks the global parameters into small enough sections such that each
    can be optimized separately while including all the pixels (i.e. no
    decimation). Optimizes the globals, then the psf separately if desired,
    then particles, then a line minimization along the step direction to
    speed up convergence.

    Parameters
    ----------
        s : :class:`peri.states.ImageState`
            The state to optimize
        desc : string, optional
            Description to append to the states.save() call every loop.
            Set to `None` to avoid saving. Default is `'finish'`.
        n_loop : Int, optional
            The number of times to loop over in the optimizer. Default is 4.
        max_mem : Numeric, optional
            The maximum amount of memory allowed for the optimizers' J's,
            for both particles & globals. Default is 1e9.
        separate_psf : Bool, optional
            If True, does the psf optimization separately from the rest of
            the globals, since the psf has a more tortuous fit landscape.
            Default is True.
        fractol : Float, optional
            Fractional change in error at which to terminate. Default 1e-4
        errtol : Float, optional
            Absolute change in error at which to terminate. Default 1e-2
        dowarn : Bool, optional
            Whether to log a warning if termination results from finishing
            loops rather than from convergence. Default is True.

    Returns
    -------
        dictionary
            Information about the optimization. Has two keys: ``'converged'``,
            a Bool which of whether optimization stopped due to convergence
            (True) or due to max number of iterations (False), and
            ``'loop_values'``, a [n_loop+1, N] ``numpy.ndarray`` of the
            state's values, at the start of optimization and at the end of
            each loop, before the line minimization.
    """
    values = [np.copy(s.state[s.params])]
    remove_params = s.get('psf').params if separate_psf else None  # FIXME explicit params
    globals = name_globals(s, remove_params=remove_params)
    #FIXME this could be done much better, since much of the globals such
    #as the ilm are local. Could be done with sparse matrices and/or taking
    #nearby globals in a group and using the update tile only as the slicer,
    #rather than the full residuals.
    gs = np.floor(max_mem / s.residuals.nbytes).astype('int')
    groups = [globals[a:a+gs] for a in range(0, len(globals), gs)]
    CLOG.info('Start  ``finish``:\t{}'.format(s.error))
    for a in range(n_loop):
        start_err = s.error
        #1. Min globals:
        for g in groups:
            do_levmarq(s, g, damping=0.1, decrease_damp_factor=20.,
                    max_iter=1, max_mem=max_mem, eig_update=False)
        if separate_psf:
            do_levmarq(s, remove_params, max_mem=max_mem, max_iter=4,
                    eig_update=False)
        CLOG.info('Globals,   loop {}:\t{}'.format(a, s.error))
        if desc is not None:
            states.save(s, desc=desc)
        #2. Min particles
        do_levmarq_all_particle_groups(s, max_iter=1, max_mem=max_mem)
        CLOG.info('Particles, loop {}:\t{}'.format(a, s.error))
        if desc is not None:
            states.save(s, desc=desc)
        #3. Append vals, line min:
        values.append(np.copy(s.state[s.params]))
        dv = (np.array(values[1:]) - np.array(values[0]))[-3:]
        do_levmarq_n_directions(s, dv, damping=1e-2, max_iter=2, errtol=3e-4)
        CLOG.info('Line min., loop {}:\t{}'.format(a, s.error))
        if desc is not None:
            states.save(s, desc=desc)
        #4. terminate?
        new_err = s.error
        derr = start_err - new_err
        dobreak = (derr/new_err < fractol) or (derr < errtol)
        if dobreak:
            break

    if dowarn and (not dobreak):
        CLOG.warn('finish() did not converge; consider re-running')
    return {'converged':dobreak, 'loop_values':np.array(values)}


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

    Other Parameters
    ----------------
        Any keyword arguments to be passed to the optimizer LMGlobals
        through do_levmarq.

    See Also
    --------
    do_levmarq : Levenberg-Marquardt minimization using a random subset
        of the image pixels.
    """
    #resetting the category to ilm:
    new_cat = new_comp.category
    new_comp.category = 'ilm'
    fake_s = states.ImageState(Image(old_comp.get().copy()), [new_comp], pad=0,
            mdl=mdl.SmoothFieldModel())
    do_levmarq(fake_s, new_comp.params, **kwargs)
    new_comp.category = new_cat
