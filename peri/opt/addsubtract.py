from builtins import range, zip

import numpy as np
import scipy.ndimage as nd

import peri
from peri import initializers
from peri.util import Tile
import peri.opt.optimize as opt

from peri.logger import log
CLOG = log.getChild('addsub')

def feature_guess(st, rad, invert='guess', minmass=None, use_tp=False,
                  trim_edge=False, **kwargs):
    """
    Makes a guess at particle positions using heuristic centroid methods.

    Parameters
    ----------
    st : :class:`peri.states.State`
        The state to check adding particles to.
    rad : Float
        The feature size for featuring.
    invert : {'guess', True, False}, optional
        Whether to invert the image; set to True for there are dark
        particles on a bright background, False for bright particles.
        The default is to guess from the state's current particles.
    minmass : Float or None, optional
        The minimum mass/masscut of a particle. Default is ``None`` =
        calculated internally.
    use_tp : Bool, optional
        Whether or not to use trackpy. Default is ``False``, since trackpy
        cuts out particles at the edge.
    trim_edge : Bool, optional
        Whether to trim particles at the edge pixels of the image. Can be
        useful for initial featuring but is bad for adding missing particles
        as they are frequently at the edge. Default is ``False``.

    Returns
    -------
    guess : [N,3] numpy.ndarray
        The featured positions of the particles, sorted in order of decreasing
        feature mass.
    npart : Int
        The number of added particles.
    """
    # FIXME does not use the **kwargs, but needs b/c called with wrong kwargs
    if invert == 'guess':
        invert = guess_invert(st)
    if invert:
        im = 1 - st.residuals
    else:
        im = st.residuals
    return _feature_guess(im, rad, minmass=minmass, use_tp=use_tp,
                          trim_edge=trim_edge)


def _feature_guess(im, rad, minmass=None, use_tp=False, trim_edge=False):
    """Workhorse of feature_guess"""
    if minmass is None:
        # we use 1% of the feature size mass as a cutoff;
        # it's easier to remove than to add
        minmass = rad**3 * 4/3.*np.pi * 0.01
        # 0.03 is a magic number; works well
    if use_tp:
        diameter = np.ceil(2*rad)
        diameter += 1-(diameter % 2)
        df = peri.trackpy.locate(im, int(diameter), minmass=minmass)
        npart = np.array(df['mass']).size
        guess = np.zeros([npart, 3])
        guess[:, 0] = df['z']
        guess[:, 1] = df['y']
        guess[:, 2] = df['x']
        mass = df['mass']
    else:
        guess, mass = initializers.local_max_featuring(
            im, radius=rad, minmass=minmass, trim_edge=trim_edge)
        npart = guess.shape[0]
    # I want to return these sorted by mass:
    inds = np.argsort(mass)[::-1]  # biggest mass first
    return guess[inds].copy(), npart


def check_add_particles(st, guess, rad='calc', do_opt=True, im_change_frac=0.2,
                        min_derr='3sig', **kwargs):
    """
    Checks whether to add particles at a given position by seeing if adding
    the particle improves the fit of the state.

    Parameters
    ----------
    st : :class:`peri.states.State`
        The state to check adding particles to.
    guess : [N,3] list-like
        The positions of particles to check to add.
    rad : {Float, ``'calc'``}, optional.
        The radius of the newly-added particles. Default is ``'calc'``,
        which uses the states current radii's median.
    do_opt : Bool, optional
        Whether to optimize the particle position before checking if it
        should be kept. Default is True (optimizes position).
    im_change_frac : Float
        How good the change in error needs to be relative to the change in
        the difference image. Default is 0.2; i.e. if the error does not
        decrease by 20% of the change in the difference image, do not add
        the particle.
    min_derr : Float or '3sig'
        The minimal improvement in error to add a particle. Default
        is ``'3sig' = 3*st.sigma``.

    Returns
    -------
    accepts : Int
        The number of added particles
    new_poses : [N,3] list
        List of the positions of the added particles. If ``do_opt==True``,
        then these positions will differ from the input 'guess'.
    """
    # FIXME does not use the **kwargs, but needs b/c called with wrong kwargs
    if min_derr == '3sig':
        min_derr = 3 * st.sigma
    accepts = 0
    new_poses = []
    if rad == 'calc':
        rad = guess_add_radii(st)
    message = ('-'*30 + 'ADDING' + '-'*30 +
               '\n  Z\t  Y\t  X\t  R\t|\t ERR0\t\t ERR1')
    with log.noformat():
        CLOG.info(message)
    for a in range(guess.shape[0]):
        p0 = guess[a]
        absent_err = st.error
        absent_d = st.residuals.copy()
        ind = st.obj_add_particle(p0, rad)
        if do_opt:
            # the slowest part of this
            opt.do_levmarq_particles(
                st, ind, damping=1.0, max_iter=1, run_length=3,
                eig_update=False, include_rad=False)
        present_err = st.error
        present_d = st.residuals.copy()
        dont_kill = should_particle_exist(
                absent_err, present_err, absent_d, present_d,
                im_change_frac=im_change_frac, min_derr=min_derr)
        if dont_kill:
            accepts += 1
            p = tuple(st.obj_get_positions()[ind].ravel())
            r = tuple(st.obj_get_radii()[ind].ravel())
            new_poses.append(p)
            part_msg = '%2.2f\t%3.2f\t%3.2f\t%3.2f\t|\t%4.3f  \t%4.3f' % (
                    p + r + (absent_err, st.error))
            with log.noformat():
                CLOG.info(part_msg)
        else:
            st.obj_remove_particle(ind)
            if np.abs(absent_err - st.error) > 1e-4:
                raise RuntimeError('updates not exact?')
    return accepts, new_poses


def check_remove_particle(st, ind, im_change_frac=0.2, min_derr='3sig',
                          **kwargs):
    """
    Checks whether to remove particle 'ind' from state 'st'. If removing the
    particle increases the error by less than max( min_derr, change in image *
    im_change_frac), then the particle is removed.

    Parameters
    ----------
    st : :class:`peri.states.State`
        The state to check adding particles to.
    ind : Int
        The index of the particle to check to remove.
    im_change_frac : Float
        How good the change in error needs to be relative to the change in
        the difference image. Default is 0.2; i.e. if the error does not
        decrease by 20% of the change in the difference image, do not add
        the particle.
    min_derr : Float or '3sig'
        The minimal improvement in error to add a particle. Default is
        ``'3sig' = 3*st.sigma``.

    Returns
    -------
    killed : Bool
        Whether the particle was removed.
    p : Tuple
        The position of the removed particle.
    r : Tuple
        The radius of the removed particle.
    """
    # FIXME does not use the **kwargs, but needs b/c called with wrong kwargs
    if min_derr == '3sig':
        min_derr = 3 * st.sigma
    present_err = st.error
    present_d = st.residuals.copy()
    p, r = st.obj_remove_particle(ind)
    p = p[0]
    r = r[0]
    absent_err = st.error
    absent_d = st.residuals.copy()

    if should_particle_exist(absent_err, present_err, absent_d, present_d,
                             im_change_frac=im_change_frac, min_derr=min_derr):
        st.obj_add_particle(p, r)
        killed = False
    else:
        killed = True
    return killed, tuple(p), (r,)


def should_particle_exist(absent_err, present_err, absent_d, present_d,
                          im_change_frac=0.2, min_derr=0.1):
    """
    Checks whether or not adding a particle should be present.

    Parameters
    ----------
    absent_err : Float
        The state error without the particle.
    present_err : Float
        The state error with the particle.
    absent_d : numpy.ndarray
        The state residuals without the particle.
    present_d : numpy.ndarray
        The state residuals with the particle.
    im_change_frac : Float, optional
        How good the change in error needs to be relative to the change in
        the residuals. Default is 0.2; i.e. return False if the error does
        not decrease by 0.2 x the change in the residuals.
    min_derr : Float, optional
        The minimal improvement in error. Default is 0.1

    Returns
    -------
    Bool
        True if the errors is better with the particle present.
    """
    delta_im = np.ravel(present_d - absent_d)
    im_change = np.dot(delta_im, delta_im)
    err_cutoff = max([im_change_frac * im_change, min_derr])
    return (absent_err - present_err) >= err_cutoff


def add_missing_particles(st, rad='calc', tries=50, **kwargs):
    """
    Attempts to add missing particles to the state.

    Operates by:
    (1) featuring the difference image using feature_guess,
    (2) attempting to add the featured positions using check_add_particles.

    Parameters
    ----------
    st : :class:`peri.states.State`
        The state to check adding particles to.
    rad : Float or 'calc', optional
        The radius of the newly-added particles and of the feature size for
        featuring. Default is 'calc', which uses the median of the state's
        current radii.
    tries : Int, optional
        How many particles to attempt to add. Only tries to add the first
        ``tries`` particles, in order of mass. Default is 50.

    Other Parameters
    ----------------
    invert : Bool, optional
        Whether to invert the image. Default is ``True``, i.e. dark particles
    minmass : Float or None, optionals
        The minimum mass/masscut of a particle. Default is ``None``=calcualted
        by ``feature_guess``.
    use_tp : Bool, optional
        Whether to use trackpy in feature_guess. Default is False, since
        trackpy cuts out particles at the edge.

    do_opt : Bool, optional
        Whether to optimize the particle position before checking if it
        should be kept. Default is True (optimizes position).
    im_change_frac : Float, optional
        How good the change in error needs to be relative to the change
        in the difference image. Default is 0.2; i.e. if the error does
        not decrease by 20% of the change in the difference image, do
        not add the particle.

    min_derr : Float or '3sig', optional
        The minimal improvement in error to add a particle. Default
        is ``'3sig' = 3*st.sigma``.

    Returns
    -------
    accepts : Int
        The number of added particles
    new_poses : [N,3] list
        List of the positions of the added particles. If ``do_opt==True``,
        then these positions will differ from the input 'guess'.
    """
    if rad == 'calc':
        rad = guess_add_radii(st)

    guess, npart = feature_guess(st, rad, **kwargs)
    tries = np.min([tries, npart])

    accepts, new_poses = check_add_particles(
        st, guess[:tries], rad=rad, **kwargs)
    return accepts, new_poses


def remove_bad_particles(st, min_rad='calc', max_rad='calc', min_edge_dist=2.0,
                         check_rad_cutoff=[3.5, 15], check_outside_im=True,
                         tries=50, im_change_frac=0.2, **kwargs):
    """
    Removes improperly-featured particles from the state, based on a
    combination of particle size and the change in error on removal.

    Parameters
    -----------
    st : :class:`peri.states.State`
        The state to remove bad particles from.
    min_rad : Float, optional
        All particles with radius below min_rad are automatically deleted.
        Set to 'calc' to make it the median rad - 25* radius std.
        Default is 'calc'.

    max_rad : Float, optional
        All particles with radius above max_rad are automatically deleted.
        Set to 'calc' to make it the median rad + 15* radius std.
        Default is 'calc'.

    min_edge_dist : Float, optional
        All particles within min_edge_dist of the (padded) image
        edges are automatically deleted. Default is 2.0

    check_rad_cutoff : 2-element list of floats, optional
        Particles with radii < check_rad_cutoff[0] or > check_rad_cutoff[1]
        are checked if they should be deleted. Set to 'calc' to make it the
        median rad +- 3.5 * radius std. Default is [3.5, 15].

    check_outside_im : Bool, optional
        If True, checks if particles located outside the unpadded image
        should be deleted. Default is True.

    tries : Int, optional
        The maximum number of particles with radii < check_rad_cutoff
        to try to remove. Checks in increasing order of radius size.
        Default is 50.

    im_change_frac : Float, , optional
        Number between 0 and 1. If removing a particle decreases the
        error by less than im_change_frac*the change in the image, then
        the particle is deleted. Default is 0.2

    Returns
    -------
    removed: Int
        The cumulative number of particles removed.
    """
    is_near_im_edge = lambda pos, pad: (((pos + st.pad) < pad) | (pos >
            np.array(st.ishape.shape) + st.pad - pad)).any(axis=1)
    # returns True if the position is within 'pad' of the _outer_ image edge
    removed = 0
    attempts = 0

    n_tot_part = st.obj_get_positions().shape[0]
    q10 = int(0.1 * n_tot_part)  # 10% quartile
    r_sig = np.sort(st.obj_get_radii())[q10:-q10].std()
    r_med = np.median(st.obj_get_radii())
    if max_rad == 'calc':
        max_rad = r_med + 15*r_sig
    if min_rad == 'calc':
        min_rad = r_med - 25*r_sig
    if check_rad_cutoff == 'calc':
        check_rad_cutoff = [r_med - 7.5*r_sig, r_med + 7.5*r_sig]

    # 1. Automatic deletion:
    rad_wrong_size = np.nonzero(
            (st.obj_get_radii() < min_rad) | (st.obj_get_radii() > max_rad))[0]
    near_im_edge = np.nonzero(is_near_im_edge(st.obj_get_positions(),
                              min_edge_dist - st.pad))[0]
    delete_inds = np.unique(np.append(rad_wrong_size, near_im_edge)).tolist()
    delete_poses = st.obj_get_positions()[delete_inds].tolist()
    message = ('-'*27 + 'SUBTRACTING' + '-'*28 +
               '\n  Z\t  Y\t  X\t  R\t|\t ERR0\t\t ERR1')
    with log.noformat():
        CLOG.info(message)

    for pos in delete_poses:
        ind = st.obj_closest_particle(pos)
        old_err = st.error
        p, r = st.obj_remove_particle(ind)
        p = p[0]
        r = r[0]
        part_msg = '%2.2f\t%3.2f\t%3.2f\t%3.2f\t|\t%4.3f  \t%4.3f' % (
                tuple(p) + (r,) + (old_err, st.error))
        with log.noformat():
            CLOG.info(part_msg)
        removed += 1

    # 2. Conditional deletion:
    check_rad_inds = np.nonzero((st.obj_get_radii() < check_rad_cutoff[0]) |
                                (st.obj_get_radii() > check_rad_cutoff[1]))[0]
    if check_outside_im:
        check_edge_inds = np.nonzero(
            is_near_im_edge(st.obj_get_positions(), st.pad))[0]
        check_inds = np.unique(np.append(check_rad_inds, check_edge_inds))
    else:
        check_inds = check_rad_inds

    check_inds = check_inds[np.argsort(st.obj_get_radii()[check_inds])]
    tries = np.min([tries, check_inds.size])
    check_poses = st.obj_get_positions()[check_inds[:tries]].copy()
    for pos in check_poses:
        old_err = st.error
        ind = st.obj_closest_particle(pos)
        killed, p, r = check_remove_particle(
            st, ind, im_change_frac=im_change_frac)
        if killed:
            removed += 1
            check_inds[check_inds > ind] -= 1  # cleaning up indices....
            delete_poses.append(pos)
            part_msg = '%2.2f\t%3.2f\t%3.2f\t%3.2f\t|\t%4.3f  \t%4.3f' % (
                    p + r + (old_err, st.error))
            with log.noformat():
                CLOG.info(part_msg)
    return removed, delete_poses


def add_subtract(st, max_iter=7, max_npart='calc', max_mem=2e8,
                 always_check_remove=False, **kwargs):
    """
    Automatically adds and subtracts missing & extra particles.

    Operates by removing bad particles then adding missing particles on
    repeat, until either no particles are added/removed or after `max_iter`
    attempts.

    Parameters
    ----------
    st: :class:`peri.states.State`
        The state to add and subtract particles to.
    max_iter : Int, optional
        The maximum number of add-subtract loops to use. Default is 7.
        Terminates after either max_iter loops or when nothing has changed.
    max_npart : Int or 'calc', optional
        The maximum number of particles to add before optimizing the non-psf
        globals. Default is ``'calc'``, which uses 5% of the initial number
        of particles.
    max_mem : Int, optional
        The maximum memory to use for optimization after adding max_npart
        particles. Default is 2e8.
    always_check_remove : Bool, optional
        Set to True to always check whether to remove particles. If ``False``,
        only checks for removal while particles were removed on the previous
        attempt. Default is False.

    Other Parameters
    ----------------
    invert : Bool, optional
        ``True`` if the particles are dark on a bright background, ``False``
        if they are bright on a dark background. Default is ``True``.
    min_rad : Float, optional
        Particles with radius below ``min_rad`` are automatically deleted.
        Default is ``'calc'`` = median rad - 25* radius std.
    max_rad : Float, optional
        Particles with radius above ``max_rad`` are automatically deleted.
        Default is ``'calc'`` = median rad + 15* radius std, but you should
        change this for your particle sizes.

    min_edge_dist : Float, optional
        Particles closer to the edge of the padded image than this are
        automatically deleted. Default is 2.0.
    check_rad_cutoff : 2-element float list.
        Particles with ``radii < check_rad_cutoff[0]`` or ``> check...[1]``
        are checked if they should be deleted (not automatic). Default is
        ``[3.5, 15]``.
    check_outside_im : Bool, optional
        Set to True to check whether to delete particles whose positions are
        outside the un-padded image.

    rad : Float, optional
        The initial radius for added particles; added particles radii are
        not fit until the end of ``add_subtract``. Default is ``'calc'``,
        which uses the median radii of active particles.

    tries : Int, optional
        The number of particles to attempt to remove or add, per iteration.
        Default is 50.

    im_change_frac : Float, optional
        How good the change in error needs to be relative to the change in
        the difference image. Default is 0.2; i.e. if the error does not
        decrease by 20% of the change in the difference image, do not add
        the particle.

    min_derr : Float, optional
        The minimum change in the state's error to keep a particle in the
        image. Default is ``'3sig'`` which uses ``3*st.sigma``.

    do_opt : Bool, optional
        Set to False to avoid optimizing particle positions after adding.
    minmass : Float, optional
        The minimum mass for a particle to be identified as a feature,
        as used by trackpy. Defaults to a decent guess.

    use_tp : Bool, optional
        Set to True to use trackpy to find missing particles inside the
        image. Not recommended since trackpy deliberately cuts out particles
        at the edge of the image. Default is ``False``.

    Returns
    -------
    total_changed : Int
        The total number of adds and subtracts done on the data. Not the
        same as ``changed_inds.size`` since the same particle or particle
        index can be added/subtracted multiple times.
    added_positions : [N_added,3] numpy.ndarray
        The positions of particles that have been added at any point in the
        add-subtract cycle.
    removed_positions : [N_added,3] numpy.ndarray
        The positions of particles that have been removed at any point in
        the add-subtract cycle.

    Notes
    ------
    Occasionally after the intial featuring a cluster of particles is
    featured as 1 big particle. To fix these mistakes, it helps to set
    max_rad to a physical value. This removes the big particle and allows
    it to be re-featured by (several passes of) the adds.

    The added/removed positions returned are whether or not the position
    has been added or removed ever. It's possible that a position is
    added, then removed during a later iteration.
    """
    if max_npart == 'calc':
        max_npart = 0.05 * st.obj_get_positions().shape[0]

    total_changed = 0
    _change_since_opt = 0
    removed_poses = []
    added_poses0 = []
    added_poses = []

    nr = 1  # Check removal on the first loop
    for _ in range(max_iter):
        if (nr != 0) or (always_check_remove):
            nr, rposes = remove_bad_particles(st, **kwargs)
        na, aposes = add_missing_particles(st, **kwargs)
        current_changed = na + nr
        removed_poses.extend(rposes)
        added_poses0.extend(aposes)
        total_changed += current_changed
        _change_since_opt += current_changed
        if current_changed == 0:
            break
        elif _change_since_opt > max_npart:
            _change_since_opt *= 0
            CLOG.info('Start add_subtract optimization.')
            opt.do_levmarq(st, opt.name_globals(st, remove_params=st.get(
                    'psf').params), max_iter=1, run_length=4, num_eig_dirs=3,
                    max_mem=max_mem, eig_update_frequency=2, rz_order=0,
                    use_accel=True)
            CLOG.info('After optimization:\t{:.6}'.format(st.error))

    # Optimize the added particles' radii:
    for p in added_poses0:
        i = st.obj_closest_particle(p)
        opt.do_levmarq_particles(st, np.array([i]), max_iter=2, damping=0.3)
        added_poses.append(st.obj_get_positions()[i])
    return total_changed, np.array(removed_poses), np.array(added_poses)


def identify_misfeatured_regions(st, filter_size=5, sigma_cutoff=8.):
    """
    Identifies regions of missing/misfeatured particles based on the
    residuals' local deviation from uniform Gaussian noise.

    Parameters
    ----------
    st : :class:`peri.states.State`
        The state in which to identify mis-featured regions.

    filter_size : Int, best if odd.
        The size of the filter for calculating the local standard deviation;
        should approximately be the size of a poorly featured region in
        each dimension. Default is 5.

    sigma_cutoff : Float or `otsu`, optional
        The max allowed deviation of the residuals from what is expected,
        in units of the residuals' standard deviation. Lower means more
        sensitive, higher = less sensitive. Default is 8.0, i.e. one pixel
        out of every 7*10^11 is mis-identified randomly. In practice the
        noise is not Gaussian so there are still some regions mis-identified
        as improperly featured. Set to ```otsu``` to calculate this number
        based on an automatic Otsu threshold.

    Returns
    -------
    tiles : List of :class:`peri.util.Tile`
        Each tile is the smallest bounding tile that contains an improperly
        featured region. The list is sorted by the tile's volume.

    Notes
    -----
    Algorithm is
    1.  Create a field of the local standard deviation, as measured over
        a hypercube of size filter_size.
    2.  Find the maximum reasonable value of the field. [The field should
        be a random variable with mean of r.std() and standard deviation
        of ~r.std() / sqrt(N), where r is the residuals and N is the
        number of pixels in the hypercube.]
    3.  Label & Identify the misfeatured regions as portions where
        the local error is too large.
    4.  Parse the misfeatured regions into tiles.
    5.  Return the sorted tiles.
    The Otsu option to calculate the sigma cutoff works well for images
    that actually contain missing particles, returning a number similar
    to one calculated with a sigma cutoff. However, if the image is
    well-featured with Gaussian residuals, then the Otsu threshold
    splits the Gaussian down the middle instead of at the tails, which
    is very bad. So use with caution.
    """
    # 1. Field of local std
    r = st.residuals
    weights = np.ones([filter_size]*len(r.shape), dtype='float')
    weights /= weights.sum()
    f = np.sqrt(nd.filters.convolve(r*r, weights, mode='reflect'))

    # 2. Maximal reasonable value of the field.
    if sigma_cutoff == 'otsu':
        max_ok = initializers.otsu_threshold(f)
    else:
        # max_ok = f.mean() * (1 + sigma_cutoff / np.sqrt(weights.size))
        max_ok = f.mean() + sigma_cutoff * f.std()

    # 3. Label & Identify
    bad = f > max_ok
    labels, n = nd.measurements.label(bad)
    inds = []
    for i in range(1, n+1):
        inds.append(np.nonzero(labels == i))

    # 4. Parse into tiles
    tiles = [Tile(np.min(ind, axis=1), np.max(ind, axis=1)+1) for ind in inds]

    # 5. Sort and return
    volumes = [t.volume for t in tiles]
    return [tiles[i] for i in np.argsort(volumes)[::-1]]


def add_subtract_misfeatured_tile(
        st, tile, rad='calc', max_iter=3, invert='guess', max_allowed_remove=20,
        minmass=None, use_tp=False, **kwargs):
    """
    Automatically adds and subtracts missing & extra particles in a region
    of poor fit.

    Parameters
    ----------
    st: :class:`peri.states.State`
        The state to add and subtract particles to.
    tile : :class:`peri.util.Tile`
        The poorly-fit region to examine.
    rad : Float or 'calc', optional
        The initial radius for added particles; added particles radii are
        not fit until the end of add_subtract. Default is ``'calc'``, which
        uses the median radii of active particles.
    max_iter : Int, optional
        The maximum number of loops for attempted adds at one tile location.
        Default is 3.
    invert : {'guess', True, False}, optional
        Whether to invert the image for feature_guess -- True for dark
        particles on a bright background, False for bright particles. The
        default is to guess from the state's current particles.
    max_allowed_remove : Int, optional
        The maximum number of particles to remove. If the misfeatured tile
        contains more than this many particles, raises an error. If it
        contains more than half as many particles, logs a warning. If more
        than this many particles are added, they are optimized in blocks of
        ``max_allowed_remove``. Default is 20.

    Other Parameters
    ----------------
    im_change_frac : Float on [0, 1], optional.
        If adding or removing a particle decreases the error less than
        ``im_change_frac``*the change in the image, the particle is deleted.
        Default is 0.2.

    min_derr : {Float, ``'3sig'``}, optional
        The minimum change in the state's error to keep a particle in the
        image. Default is ``'3sig'`` which uses ``3*st.sigma``.

    do_opt : Bool, optional
        Set to False to avoid optimizing particle positions after adding
        them. Default is True.

    minmass : Float, optional
        The minimum mass for a particle to be identified as a feature, as
        used by trackpy. Defaults to a decent guess.

    use_tp : Bool, optional
        Set to True to use trackpy to find missing particles inside the
        image. Not recommended since trackpy deliberately cuts out particles
        at the edge of the image. Default is False.

    Outputs
    -------
    n_added : Int
        The change in the number of particles, i.e. ``n_added-n_subtracted``
    ainds: List of ints
        The indices of the added particles.

    Notes
    --------
    The added/removed positions returned are whether or not the
    position has been added or removed ever. It's possible/probably that
    a position is added, then removed during a later iteration.

    Algorithm is:
    1.  Remove all particles within the tile.
    2.  Feature and add particles to the tile.
    3.  Optimize the added particles positions only.
    4.  Run 2-3 until no particles have been added.
    5.  Optimize added particle radii
    Because all the particles are removed within a tile, it is important
    to set max_allowed_remove to a reasonable value. Otherwise, if the
    tile is the size of the image it can take a long time to remove all
    the particles and re-add them.
    """
    if rad == 'calc':
        rad = guess_add_radii(st)
    if invert == 'guess':
        invert = guess_invert(st)
    # 1. Remove all possibly bad particles within the tile.
    initial_error = np.copy(st.error)
    rinds = np.nonzero(tile.contains(st.obj_get_positions()))[0]
    if rinds.size >= max_allowed_remove:
        CLOG.fatal('Misfeatured region too large!')
        raise RuntimeError
    elif rinds.size >= max_allowed_remove/2:
        CLOG.warn('Large misfeatured regions.')
    elif rinds.size > 0:
        rpos, rrad = st.obj_remove_particle(rinds)

    # 2-4. Feature & add particles to the tile, optimize, run until none added
    n_added = -rinds.size
    added_poses = []
    for _ in range(max_iter):
        if invert:
            im = 1 - st.residuals[tile.slicer]
        else:
            im = st.residuals[tile.slicer]
        guess, _ = _feature_guess(im, rad, minmass=minmass, use_tp=use_tp)
        accepts, poses = check_add_particles(
                st, guess+tile.l, rad=rad, do_opt=True, **kwargs)
        added_poses.extend(poses)
        n_added += accepts
        if accepts == 0:
            break
    else:  # for-break-else
        CLOG.warn('Runaway adds or insufficient max_iter')

    # 5. Optimize added pos + rad:
    ainds = []
    for p in added_poses:
        ainds.append(st.obj_closest_particle(p))
    if len(ainds) > max_allowed_remove:
        for i in range(0, len(ainds), max_allowed_remove):
            opt.do_levmarq_particles(
                st, np.array(ainds[i:i + max_allowed_remove]),
                include_rad=True, max_iter=3)
    elif len(ainds) > 0:
        opt.do_levmarq_particles(st, ainds, include_rad=True, max_iter=3)

    # 6. Ensure that current error after add-subtracting is lower than initial
    did_something = (rinds.size > 0) or (len(ainds) > 0)
    if did_something & (st.error > initial_error):
        CLOG.info('Failed addsub, Tile {} -> {}'.format(
            tile.l.tolist(), tile.r.tolist()))
        if len(ainds) > 0:
            _ = st.obj_remove_particle(ainds)
        if rinds.size > 0:
            for p, r in zip(rpos.reshape(-1, 3), rrad.reshape(-1)):
                _ = st.obj_add_particle(p, r)
        n_added = 0
        ainds = []
    return n_added, ainds


def add_subtract_locally(st, region_depth=3, filter_size=5, sigma_cutoff=8,
                         **kwargs):
    """
    Automatically adds and subtracts missing particles based on local
    regions of poor fit.

    Calls identify_misfeatured_regions to identify regions, then
    add_subtract_misfeatured_tile on the tiles in order of size until
    region_depth tiles have been checked without adding any particles.

    Parameters
    ----------
    st: :class:`peri.states.State`
        The state to add and subtract particles to.
    region_depth : Int
        The minimum amount of regions to try; the algorithm terminates if
        region_depth regions have been tried without adding particles.

    Other Parameters
    ----------------
    filter_size : Int, optional
        The size of the filter for calculating the local standard deviation;
        should approximately be the size of a poorly featured region in each
        dimension. Best if odd. Default is 5.
    sigma_cutoff : Float, optional
        The max allowed deviation of the residuals from what is expected,
        in units of the residuals' standard deviation. Lower means more
        sensitive, higher = less sensitive. Default is 8.0, i.e. one pixel
        out of every ``7*10^11`` is mis-identified randomly. In practice the
        noise is not Gaussian so there are still some regions mis-
        identified as improperly featured.
    rad : Float or 'calc', optional
        The initial radius for added particles; added particles radii are
        not fit until the end of add_subtract. Default is ``'calc'``, which
        uses the median radii of active particles.
    max_iter : Int, optional
        The maximum number of loops for attempted adds at one tile location.
        Default is 3.
    invert : Bool, optional
        Whether to invert the image for feature_guess. Default is ``True``,
        i.e. dark particles on bright background.
    max_allowed_remove : Int, optional
        The maximum number of particles to remove. If the misfeatured tile
        contains more than this many particles, raises an error. If it
        contains more than half as many particles, throws a warning. If more
        than this many particles are added, they are optimized in blocks of
        ``max_allowed_remove``. Default is 20.
    im_change_frac : Float, between 0 and 1.
        If adding or removing a particle decreases the error less than
        ``im_change_frac *`` the change in the image, the particle is deleted.
        Default is 0.2.
    min_derr : Float
        The minimum change in the state's error to keep a particle in the
        image. Default is ``'3sig'`` which uses ``3*st.sigma``.
    do_opt : Bool, optional
        Set to False to avoid optimizing particle positions after adding
        them. Default is True
    minmass : Float, optional
        The minimum mass for a particle to be identified as a feature, as
        used by trackpy. Defaults to a decent guess.
    use_tp : Bool, optional
        Set to True to use trackpy to find missing particles inside the
        image. Not recommended since trackpy deliberately cuts out
        particles at the edge of the image. Default is False.
    max_allowed_remove : Int, optional
        The maximum number of particles to remove. If the misfeatured tile
        contains more than this many particles, raises an error. If it
        contains more than half as many particles, throws a warning. If more
        than this many particles are added, they are optimized in blocks of
        ``max_allowed_remove``. Default is 20.

    Returns
    -------
    n_added : Int
        The change in the number of particles; i.e the number added - number
        removed.
    new_poses : List
        [N,3] element list of the added particle positions.

    Notes
    -----
    Algorithm Description

    1. Identify mis-featured regions by how much the local residuals
       deviate from the global residuals, as measured by the standard
       deviation of both.
    2. Loop over each of those regions, and:

       a. Remove every particle in the current region.
       b. Try to add particles in the current region until no more
          can be added while adequately decreasing the error.
       c. Terminate if at least region_depth regions have been
          checked without successfully adding a particle.

    Because this algorithm is more judicious about chooosing regions to
    check, and more aggressive about removing particles in those regions,
    it runs faster and does a better job than the (global) add_subtract.
    However, this function usually does not work better as an initial add-
    subtract on an image, since (1) it doesn't check for removing small/big
    particles per se, and (2) when the poorly-featured regions of the image
    are large or when the fit is bad, it will remove essentially all of the
    particles, taking a long time. As a result, it's usually best to do a
    normal add_subtract first and using this function for tough missing or
    double-featured particles.
    """
    # 1. Find regions of poor tiles:
    tiles = identify_misfeatured_regions(
        st, filter_size=filter_size, sigma_cutoff=sigma_cutoff)
    # 2. Add and subtract in the regions:
    n_empty = 0
    n_added = 0
    new_poses = []
    for t in tiles:
        curn, curinds = add_subtract_misfeatured_tile(st, t, **kwargs)
        if curn == 0:
            n_empty += 1
        else:
            n_added += curn
            new_poses.extend(st.obj_get_positions()[curinds])
        if n_empty > region_depth:
            break  # some message or something?
    else:  # for-break-else
        pass
        # CLOG.info('All regions contained particles.')
        # something else?? this is not quite true
    return n_added, new_poses


def guess_invert(st):
    """Guesses whether particles are bright on a dark bkg or vice-versa

    Works by checking whether the intensity at the particle centers is
    brighter or darker than the average intensity of the image, by
    comparing the median intensities of each.

    Parameters
    ----------
    st : :class:`peri.states.ImageState`

    Returns
    -------
    invert : bool
        Whether to invert the image for featuring.
    """
    pos = st.obj_get_positions()
    pxinds_ar = np.round(pos).astype('int')
    inim = st.ishape.translate(-st.pad).contains(pxinds_ar)
    pxinds_tuple = tuple(pxinds_ar[inim].T)
    pxvals = st.data[pxinds_tuple]
    invert = np.median(pxvals) < np.median(st.data)  # invert if dark particles
    return invert


def guess_add_radii(st):
    current_radii = st.obj_get_radii()
    return np.percentile(current_radii, 20)

