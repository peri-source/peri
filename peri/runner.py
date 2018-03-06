"""
Basically I'm trying to cover 4 options, one of each:
    (Previously featured state?,     Use previous positions?)
    ---------------------------------------------------------
    (no, no)    = get_initial_featuring
    (yes, no)   = get_particle_featuring
    (yes, yes)  = translate_featuring
    (no, yes)   = feature_from_pos_rad

These do:
    (use globals to start,          start from nothing)
    (use positions to start,        start from trackpy)
"""
from future import standard_library
standard_library.install_aliases()
from builtins import range

import os
try:
    import tkinter as tk
    import tkinter.filedialog as tkfd
except ImportError:
    import Tkinter as tk
    import tkFileDialog as tkfd
import numpy as np
import peri
from peri import initializers, util, models, states, logger
from peri.comp import ilms
import peri.opt.optimize as opt
import peri.opt.addsubtract as addsub

RLOG = logger.log.getChild('runner')


def locate_spheres(image, feature_rad, dofilter=False, order=(3 ,3, 3),
                    trim_edge=True, **kwargs):
    """
    Get an initial featuring of sphere positions in an image.

    Parameters
    -----------
    image : :class:`peri.util.Image` object
        Image object which defines the image file as well as the region.

    feature_rad : float
        Radius of objects to find, in pixels. This is a featuring radius
        and not a real radius, so a better value is frequently smaller
        than the real radius (half the actual radius is good). If ``use_tp``
        is True, then the twice ``feature_rad`` is passed as trackpy's
        ``diameter`` keyword.

    dofilter : boolean, optional
        Whether to remove the background before featuring. Doing so can
        often greatly increase the success of initial featuring and
        decrease later optimization time. Filtering functions by fitting
        the image to a low-order polynomial and featuring the residuals.
        In doing so, this will change the mean intensity of the featured
        image and hence the good value of ``minmass`` will change when
        ``dofilter`` is True. Default is False.

    order : 3-element tuple, optional
        If `dofilter`, the 2+1D Leg Poly approximation to the background
        illumination field. Default is (3,3,3).

    Other Parameters
    ----------------
    invert : boolean, optional
        Whether to invert the image for featuring. Set to True if the
        image is dark particles on a bright background. Default is True
    minmass : Float or None, optional
        The minimum mass/masscut of a particle. Default is None, which
        calculates internally.
    use_tp : Bool, optional
        Whether or not to use trackpy. Default is False, since trackpy
        cuts out particles at the edge.

    Returns
    --------
    positions : np.ndarray [N,3]
        Positions of the particles in order (z,y,x) in image pixel units.

    Notes
    -----
    Optionally filters the image by fitting the image I(x,y,z) to a
    polynomial, then subtracts this fitted intensity variation and uses
    centroid methods to find the particles.
    """
    # We just want a smoothed field model of the image so that the residuals
    # are simply the particles without other complications
    m = models.SmoothFieldModel()
    I = ilms.LegendrePoly2P1D(order=order, constval=image.get_image().mean())
    s = states.ImageState(image, [I], pad=0, mdl=m)
    if dofilter:
        opt.do_levmarq(s, s.params)
    pos = addsub.feature_guess(s, feature_rad, trim_edge=trim_edge, **kwargs)[0]
    return pos


def get_initial_featuring(statemaker, feature_rad, actual_rad=None,
        im_name=None, tile=None, invert=True, desc='', use_full_path=False,
        featuring_params={}, statemaker_kwargs={}, **kwargs):
    """
    Completely optimizes a state from an image of roughly monodisperse
    particles.

    The user can interactively select the image. The state is periodically
    saved during optimization, with different filename for different stages
    of the optimization.

    Parameters
    ----------
        statemaker : Function
            A statemaker function. Given arguments `im` (a
            :class:`~peri.util.Image`), `pos` (numpy.ndarray), `rad` (ndarray),
            and any additional `statemaker_kwargs`, must return a
            :class:`~peri.states.ImageState`.  There is an example function in
            scripts/statemaker_example.py
        feature_rad : Int, odd
            The particle radius for featuring, as passed to locate_spheres.
        actual_rad : Float, optional
            The actual radius of the particles. Default is feature_rad
        im_name : string, optional
            The file name of the image to load. If not set, it is selected
            interactively through Tk.
        tile : :class:`peri.util.Tile`, optional
            The tile of the raw image to be analyzed. Default is None, the
            entire image.
        invert : Bool, optional
            Whether to invert the image for featuring, as passed to trackpy.
            Default is True.
        desc : String, optional
            A description to be inserted in saved state. The save name will
            be, e.g., '0.tif-peri-' + desc + 'initial-burn.pkl'. Default is ''
        use_full_path : Bool, optional
            Set to True to use the full path name for the image. Default
            is False.
        featuring_params : Dict, optional
            kwargs-like dict of any additional keyword arguments to pass to
            ``get_initial_featuring``, such as ``'use_tp'`` or ``'minmass'``.
            Default is ``{}``.
        statemaker_kwargs : Dict, optional
            kwargs-like dict of any additional keyword arguments to pass to
            the statemaker function. Default is ``{}``.

    Other Parameters
    ----------------
        max_mem : Numeric
            The maximum additional memory to use for the optimizers, as
            passed to optimize.burn. Default is 1e9.
        min_rad : Float, optional
            The minimum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius smaller than this are identified
            as fake and removed. Default is 0.5 * actual_rad.
        max_rad : Float, optional
            The maximum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius larger than this are identified
            as fake and removed. Default is 1.5 * actual_rad, however you
            may find better results if you make this more stringent.
        rz_order : int, optional
            If nonzero, the order of an additional augmented rscl(z)
            parameter for optimization. Default is 0; i.e. no rscl(z)
            optimization.
        zscale : Float, optional
            The zscale of the image. Default is 1.0

    Returns
    -------
        s : :class:`peri.states.ImageState`
            The optimized state.

    See Also
    --------
        feature_from_pos_rad    : Using a previous state's globals and
            user-provided positions and radii as an initial guess,
            completely optimizes a state.

        get_particle_featuring  : Using a previous state's globals and
            positions as an initial guess, completely optimizes a state.

        translate_featuring     : Use a previous state's globals and
            centroids methods for an initial particle guess, completely
            optimizes a state.

    Notes
    -----
    Proceeds by centroid-featuring the image for an initial guess of
    particle positions, then optimizing the globals + positions until
    termination as called in _optimize_from_centroid.
    The ``Other Parameters`` are passed to _optimize_from_centroid.
    """
    if actual_rad is None:
        actual_rad = feature_rad

    _,  im_name = _pick_state_im_name('', im_name, use_full_path=use_full_path)
    im = util.RawImage(im_name, tile=tile)

    pos = locate_spheres(im, feature_rad, invert=invert, **featuring_params)
    if np.size(pos) == 0:
        msg = 'No particles found. Try using a smaller `feature_rad`.'
        raise ValueError(msg)

    rad = np.ones(pos.shape[0], dtype='float') * actual_rad
    s = statemaker(im, pos, rad, **statemaker_kwargs)
    RLOG.info('State Created.')
    if desc is not None:
        states.save(s, desc=desc+'initial')
    optimize_from_initial(s, invert=invert, desc=desc, **kwargs)
    return s


def feature_from_pos_rad(statemaker, pos, rad, im_name=None, tile=None,
        desc='', use_full_path=False, statemaker_kwargs={}, **kwargs):
    """
    Gets a completely-optimized state from an image and an initial guess of
    particle positions and radii.

    The state is periodically saved during optimization, with different
    filename for different stages of the optimization. The user can select
    the image.

    Parameters
    ----------
        statemaker : Function
            A statemaker function. Given arguments `im` (a
            :class:`~peri.util.Image`), `pos` (numpy.ndarray), `rad` (ndarray),
            and any additional `statemaker_kwargs`, must return a
            :class:`~peri.states.ImageState`.  There is an example function in
            scripts/statemaker_example.py
        pos : [N,3] element numpy.ndarray.
            The initial guess for the N particle positions.
        rad : N element numpy.ndarray.
            The initial guess for the N particle radii.
        im_name : string or None, optional
            The filename of the image to feature. Default is None, in which
            the user selects the image.
        tile : :class:`peri.util.Tile`, optional
            A tile of the sub-region of the image to feature. Default is
            None, i.e. entire image.
        desc : String, optional
            A description to be inserted in saved state. The save name will
            be, e.g., '0.tif-peri-' + desc + 'initial-burn.pkl'. Default is ''
        use_full_path : Bool, optional
            Set to True to use the full path name for the image. Default
            is False.
        statemaker_kwargs : Dict, optional
            kwargs-like dict of any additional keyword arguments to pass to
            the statemaker function. Default is ``{}``.

    Other Parameters
    ----------------
        max_mem : Numeric
            The maximum additional memory to use for the optimizers, as
            passed to optimize.burn. Default is 1e9.
        min_rad : Float, optional
            The minimum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius smaller than this are identified
            as fake and removed. Default is 0.5 * actual_rad.
        max_rad : Float, optional
            The maximum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius larger than this are identified
            as fake and removed. Default is 1.5 * actual_rad, however you
            may find better results if you make this more stringent.
        invert : {'guess', True, False}
            Whether to invert the image for featuring, as passed to
            addsubtract.add_subtract. Default is to guess from the
            current state's particle positions.
        rz_order : int, optional
            If nonzero, the order of an additional augmented rscl(z)
            parameter for optimization. Default is 0; i.e. no rscl(z)
            optimization.
        zscale : Float, optional
            The zscale of the image. Default is 1.0

    Returns
    -------
        s : :class:`peri.states.ImageState`
            The optimized state.

    See Also
    --------
        get_initial_featuring   : Features an image from scratch, using
            centroid methods as initial particle locations.

        get_particle_featuring  : Using a previous state's globals and
            positions as an initial guess, completely optimizes a state.

        translate_featuring     : Use a previous state's globals and
            centroids methods for an initial particle guess, completely
            optimizes a state.

    Notes
    -----
    The ``Other Parameters`` are passed to _optimize_from_centroid.
    Proceeds by centroid-featuring the image for an initial guess of
    particle positions, then optimizing the globals + positions until
    termination as called in _optimize_from_centroid.
    """
    if np.size(pos) == 0:
        raise ValueError('`pos` is an empty array.')
    elif np.shape(pos)[1] != 3:
        raise ValueError('`pos` must be an [N,3] element numpy.ndarray.')
    _,  im_name = _pick_state_im_name('', im_name, use_full_path=use_full_path)
    im = util.RawImage(im_name, tile=tile)
    s = statemaker(im, pos, rad, **statemaker_kwargs)
    RLOG.info('State Created.')
    if desc is not None:
        states.save(s, desc=desc+'initial')
    optimize_from_initial(s, desc=desc, **kwargs)
    return s


def optimize_from_initial(s, max_mem=1e9, invert='guess', desc='', rz_order=3,
        min_rad=None, max_rad=None):
    """
    Optimizes a state from an initial set of positions and radii, without
    any known microscope parameters.

    Parameters
    ----------
        s : :class:`peri.states.ImageState`
            The state to optimize. It is modified internally and returned.
        max_mem : Numeric, optional
            The maximum memory for the optimizer to use. Default is 1e9 (bytes)
        invert : Bool or `'guess'`, optional
            Set to True if the image is dark particles on a bright
            background, False otherwise. Used for add-subtract. The
            default is to guess from the state's current particles.
        desc : String, optional
            An additional description to infix for periodic saving along the
            way. Default is the null string ``''``.
        rz_order : int, optional
            ``rz_order`` as passed to opt.burn. Default is 3
        min_rad : Float or None, optional
            The minimum radius to identify a particles as bad, as passed to
            add-subtract. Default is None, which picks half the median radii.
            If your sample is not monodisperse you should pick a different
            value.
        max_rad : Float or None, optional
            The maximum radius to identify a particles as bad, as passed to
            add-subtract. Default is None, which picks 1.5x the median radii.
            If your sample is not monodisperse you should pick a different
            value.

    Returns
    -------
        s : :class:`peri.states.ImageState`
            The optimized state, which is the same as the input ``s`` but
            modified in-place.
    """
    RLOG.info('Initial burn:')
    if desc is not None:
        desc_burn = desc + 'initial-burn'
        desc_polish = desc + 'addsub-polish'
    else:
        desc_burn, desc_polish = [None] * 2
    opt.burn(s, mode='burn', n_loop=3, fractol=0.1, desc=desc_burn,
            max_mem=max_mem, include_rad=False, dowarn=False)
    opt.burn(s, mode='burn', n_loop=3, fractol=0.1, desc=desc_burn,
            max_mem=max_mem, include_rad=True, dowarn=False)

    RLOG.info('Start add-subtract')
    rad = s.obj_get_radii()
    if min_rad is None:
        min_rad = 0.5 * np.median(rad)
    if max_rad is None:
        max_rad = 1.5 * np.median(rad)
    addsub.add_subtract(s, tries=30, min_rad=min_rad, max_rad=max_rad,
            invert=invert)
    if desc is not None:
        states.save(s, desc=desc + 'initial-addsub')

    RLOG.info('Final polish:')
    d = opt.burn(s, mode='polish', n_loop=8, fractol=3e-4, desc=desc_polish,
            max_mem=max_mem, rz_order=rz_order, dowarn=False)
    if not d['converged']:
        RLOG.warn('Optimization did not converge; consider re-running')
    return s


def translate_featuring(state_name=None, im_name=None, use_full_path=False,
        **kwargs):
    """
    Translates one optimized state into another image where the particles
    have moved by a small amount (~1 particle radius).

    Returns a completely-optimized state. The user can interactively
    selects the initial state and the second raw image. The state is
    periodically saved during optimization, with different filename for
    different stages of the optimization.

    Parameters
    ----------
        state_name : String or None, optional
            The name of the initially-optimized state. Default is None,
            which prompts the user to select the name interactively
            through a Tk window.
        im_name : String or None, optional
            The name of the new image to optimize. Default is None,
            which prompts the user to select the name interactively
            through a Tk window.
        use_full_path : Bool, optional
            Set to True to use the full path of the state instead of
            partial path names (e.g. /full/path/name/state.pkl vs
            state.pkl). Default is False

    Other Parameters
    ----------------
        max_mem : Numeric
            The maximum additional memory to use for the optimizers, as
            passed to optimize.burn. Default is 1e9.
        desc : String, optional
            A description to be inserted in saved state. The save name will
            be, e.g., '0.tif-peri-' + desc + 'initial-burn.pkl'. Default is ''
        min_rad : Float, optional
            The minimum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius smaller than this are identified
            as fake and removed. Default is 0.5 * actual_rad.
        max_rad : Float, optional
            The maximum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius larger than this are identified
            as fake and removed. Default is 1.5 * actual_rad, however you
            may find better results if you make this more stringent.
        invert : {True, False, 'guess'}
            Whether to invert the image for featuring, as passed to
            addsubtract.add_subtract. Default is to guess from the
            state's current particles.
        rz_order : int, optional
            If nonzero, the order of an additional augmented rscl(z)
            parameter for optimization. Default is 0; i.e. no rscl(z)
            optimization.
        do_polish : Bool, optional
            Set to False to only optimize the particles and add-subtract.
            Default is True, which then runs a polish afterwards.

    Returns
    -------
        s : :class:`peri.states.ImageState`
            The optimized state.

    See Also
    --------
        get_initial_featuring   : Features an image from scratch, using
            centroid methods as initial particle locations.

        feature_from_pos_rad    : Using a previous state's globals and
            user-provided positions and radii as an initial guess,
            completely optimizes a state.

        get_particle_featuring  : Using a previous state's globals and
            positions as an initial guess, completely optimizes a state.

    Notes
    -----
    The ``Other Parameters`` are passed to _translate_particles.
    Proceeds by:
        1. Optimize particle positions only.
        2. Optimize particle positions and radii only.
        3. Add-subtract missing and bad particles.
        4. If polish, optimize the illumination, background, and particles.
        5. If polish, optimize everything.
    """
    state_name, im_name = _pick_state_im_name(
            state_name, im_name, use_full_path=use_full_path)

    s = states.load(state_name)
    im = util.RawImage(im_name, tile=s.image.tile)

    s.set_image(im)
    _translate_particles(s, **kwargs)
    return s


def get_particles_featuring(feature_rad, state_name=None, im_name=None,
        use_full_path=False, actual_rad=None, invert=True, featuring_params={},
        **kwargs):
    """
    Combines centroid featuring with the globals from a previous state.

    Runs trackpy.locate on an image, sets the globals from a previous state,
    calls _translate_particles

    Parameters
    ----------
        feature_rad : Int, odd
            The particle radius for featuring, as passed to locate_spheres.

        state_name : String or None, optional
            The name of the initially-optimized state. Default is None,
            which prompts the user to select the name interactively
            through a Tk window.
        im_name : String or None, optional
            The name of the new image to optimize. Default is None,
            which prompts the user to select the name interactively
            through a Tk window.
        use_full_path : Bool, optional
            Set to True to use the full path of the state instead of
            partial path names (e.g. /full/path/name/state.pkl vs
            state.pkl). Default is False
        actual_rad : Float or None, optional
            The initial guess for the particle radii. Default is the median
            of the previous state.
        invert : Bool
            Whether to invert the image for featuring, as passed to
            addsubtract.add_subtract and locate_spheres. Set to False
            if the image is bright particles on a dark background.
            Default is True (dark particles on bright background).
        featuring_params : Dict, optional
            kwargs-like dict of any additional keyword arguments to pass to
            ``get_initial_featuring``, such as ``'use_tp'`` or ``'minmass'``.
            Default is ``{}``.


    Other Parameters
    ----------------
        max_mem : Numeric
            The maximum additional memory to use for the optimizers, as
            passed to optimize.burn. Default is 1e9.
        desc : String, optional
            A description to be inserted in saved state. The save name will
            be, e.g., '0.tif-peri-' + desc + 'initial-burn.pkl'. Default is ''
        min_rad : Float, optional
            The minimum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius smaller than this are identified
            as fake and removed. Default is 0.5 * actual_rad.
        max_rad : Float, optional
            The maximum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius larger than this are identified
            as fake and removed. Default is 1.5 * actual_rad, however you
            may find better results if you make this more stringent.
        rz_order : int, optional
            If nonzero, the order of an additional augmented rscl(z)
            parameter for optimization. Default is 0; i.e. no rscl(z)
            optimization.
        do_polish : Bool, optional
            Set to False to only optimize the particles and add-subtract.
            Default is True, which then runs a polish afterwards.

    Returns
    -------
        s : :class:`peri.states.ImageState`
            The optimized state.

    See Also
    --------
        get_initial_featuring   : Features an image from scratch, using
            centroid methods as initial particle locations.

        feature_from_pos_rad    : Using a previous state's globals and
            user-provided positions and radii as an initial guess,
            completely optimizes a state.

        translate_featuring     : Use a previous state's globals and
            centroids methods for an initial particle guess, completely
            optimizes a state.

    Notes
    -----
        The ``Other Parameters`` are passed to _translate_particles.
    Proceeds by:
        1. Find a guess of the particle positions through centroid methods.
        2. Optimize particle positions only.
        3. Optimize particle positions and radii only.
        4. Add-subtract missing and bad particles.
        5. If polish, optimize the illumination, background, and particles.
        6. If polish, optimize everything.
    """
    state_name, im_name = _pick_state_im_name(
            state_name, im_name, use_full_path=use_full_path)
    s = states.load(state_name)

    if actual_rad is None:
        actual_rad = np.median(s.obj_get_radii())
    im = util.RawImage(im_name, tile=s.image.tile)
    pos = locate_spheres(im, feature_rad, invert=invert, **featuring_params)
    _ = s.obj_remove_particle(np.arange(s.obj_get_radii().size))
    s.obj_add_particle(pos, np.ones(pos.shape[0])*actual_rad)

    s.set_image(im)
    _translate_particles(s, invert=invert, **kwargs)
    return s


def _pick_state_im_name(state_name, im_name, use_full_path=False):
    """
    If state_name or im_name is None, picks them interactively through Tk,
    and then sets with or without the full path.

    Parameters
    ----------
        state_name : {string, None}
            The name of the state. If None, selected through Tk.
        im_name : {string, None}
            The name of the image. If None, selected through Tk.
        use_full_path : Bool, optional
            Set to True to return the names as full paths rather than
            relative paths. Default is False (relative path).
    """
    initial_dir = os.getcwd()
    if (state_name is None) or (im_name is None):
        wid = tk.Tk()
        wid.withdraw()
    if state_name is None:
        state_name = tkfd.askopenfilename(
                initialdir=initial_dir, title='Select pre-featured state')
        os.chdir(os.path.dirname(state_name))

    if im_name is None:
        im_name = tkfd.askopenfilename(
                initialdir=initial_dir, title='Select new image')

    if (not use_full_path) and (os.path.dirname(im_name) != ''):
        im_path = os.path.dirname(im_name)
        os.chdir(im_path)
        im_name = os.path.basename(im_name)
    else:
        os.chdir(initial_dir)
    return state_name, im_name


def _translate_particles(s, max_mem=1e9, desc='', min_rad='calc',
        max_rad='calc', invert='guess', rz_order=0, do_polish=True):
    """
    Workhorse for translating particles. See get_particles_featuring for docs.
    """
    if desc is not None:
        desc_trans = desc + 'translate-particles'
        desc_burn = desc + 'addsub_burn'
        desc_polish = desc + 'addsub_polish'
    else:
        desc_trans, desc_burn, desc_polish = [None]*3
    RLOG.info('Translate Particles:')
    opt.burn(s, mode='do-particles', n_loop=4, fractol=0.1, desc=desc_trans,
            max_mem=max_mem, include_rad=False, dowarn=False)
    opt.burn(s, mode='do-particles', n_loop=4, fractol=0.05, desc=desc_trans,
            max_mem=max_mem, include_rad=True, dowarn=False)

    RLOG.info('Start add-subtract')
    addsub.add_subtract(s, tries=30, min_rad=min_rad, max_rad=max_rad,
        invert=invert)
    if desc is not None:
        states.save(s, desc=desc + 'translate-addsub')

    if do_polish:
        RLOG.info('Final Burn:')
        opt.burn(s, mode='burn', n_loop=3, fractol=3e-4, desc=desc_burn,
                max_mem=max_mem, rz_order=rz_order,dowarn=False)
        RLOG.info('Final Polish:')
        d = opt.burn(s, mode='polish', n_loop=4, fractol=3e-4, desc=desc_polish,
                max_mem=max_mem, rz_order=rz_order, dowarn=False)
        if not d['converged']:
            RLOG.warn('Optimization did not converge; consider re-running')


def link_zscale(st):
    """Links the state ``st`` psf zscale with the global zscale"""
    # FIXME should be made more generic to other parameters and categories
    psf = st.get('psf')
    psf.param_dict['zscale'] = psf.param_dict['psf-zscale']
    psf.params[psf.params.index('psf-zscale')] = 'zscale'
    psf.global_zscale = True
    psf.param_dict.pop('psf-zscale')
    st.trigger_parameter_change()
    st.reset()


def finish_state(st, desc='finish-state', invert='guess'):
    """
    Final optimization for the best-possible state.

    Runs a local add-subtract to capture any difficult-to-feature particles,
    then does another set of optimization designed to get to the best
    possible fit.

    Parameters
    ----------
        st : :class:`peri.states.ImageState`
            The state to finish
        desc : String, optional
            Description to intermittently save the state as, as passed to
            state.save. Default is `'finish-state'`.
        invert : {'guess', True, False}
            Whether to invert the image for featuring, as passed to
            addsubtract.add_subtract. Default is to guess from the
            state's current particles.

    See Also
    --------
        `peri.opt.addsubtract.add_subtract_locally`
        `peri.opt.optimize.finish`
    """
    for minmass in [None, 0]:
        for _ in range(3):
            npart, poses = addsub.add_subtract_locally(st, region_depth=7,
                    minmass=minmass, invert=invert)
            if npart == 0:
                break
    opt.finish(st, n_loop=1, separate_psf=True, desc=desc, dowarn=False)
    opt.burn(st, mode='polish', desc=desc, n_loop=2, dowarn=False)
    d = opt.finish(st, desc=desc, n_loop=4, dowarn=False)
    if not d['converged']:
        RLOG.warn('Optimization did not converge; consider re-running')

