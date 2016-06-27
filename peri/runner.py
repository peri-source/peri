"""
Comments:
If you start from featured globals it seems that slab etc is off enough where
particles get added in regions they shouldn't be (e.g. below the slab).
So maybe you do a polish or a burn before an addsubtract?
"""
import os
import Tkinter as tk
import tkFileDialog as tkfd
import numpy as np
import peri
from peri import initializers, util, models, states, logger
from peri.comp import ilms, objs, psfs, exactpsf, comp

RLOG = logger.log.getChild('runner')

import peri.opt.optimize as opt
import peri.opt.addsubtract as addsub

def locate_spheres(image, radius, dofilter=True, order=(7,7,7), invert=False):
    """
    Get an initial featuring of sphere positions in an image.

    Parameters:
    -----------
    image : `peri.util.Image` object
        Image object which defines the image file as well as the region.

    radius : float
        Radius of objects to find, in pixels

    dofilter : boolean
        Whether to remove the background before featuring. Doing so can often
        greatly increase the success of initial featuring and decrease later
        optimization time.

    zslab : float
        Location of the coverslip in z pixel distance.

    invert : boolean
        Whether to invert the image for featuring. Should be True is the image
        is bright particles on a dark background.

    Returns:
    --------
    positions : np.ndarray [N,3]
        Positions of the particles in order (z,y,x) in pixel units of the image.
    """
    # We just want a smoothed field model of the image so that the residuals
    # are simply the particles without other complications
    m = models.SmoothFieldModel()

    I = ilms.LegendrePoly2P1D(order=order, constval=image.get_image().mean())
    s = states.ImageState(image, [I], pad=0, mdl=m)

    if dofilter:
        opt.do_levmarq(s, s.params)

    return addsub.feature_guess(s, radius, invert=not invert)[0]

def get_initial_featuring(feature_diam, actual_rad=None, im_name=None, desc='',
        tile=None, invert=True, minmass=100.0, slab=None, min_rad=None,
         max_rad=None, max_mem=1e9, zscale=0.9):
    """
    Gets a completely-optimized state from an initial image of single-sized
    particles. The user interactively selects the image.

    Parameters
    ----------
        feature_diam : Int, odd
            The particle diameter for featuring, as passed to trackpy.locate
        actual_rad : Float
            The actual radius of the particles. Default is 0.5 * feature_diam

        desc : String
            A description to be inserted in saved state. The save name will
            be, e.g., '0.tif-peri-' + desc + 'initial-burn.pkl'. Default is ''
        tile : peri.util.Tile instance
            The tile of the raw image to be analyzed. Default is None, the
            entire image.

        invert : Bool
            Whether to invert the image for featuring, as passed to trackpy.
            Default is True.
        minmass : Float
            minmass for featuring, as passed to trackpy. Default is 100.

        slab : peri.comp.objs.Slab instance or None
            If not None, a slab corresponding to that in the image. Default
            is None.

        min_rad : Float
            The minimum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius smaller than this are identified
            as fake and removed. Default is 0.5 * actual_rad.
        max_rad : Float
            The maximum particle radius, as passed to addsubtract.add_subtract.
            Particles with a fitted radius larger than this are identified
            as fake and removed. Default is 1.5 * actual_rad, however you
            may find better results if you make this more stringent.

        max_mem : Numeric
            The maximum additional memory to use for the optimizers, as
            passed to optimize.burn. Default is 1e9.
    """
    if actual_rad is None:
        actual_rad = feature_diam * 0.5
    if min_rad is None:
        min_rad = 0.5 * actual_rad
    if max_rad is None:
        max_rad = 1.5 * actual_rad

    initial_dir = os.getcwd()
    wid = tk.Tk()
    wid.withdraw()
    if im_name is None:
        im_name = tkfd.askopenfilename(initialdir=initial_dir, title=
                'Select initial image for featuring')
        os.chdir(os.path.dirname(im_name))
    im = util.RawImage(im_name, tile=tile)

    f = peri.trackpy.locate(im.get_image()*255, feature_diam, invert=invert,
            minmass=minmass)
    npart = f['x'].size
    pos = np.zeros([npart,3])
    pos[:,0] = f['z']
    pos[:,1] = f['y']
    pos[:,2] = f['x']

    rad = np.ones(npart, dtype='float') * actual_rad
    if slab is not None:
        o = comp.ComponentCollection(
                [
                    objs.PlatonicSpheresCollection(pos, rad, zscale=zscale),
                    slab
                ],
                category='obj'
            )
    else:
        o = objs.PlatonicSpheresCollection(pos, rad, zscale=zscale)

    p = exactpsf.FixedSSChebLinePSF()
    i = ilms.BarnesStreakLegPoly2P1D(npts=(200,120,80,50,30,30,30,30,30,30,30),
            zorder=9)
    b = ilms.LegendrePoly2P1D(order=(9,3,5), category='bkg')
    c = comp.GlobalScalar('offset', 0.0)
    s = states.ImageState(im, [o, i, b, c, p])
    RLOG.info('State Created.')

    opt.do_levmarq(s, ['ilm-scale'], max_iter=1, run_length=6, max_mem=max_mem)
    states.save(s, desc=desc+'-initial')

    RLOG.info('Initial burn:')
    opt.burn(s, mode='burn', n_loop=4, ftol=1, desc=desc+'initial-burn',
            max_mem=max_mem)

    RLOG.info('Start add-subtract')
    addsub.add_subtract(s, tries=30, min_rad=min_rad, max_rad=max_rad,
            invert=invert)
    states.save(s, desc=desc+'initial-addsub')

    RLOG.info('Final polish:')
    opt.burn(s, mode='polish', n_loop=7, ftol=1e-3, desc=desc+'addsub-polish',
            max_mem=max_mem)

    os.chdir(initial_dir)
    return s

def translate_featuring(state_name=None, im_name=None, desc='', invert=True,
        min_rad='calc', max_rad='calc', max_mem=1e9, do_polish=True):
    """
    Translates one optimized state into another image where the particles
    have moved by a small amount (~1 particle radius).
    Returns a completely-optimized state. The user can interactively selects
    the initial state and the second raw image.

    Parameters
    ----------
        desc : String
            A description to be inserted in saved state. The save name will
            be, e.g., '0.tif-peri-' + desc + 'initial-burn.pkl'. Default is ''

        invert : Bool
            Whether to invert the image for featuring, as passed to
            addsubtract.add_subtract. Default is True.
        min_rad : Float
            The minimum particle radius, as passed to addsubtract.add_subtract.
            Default is 'calc', but a known physical value will give better
            answers.
        max_rad : Float
            The maximum particle radius, as passed to addsubtract.add_subtract.
            Default is 'calc', but a known physical value will give better
            answers.

        max_mem : Numeric
            The maximum additional memory to use for the optimizers, as
            passed to optimize.burn. Default is 1e9.
    """
    initial_dir = os.getcwd()
    wid = tk.Tk()
    wid.withdraw()
    if state_name is None:
        state_name = tkfd.askopenfilename(initialdir=initial_dir, title=
                'Select pre-featured state')
        os.chdir(os.path.dirname(state_name))

    if im_name is None:
        im_name = tkfd.askopenfilename(initialdir=initial_dir, title=
                'Select new image')

    s = states.load(state_name)
    im = util.RawImage(im_name, tile=s.image.tile)  #should have get_scale? FIXME

    s.set_image(im)
    _translate_particles(s, desc, max_mem, min_rad, max_rad, invert,
            do_polish=do_polish)
    return s

def get_particles_featuring(feature_diam, state_name=None, im_name=None,
        actual_rad=None, desc='', invert=True, min_rad='calc', max_rad='calc',
         max_mem=1e9, zscale=0.9, minmass=100):
    """
    Runs trackpy.locate on an image, sets the globals from a previous state,
    calls _translate_particles

    Basically I'm trying to cover 4 options, one of each:
        (Previously featured state,     No previously featured state)
        (using previous positions,      no previous positions)
    (no, no)    = get_initial_featuring
    (yes, no)   = get_particle_featuring
    (yes, yes)  = translate_featuring
    (no, yes)   = stupid, not doing. Or should I???

    These do:
        (use globals to start,          start from nothing)
        (use positions to start,        start from trackpy)
    """
    initial_dir = os.getcwd()
    # here to xxx in a sub-function? It's in this, translate_featuring
    initial_dir = os.getcwd()
    wid = tk.Tk()
    wid.withdraw()
    if state_name is None:
        state_name = tkfd.askopenfilename(initialdir=initial_dir, title=
                'Select pre-featured state')
        os.chdir(os.path.dirname(state_name))

    if im_name is None:
        im_name = tkfd.askopenfilename(initialdir=initial_dir, title=
                'Select new image')

    s = states.load(state_name)
    # xxx

    if actual_rad == None:
        actual_rad = np.median(s.obj_get_radii())  #or 2 x feature_diam?
    im = util.RawImage(im_name, tile=s.image.tile)  #should have get_scale? FIXME
    f = peri.trackpy.locate(im.get_image()*255, feature_diam, invert=invert,
            minmass=minmass)
    npart = f['x'].size
    pos = np.zeros([npart,3])
    pos[:,0] = f['z']
    pos[:,1] = f['y']
    pos[:,2] = f['x']
    #Right now there is not a good way to translate, so:

    slab = s.get('obj').comps[1] #FIXME I don't think this will always be correct
    sph = objs.PlatonicSpheresCollection(pos=pos, rad=actual_rad, zscale=
            s.state['zscale'])
    o = comp.ComponentCollection([sph, slab], category='obj')
    s.set('obj', o); s.reset()

    s.set_image(im)
    _translate_particles(s, desc, max_mem, min_rad, max_rad, invert,
        do_polish=True)
    return s

def _translate_particles(s, desc, max_mem, min_rad, max_rad, invert,
        do_polish=True):
    RLOG.info('Translate Particles:')
    opt.burn(s, mode='do-particles', n_loop=2, ftol=1, desc=desc+
            'translate-particles', max_mem=max_mem, include_rad=False)
    opt.burn(s, mode='do-particles', n_loop=2, ftol=1, desc=desc+
            'translate-particles', max_mem=max_mem, include_rad=True)

    RLOG.info('Start add-subtract')
    addsub.add_subtract(s, tries=30, min_rad=min_rad, max_rad=max_rad,
        invert=invert)
    states.save(s, desc=desc+'translate-addsub')

    if do_polish:
        RLOG.info('Final polish:')
        opt.burn(s, mode='polish', n_loop=6, ftol=1e-3, desc=desc+
            'addsub-polish', max_mem=max_mem)
