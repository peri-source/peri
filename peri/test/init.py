import numpy as np
import scipy as sp

from peri import states, models
from peri.test import poissondisks

def _seed_or_not(seed=None):
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

def _toarr(i):
    if not hasattr(i, '__iter__'):
        i = np.array((i,)*3)
    return np.array(i)

#=============================================================================
# Initialization methods to go full circle
#=============================================================================
def create_state(image, pos, rad, sigma=0.05, slab=None, ignoreimage=False,
        psftype='gauss4d', ilmtype='poly3d', bkgtype=None,
        psfargs={}, ilmargs={}, objargs={}, stateargs={}):
    """
    Create a state from a blank image, set of pos and radii

    Parameters:
    -----------
    image : ndarray or `peri.util.RawImage`
        raw confocal image with which to compare.

    pos : initial conditions for positions (in raw image coordinates)
    rad : initial conditions for radii array (can be scalar)
    sigma : float, noise level

    slab : float
        z-position of the microscope slide in the image (pixel units)

    pad_for_extra : boolean
        whether to include extra blank particles for sampling over

    ignoreimage : boolean
        whether to refer to `image` just for shape. really use the model
        image as the true raw image

    psftype : ['identity', 'gauss2d', 'gauss3d', 'gauss4d', 'gaussian_pca',
               'linescan', 'cheb-linescan', 'cheb-linescan-fixedss']
        which type of psf to use in the state

    ilmtype : ['poly3d', 'leg3d', 'cheb2p1d', 'poly2p1d', 'leg2p1d',
               'barnesleg2p1d', 'barnesleg2p1dx']
        which type of illumination field

    psfargs : arguments to the psf object
    ilmargs: arguments to the ilm object
    objargs: arguments to the sphere collection object
    stateargs : dictionary of arguments to pass to state
    """
    tpsfs = ['gauss3d', 'gauss4d']
    tilms = ['poly3d', 'leg3d', 'cheb2p1d', 'poly2p1d', 'leg2p1d']

    # first, decide if we got a RawImage or just an image array
    rawimage = None
    if isinstance(image, RawImage):
        rawimage = image
        image = rawimage.get_image()

    # we accept radius as a scalar, so check if we need to expand it
    if not hasattr(rad, '__iter__'):
        rad = rad*np.ones(pos.shape[0])

    # let's create the padded image required of the state
    pad = stateargs.get('pad', 24)
    image, pos, rad = states.prepare_for_state(image, pos, rad, pad=pad)

    # create some default arguments for the image components
    def_obj = {'pos': pos, 'rad': rad, 'shape': image.shape}
    def_psf = {'shape': image.shape}
    def_ilm = {'order': (1,1,1), 'shape': image.shape}

    # create the SphereCollectionRealSpace object
    def_obj.update(objargs)

    nfake = None
    if pad_extra_particles:
        nfake = pos.shape[0]
        pos, rad = pad_fake_particles(pos, rad, nfake)

    def_obj.update({'pad': nfake})
    obj = objs.SphereCollectionRealSpace(**def_obj)

    # setup the ilm based on the choice and arguments
    if ilmtype == 'poly3d':
        def_ilm.update(ilmargs)
        ilm = ilms.Polynomial3D(**def_ilm)
    elif ilmtype == 'leg3d':
        def_ilm.update(ilmargs)
        ilm = ilms.LegendrePoly3D(**def_ilm)
    elif ilmtype == 'cheb2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.ChebyshevPoly2P1D(**def_ilm)
    elif ilmtype == 'poly2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.Polynomial2P1D(**def_ilm)
    elif ilmtype == 'leg2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.LegendrePoly2P1D(**def_ilm)
    elif ilmtype == 'barnesleg2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.BarnesStreakLegPoly2P1D(**def_ilm)
    elif ilmtype == 'barnesleg2p1dx':
        def_ilm.update(ilmargs)
        ilm = ilms.BarnesStreakLegPoly2P1DX3(**def_ilm)
    else:
        raise AttributeError("ilmtype not one of supplied options, see help")

    # setup the psf based on the choice and arguments
    if psftype == 'identity':
        def_psf.update({'params': (1.0, 1.0)})
        def_psf.update(psfargs)
        psf = psfs.IdentityPSF(**def_psf)
    elif psftype == 'gauss2d':
        def_psf.update({'params': (2.0, 4.0)})
        def_psf.update(psfargs)
        psf = psfs.AnisotropicGaussian(**def_psf)
    elif psftype == 'gauss3d':
        def_psf.update({'params': (2.0, 1.0, 4.0)})
        def_psf.update(psfargs)
        psf = psfs.AnisotropicGaussianXYZ(**def_psf)
    elif psftype == 'gauss4d':
        def_psf.update({'params': (1.5, 0.7, 3.0)})
        def_psf.update(psfargs)
        psf = psfs.Gaussian4DPoly(**def_psf)
    elif psftype == 'linescan':
        def_psf.update({
            'zrange': (0, image.shape[0]),
            'cutoffval': 1./255,
            'measurement_iterations': 3,
        })
        def_psf.update(psfargs)
        psf = exactpsf.ExactLineScanConfocalPSF(**def_psf)
    elif psftype == 'cheb-linescan':
        def_psf.update({
            'zrange': (0, image.shape[0]),
            'cutoffval': 1./255,
            'measurement_iterations': 3,
        })
        def_psf.update(psfargs)
        psf = exactpsf.ChebyshevLineScanConfocalPSF(**def_psf)
    elif psftype == 'cheb-linescan-fixedss':
        def_psf.update({
            'support_size': [31, 17, 29],
            'zrange': (0, image.shape[0]),
            'cutoffval': 1./255,
            'measurement_iterations': 3,
        })
        def_psf.update(psfargs)
        psf = exactpsf.FixedSSChebLinePSF(**def_psf)
    else:
        raise AttributeError("psftype not one of supplied options, see help")

    if slab is not None:
        slab = objs.Slab(zpos=slab+pad, shape=image.shape)
    if rawimage is not None:
        image = rawimage

    stateargs.update({'sigma': sigma})
    stateargs.update({'slab': slab})
    s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm, **stateargs)

    if ignoreimage:
        s.model_to_true_image()
    return s

#=======================================================================
# Generating fake data
#=======================================================================
def create_state_random_packing(imsize, radius=5.0, phi=0.5, seed=None, **kwargs):
    """
    Creates a random packing of spheres and generates the state

    Parameters:
    -----------
    imsize : tuple, array_like, or integer
        the unpadded image size to fill with particles

    radius : float
        radius of particles to add

    seed : integer
        set the seed if desired

    *args, **kwargs : see create_state
    """
    _seed_or_not(seed)
    imsize = _toarr(imsize)

    disks = poissondisks.DiskCollection(imsize-radius, 2*radius)
    pos = disks.get_positions() + radius/4
    
    return create_state(np.zeros(imsize), pos, rad, ignoreimage=True, **kwargs)

def create_single_particle_state(imsize, radius=5.0, seed=None, **kwargs):
    """
    Creates a single particle state

    Parameters:
    -----------
    imsize : tuple, array_like, or integer
        the unpadded image size to fill with particles

    radius : float
        radius of particles to add

    seed : integer
        set the seed if desired

    *args, **kwargs : see create_state
    """
    _seed_or_not(seed)
    imsize = _toarr(imsize)

    pos = imsize.reshape(-1,3)/2.0
    rad = radius

    return create_state(np.zeros(imsize), pos, rad, ignoreimage=True, **kwargs)

def create_two_particle_state(imsize, radius=5.0, delta=1.0, seed=None, axis='x', **kwargs):
    """
    Creates a two particle state

    Parameters:
    -----------
    imsize : tuple, array_like, or integer
        the unpadded image size to fill with particles

    radius : float
        radius of particles to add

    delta : float
        separation between the two particles

    seed : integer
        set the seed if desired

    *args, **kwargs : see create_state
    """
    _seed_or_not(seed)
    imsize = _toarr(imsize)

    comp = {'x': 2, 'y': 1, 'z': 0}
    t = float(radius)+float(delta)/2
    d = np.array([0.0, 0.0, 0.0])
    d[comp[axis]] = t

    pos = np.array([imsize/2 - d, imsize/2 + d]).reshape(-1,3)
    rad = np.array([radius, radius])

    return create_state(np.zeros(imsize), pos, rad, ignoreimage=True, **kwargs)

