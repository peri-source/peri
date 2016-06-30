import numpy as np
import scipy as sp

from peri import states, models, util
from peri.comp import objs

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
def create_state(image, pos, rad, slab=None, sigma=0.05, conf=models.conf_confocal):
    """
    Create a state from a blank image, set of pos and radii

    Parameters:
    -----------
    image : `peri.util.Image` object
        raw confocal image with which to compare.

    pos : initial conditions for positions (in raw image coordinates)
    rad : initial conditions for radii array (can be scalar)
    sigma : float, noise level

    slab : float
        z-position of the microscope slide in the image (pixel units)
    """
    # we accept radius as a scalar, so check if we need to expand it
    if not hasattr(rad, '__iter__'):
        rad = rad*np.ones(pos.shape[0])

    model = models.models[conf.get('model')]()

    # setup the components based on the configuration
    components = []
    for k,v in conf.get('comps', {}).iteritems():
        args = conf.get('args').get(k, {})
        comp = model.registry[k][v](**args)
        components.append(comp)

    sphs = objs.PlatonicSpheresCollection(pos, rad)
    if slab is not None:
        sphs = ComponentCollection([sphs, objs.Slab(zpos=slab+pad)], category='obj')
    components.append(sphs)

    s = states.ImageState(image, components, sigma=sigma)

    if isinstance(image, util.NullImage):
        s.model_to_data()
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
    
    return create_state(util.NullImage(shape=imsize), pos, rad, **kwargs)

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

    return create_state(util.NullImage(shape=imsize), pos, rad, **kwargs)

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

    return create_state(util.NullImage(shape=imsize), pos, rad, **kwargs)

