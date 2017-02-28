from future.utils import iteritems

import numpy as np
import scipy as sp

from peri import states, models, util
from peri.comp import objs, ilms
from peri.test import nbody

def _seed_or_not(seed=None):
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

def _toarr(i):
    if not hasattr(i, '__iter__'):
        i = np.array((i,)*3)
    return np.array(i)

conf_cohen_hot = {
    'model': 'confocal-dyedfluid',
    'comps': {
        'psf': 'cheb-linescan-fixedss',
        'ilm': 'barnesleg2p1d',
        'bkg': 'leg2p1d',
        'offset': 'const',
    },
    'args': {
        'ilm': {'npts': (50,30,20,12,12,12,12), 'zorder': 7},
        'bkg': {'order': (15, 3, 5), 'category': 'bkg'},
        'offset': {'name': 'offset', 'value': 0}
    }
}

conf_simple = {
    'model': 'confocal-dyedfluid',
    'comps': {
        'psf': 'gauss3d',
        'ilm': 'barnesleg2p1d',
        'bkg': 'const',
        'offset': 'const',
    },
    'args': {
        'ilm': {'npts': (20,10,5), 'zorder': 5},
        'bkg': {'name': 'bkg', 'value': 0},
        'offset': {'name': 'offset', 'value': 0},
    }
}

#=============================================================================
# Initialization methods to go full circle
#=============================================================================
def create_state(image, pos, rad, slab=None, sigma=0.05, conf=conf_simple):
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
    for k,v in iteritems(conf.get('comps', {})):
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
def create_many_particle_state(imsize=None, N=None, phi=None, radius=5.0,
        polydispersity=0.0, seed=None, **kwargs):
    """
    Creates a random packing of spheres and generates the state. In order to
    specify the state, either (phi, imsize) or (N, phi) or (N, imsize) must be
    given. Any more than that and it would be over-specified.

    Parameters:
    -----------
    imsize : tuple, array_like, or integer
        the unpadded image size to fill with particles

    N : integer
        number of particles

    phi : float
        packing fraction

    radius : float
        radius of particles to add

    N : integer
        Number of particles

    seed : integer
        set the seed if desired

    *args, **kwargs : see create_state
    """
    _seed_or_not(seed)

    if imsize is not None:
        imsize = _toarr(imsize)
        tile = util.Tile(imsize)
    else:
        tile = None

    pos, rad, tile = nbody.create_configuration(
        N, tile, radius=radius, phi=phi, polydispersity=polydispersity
    )
    s = create_state(util.NullImage(shape=tile.shape), pos, rad, **kwargs)

    if isinstance(s.get('ilm'), ilms.BarnesStreakLegPoly2P1D):
        ilm = s.get('ilm')
        ilm.randomize_parameters()
        s.reset()
        s.model_to_data(s.sigma)
    return s

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

    pos = np.array([imsize/2.0 - d, imsize/2.0 + d]).reshape(-1,3)
    rad = np.array([radius, radius])

    return create_state(util.NullImage(shape=imsize), pos, rad, **kwargs)

