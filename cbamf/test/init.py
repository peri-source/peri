import numpy as np
import scipy as sp

from cbamf.runner import create_state
from cbamf.test import poissondisks
from cbamf import states
from cbamf import const

def _seed_or_not(seed=None):
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

def _toarr(i):
    if not hasattr(i, '__iter__'):
        i = np.array((i,)*3)
    return np.array(i)

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

from cbamf.comp.ilms import BarnesStreakLegPoly2P1DX3
def randomize_barnes_ilm(ilm, inner=np.s_[:], ptp=0.2, fourier=False):
    if not isinstance(ilm, BarnesStreakLegPoly2P1DX3):
        raise AttributeError("needs to be BarnesStreakLegPoly2P1DX3")

    def rng(tmp, ptp, fourier=fourier):
        for i, o in enumerate(ilm._indices_xy[1:]):
            tmp.params[i] = ptp*(np.random.rand() - 0.5) / (np.prod(o)+1) / 2

        off = len(ilm._indices_xy)
        for i, o in enumerate(ilm._indices_z[1:]):
            tmp.params[i+off+1] = ptp*(np.random.rand() - 0.5) / (np.prod(o)+1) / 2

        for i, s in enumerate(ilm.slicers):
            N = tmp.params[s].shape[0]
            if fourier:
                t = ((np.random.rand(N)-0.5) + 1.j*(np.random.rand(N)-0.5))/(np.arange(N)+1)
                q = np.real(np.fft.ifftn(t))
            else:
                t = ptp*np.sqrt(N)*(np.random.rand(N)-0.5)
                q = np.cumsum(t)

            q = ptp * q / q.ptp() / len(ilm.slicers)
            q -= q.mean()
            tmp.params[s] = 1.0*(i==0) + q

        tmp.initialize()

    tmp = BarnesStreakLegPoly2P1DX3(shape=ilm.shape, order=ilm.order, npts=ilm.npts)
    rng(tmp, ptp)
    return tmp
