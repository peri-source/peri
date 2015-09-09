import numpy as np
import scipy as sp

from cbamf.priors import overlap
from cbamf.util import Tile

def iter_pos_rad(state, samples):
    for sample in samples:
        pos = sample[state.b_pos].reshape(-1,3)
        rad = sample[state.b_rad]
        yield pos, rad

def gofr(pos, rad, zscale):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])

    seps = []
    for i in xrange(N-1):
        o = np.arange(i+1, N)
        d = np.sqrt( ((z*(pos[i] - pos[o]))**2).sum(axis=-1) )
        seps.extend(d)
    return np.array(seps)

def gofr_surfaces(pos, rad, zscale):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])

    seps = []
    for i in xrange(N-1):
        o = np.arange(i+1, N)
        d = np.sqrt( ((z*(pos[i] - pos[o]))**2).sum(axis=-1) )
        r = rad[i] + rad[o]

        diff = d-r
        seps.extend(diff)
    return np.array(seps)

def gofr_full(pos, rad, zscale, resolution=3e-2, rmax=10, method=gofr,
        mask_start=None):
    d = 2*rad.mean()

    o = method(pos, rad, zscale)
    y,x = np.histogram(o/d, bins=np.linspace(0, rmax, d*rmax/resolution))
    x = (x[1:] + x[:-1])/2

    if mask_start is not None:
        mask = x > mask_start
        x = x[mask]
        y = y[mask]

    return x, y/(4*np.pi*x**2*resolution)

def packing_fraction(pos, rad, bounds=None, state=None):
    if state is not None:
        bounds = Tile(left=state.pad,
                right=np.array(state.image.shape)-state.pad)

    box_volume = np.prod(bounds.r - bounds.l)
    particle_volume = 0.0
    nparticles = 0

    for p,r in zip(pos, rad):
        if (p > bounds.l-r).all() and (p < bounds.r+r).all():
            vol = 4*np.pi/3 * r**3

            d0 = p + r - bounds.l
            d1 = bounds.r - p - r

            d0 = d0[d0<0]
            d1 = d1[d1<0]

            vol += (np.pi*d0/6 * (3*r**2 + d0**2)).sum()
            vol += (np.pi*d1/6 * (3*r**2 + d1**2)).sum()

            particle_volume += vol
            nparticles += 1

    return particle_volume / box_volume, nparticles

def average_packing_fraction(state, samples):
    phi = []

    for p,r in iter_pos_rad(state, samples):
        phi.append(packing_fraction(p,r,state=state))

    phi = np.array(phi)

    return phi.mean(axis=0)[0], phi.std(axis=0)[0]/np.sqrt(len(phi))
