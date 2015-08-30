import numpy as np
import scipy as sp

from cbamf.priors import overlap

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
