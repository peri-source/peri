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

