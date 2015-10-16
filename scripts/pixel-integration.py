import numpy as np
import scipy as sp
import scipy.ndimage as nd

import matplotlib.pyplot as pl

from cbamf import const, runner
from cbamf.test import init
from cbamf.states import prepare_image

def set_image(state, cg, sigma):
    image = cg + np.random.randn(*cg.shape)*sigma
    image = np.pad(image, const.PAD, mode='constant', constant_values=const.PADVAL)
    state.set_image(image)
    state.sigma = sigma
    state.reset()

def pxint(factor=8, dx=np.array([0,0,0])):
    # the factor of coarse-graining, goal particle size, and larger size
    f = factor

    goalsize = 8
    goalpsf = np.array([2.0, 1.0, 3.0])

    bigsize = goalsize * f
    bigpsf = goalpsf * np.array([f,f,1])

    s0 = init.create_single_particle_state(
            imsize=np.array((4*goalsize, 4*bigsize, 4*bigsize)),
            radius=bigsize, psfargs={'params': bigpsf, 'error': 1e-6},
            stateargs={'zscale': 1.0*f})
    s0.obj.pos += np.array([0,1,1]) * (f-1.0)/2.0
    s0.obj.pos += np.array([1,f,f]) * dx
    s0.reset()

    # coarse-grained image
    sl = np.s_[s0.pad:-s0.pad,s0.pad:-s0.pad,s0.pad:-s0.pad]
    m = s0.get_model_image()[sl]

    # indices for coarse-graining
    e = m.shape[1]
    i = np.linspace(0, e/f, e, endpoint=False).astype('int')
    j = np.linspace(0, e/f, e/f, endpoint=False).astype('int')
    z,y,x = np.meshgrid(*(j,i,i), indexing='ij')
    ind = x + e*y + e*e*z

    # finally, c-g'ed image
    cg = nd.mean(m, labels=ind, index=np.unique(ind)).reshape(e/f, e/f, e/f)

    # place that into a new image at the expected parameters
    s = init.create_single_particle_state(imsize=4*goalsize, sigma=0.05,
            radius=goalsize, psfargs={'params': goalpsf, 'error': 1e-6})
    s.obj.pos += dx
    s.reset()

    # measure the true inferred parameters
    return s, cg

def doplot():
    """
    we want to display the errors introduced by pixelation so we plot:
        * zero noise, cg image, fit
        * SNR 20, cg image, fit
        * CRB for both
    """
    s,im = pxint()

    labels = [
        'pos-z', 'pos-y', 'pos-x', 'rad',
        'psf-x', 'psf-y', 'psf-z', 'ilm',
        'off', 'rscale', 'zscale', 'sigma'
    ]

    set_image(s, im, 1e-6)
    set_image(s, im, 5e-2)

    fig = pl.figure()
