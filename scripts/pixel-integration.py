import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd

import matplotlib.pyplot as pl

from cbamf import const, runner
from cbamf.test import init
from cbamf.states import prepare_image
from cbamf.viz.util import COLORS

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

def crb(state):
    crb = []

    blocks = state.explode(s.block_all())
    for block in blocks:
        tc = np.sqrt(1.0/np.abs(state.fisher_information(blocks=[block])))
        crb.append(tc)

    return np.squeeze(np.array(crb))

def sample(state, im, noise, N=20):
    values, errors = [], []

    for i in xrange(N):
        print '{:=^79}'.format(' %i ' % i)
        set_image(state, im, noise)
        h,l = runner.do_samples(state, 30, 15)

        h = np.array(h)
        values.append(h.mean(axis=0))
        errors.append(h.std(axis=0))

    return np.array(values), np.array(errors)

def dorun():
    """
    we want to display the errors introduced by pixelation so we plot:
        * zero noise, cg image, fit
        * SNR 20, cg image, fit
        * CRB for both
    """
    s,im = pxint(factor=4)
    goodstate = s.state.copy()

    set_image(s, im, 1e-6)
    crb0 = crb(s)
    val0, err0 = sample(s, im, 1e-6)

    set_image(s, im, 5e-2)
    crb1 = crb(s)
    val1, err1 = sample(s, im, 5e-2)

def doplot(filename='/media/scratch/peri/pixel-integration.pkl'):
    labels = [
        'pos-z', 'pos-y', 'pos-x', 'rad',
        'psf-x', 'psf-y', 'psf-z', 'ilm',
        'off', 'rscale', 'zscale', 'sigma'
    ]

    crb0,val0,err0,crb1,val1,err1,goodstate = pickle.load(open(filename))

    fig = pl.figure()

    pl.plot(crb0, '-', c=COLORS[0], lw=2, label=r"$\rm{SNR} = 10^6$ CRB")
    pl.plot(err0.mean(axis=0), 'o', c=COLORS[0], label=r"$\rm{SNR} = 10^6$ Error", ms=12)
    pl.plot(crb1, '-', c=COLORS[1], lw=2, label=r"$\rm{SNR} = 20$ CRB")
    pl.plot(err1.mean(axis=0), 'o', c=COLORS[1], label=r"$\rm{SNR} = 20$ Error", ms=12)
    pl.semilogy()
    pl.ylim(1e-9, 1e0)
    pl.xlim(0, len(labels)-3)
    pl.xticks(xrange(len(labels)-1), labels[:-1], rotation='vertical')
    pl.legend(bbox_to_anchor=(1.07,1.0), prop={'size': 18}, numpoints=1)
    pl.ylabel(r"CRB, $\bar{\sigma}$")
    pl.grid(False, which='minor', axis='both')
    pl.title("Pixel integration")
