import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd
import scipy.interpolate as intr

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

def pxint(radius=8, factor=8, dx=np.array([0,0,0])):
    # the factor of coarse-graining, goal particle size, and larger size
    f = factor

    goalsize = radius
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

    blocks = state.explode(state.block_all())
    for block in blocks:
        tc = np.sqrt(1.0/np.abs(state.fisher_information(blocks=[block])))
        crb.append(tc)

    return np.squeeze(np.array(crb))

def sample(state, im, noise, N=20, sweeps=20, burn=10):
    values, errors = [], []

    for i in xrange(N):
        print i, 
        set_image(state, im, noise)
        h,l = runner.do_samples(state, sweeps, burn, quiet=True)

        h = np.array(h)
        values.append(h.mean(axis=0))
        errors.append(h.std(axis=0))

    print ''
    return np.array(values), np.array(errors)

def dorun(SNR=20, sweeps=20, burn=8, noise_samples=10):
    """
    we want to display the errors introduced by pixelation so we plot:
        * zero noise, cg image, fit
        * SNR 20, cg image, fit
        * CRB for both
    """
    radii = np.linspace(2,10,8, endpoint=False)
    crbs, vals, errs = [], [], []

    for radius in radii:
        print 'radius', radius
        s,im = pxint(radius=radius, factor=4)
        goodstate = s.state.copy()

        set_image(s, im, 1.0/SNR)
        tcrb = crb(s)
        tval, terr = sample(s, im, 1.0/SNR, N=noise_samples, sweeps=sweeps, burn=burn)
        crbs.append(tcrb)
        vals.append(tval)
        errs.append(terr)

    return np.array(crbs), np.array(vals), np.array(errs), radii

def doplot(prefix='/media/scratch/peri/pixel-integration', snrs=[20,200,2000]):
    fig = pl.figure()

    def interp(t, c):
        x = np.linspace(t[0], t[-1], 1000)
        f = intr.interp1d(t, c, kind='quadratic')
        return x, f(x)

    for i,(c,snr) in enumerate(zip(COLORS, snrs)):
        fn = prefix+'-snr'+str(snr)+'.pkl'

        crb, val, err = pickle.load(open(fn))
        radii = np.linspace(2,10,len(crb))

        d = lambda x: x.mean(axis=1)[:,1]

        pl.plot(*interp(radii, crb[:,1]), ls='-', c=c, lw=2,
                label=r"$\rm{SNR} = %i$ CRB" % snr)
        pl.plot(radii, d(err), 'o', c=c, ms=12, 
                label=r"$\rm{SNR} = %i$ Error" % snr)

        pl.semilogy()
 
    pl.xlim(2, 10)
    pl.ylim(1e-5, 1e0)
    pl.xlabel(r"Radius (px)")
    pl.ylabel(r"CRB, $\bar{\sigma}$ (px)")

    pl.legend(loc='best', prop={'size': 18}, numpoints=1)
    pl.grid(False, which='minor', axis='both')
    pl.title("Pixel integration")
