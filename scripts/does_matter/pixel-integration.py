import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd
import scipy.interpolate as intr

import common
from cbamf import const, runner
from cbamf.test import init

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

def dorun(SNR=20, sweeps=20, burn=8, noise_samples=10):
    """
    we want to display the errors introduced by pixelation so we plot:
        * zero noise, cg image, fit
        * SNR 20, cg image, fit
        * CRB for both

    a = dorun(noise_samples=30, sweeps=24, burn=12, SNR=20)
    """
    radii = np.linspace(2,10,8, endpoint=False)
    crbs, vals, errs = [], [], []

    for radius in radii:
        print 'radius', radius
        s,im = pxint(radius=radius, factor=4)
        goodstate = s.state.copy()

        common.set_image(s, im, 1.0/SNR)
        tcrb = crb(s)
        tval, terr = sample(s, im, 1.0/SNR, N=noise_samples, sweeps=sweeps, burn=burn)
        crbs.append(tcrb)
        vals.append(tval)
        errs.append(terr)

    return np.array(crbs), np.array(vals), np.array(errs), radii

def doplot(prefix='/media/scratch/peri/does_matter/pixint', snrs=[20,200,2000]):
    s,im = pxint(radius=8, factor=8, dx=np.array([0,0,0]))
    nn = np.s_[:,:,im.shape[2]/2]
    diff = (im - s.get_model_image()[s.inner])
    image0, image1 = im[nn], diff[nn]

    def interp(t, c):
        x = np.linspace(t[0], t[-1], 1000)
        f = intr.interp1d(t, c, kind='quadratic')
        return x, f(x)

    for i,(c,snr) in enumerate(zip(COLORS, snrs)):
        fn = prefix+'-snr'+str(snr)+'.pkl'
        crb, val, err, radii = pickle.load(open(fn))

        d = lambda x: x.mean(axis=1)[:,0]

        if i == 0:
            label0 = r"$\rm{SNR} = %i$ CRB" % snr
            label1 = r"$\rm{SNR} = %i$ Error" % snr
        else:
            label0 = r"$%i$, CRB" % snr
            label1 = r"$%i$, Error" % snr

        ax.plot(*interp(radii, crb[:,1]), ls='-', c=c, lw=3, label=label0)
        ax.plot(radii, d(err), 'o', ls='--', lw=0, c=c, ms=12, label=label1)

        #if i == 1:
        #    x,y = interp(radii, crb[:,1])
        #    pl.fill_between(x, y/2-y/2/7, y/2+y/2/7, color='k', alpha=0.2)

    ax.semilogy()
 
    ax.set_xlim(radii[0], radii[-1])
    ax.set_ylim(1e-5, 1e0)
    ax.set_xlabel(r"Particle radius (px)")
    ax.set_ylabel(r"Position CRB, Error (px)")

    ax.legend(loc='best', numpoints=1, ncol=3, prop={'size': 16})
    ax.grid(False, which='both', axis='both')
    ax.set_title("Pixel integration")
