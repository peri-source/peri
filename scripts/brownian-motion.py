"""
Plot the average positional / radius error vs fraction of self-diffusion time
"""
import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd

import matplotlib.pyplot as pl

from cbamf import const, runner, initializers
from cbamf.test import init
from cbamf.states import prepare_image
from cbamf.viz.util import COLORS

def set_image(state, cg, sigma):
    image = cg + np.random.randn(*cg.shape)*sigma
    image = np.pad(image, const.PAD, mode='constant', constant_values=const.PADVAL)
    state.set_image(image)
    state.sigma = sigma
    state.reset()

def diffusion(diffusion_constant, exposure_time, samples=200):
    """
    diffusion_constant is in terms of seconds and pixel sizes
    exposure_time is in seconds

    for 80% (60mPas), D = kT/(\pi\eta r) ~ 1 px^2/sec
    a full 60 layer scan takes 0.1 sec, so a particle is 0.016 sec exposure
    """
    radius = 5
    psfsize = np.array([2.0, 1.0, 3.0])
    psfsize = np.array([2.0, 1.0, 3.0])/10.

    # create a base image of one particle
    s0 = init.create_single_particle_state(imsize=4*radius, 
            radius=radius, psfargs={'params': psfsize, 'error': 1e-6})
    sl = np.s_[s0.pad:-s0.pad,s0.pad:-s0.pad,s0.pad:-s0.pad]

    # add up a bunch of trajectories
    finalimage = 0*s0.get_model_image()[sl]
    position = 0*s0.obj.pos[0]

    for i in xrange(samples):
        offset = np.sqrt(6*diffusion_constant*exposure_time)*np.random.randn(3)
        s0.obj.pos[0] = np.array(s0.image.shape)/2 + offset
        s0.reset()

        finalimage += s0.get_model_image()[sl]
        position += s0.obj.pos[0]

    finalimage /= float(samples)
    position /= float(samples)

    # place that into a new image at the expected parameters
    s = init.create_single_particle_state(imsize=4*radius, sigma=0.05,
            radius=radius, psfargs={'params': psfsize, 'error': 1e-6})
    s.reset()

    # measure the true inferred parameters
    return s, finalimage, position

def crb(state):
    crb = []

    blocks = state.explode(state.block_all())
    for block in blocks:
        tc = np.sqrt(1.0/np.abs(state.fisher_information(blocks=[block])))
        crb.append(tc)

    return np.squeeze(np.array(crb))

def sample(state, im, noise, N=10, burn=10, sweeps=20):
    values, errors = [], []

    for i in xrange(N):
        print i, ' ',
        set_image(state, im, noise)
        h,l = runner.do_samples(state, sweeps, burn, quiet=True)

        h = np.array(h)
        values.append(h.mean(axis=0))
        errors.append(h.std(axis=0))

    print ''
    return np.array(values), np.array(errors)

def dorun(SNR=20, ntimes=20, samples=10, noise_samples=10, sweeps=20, burn=10):
    """
    we want to display the errors introduced by pixelation so we plot:
        * CRB, sampled error vs exposure time
    """
    times = np.logspace(-4, 0, ntimes)
    crbs, vals, errs, poss = [], [], [], []

    for i,t in enumerate(times):
        print '###### time', i, t

        for j in xrange(samples):
            print 'image', j, '|', 
            s,im,pos = diffusion(diffusion_constant=1, exposure_time=t)

            # typical image
            set_image(s, im, 1.0/SNR)
            crbs.append(crb(s))

            val, err = sample(s, im, 1.0/SNR, N=noise_samples, sweeps=sweeps, burn=burn)
            poss.append(pos)
            vals.append(val)
            errs.append(err)


    shape0 = (ntimes, samples, -1)
    shape1 = (ntimes, samples, noise_samples, -1)

    crbs = np.array(crbs).reshape(shape0)
    vals = np.array(vals).reshape(shape1)
    errs = np.array(errs).reshape(shape1)
    poss = np.array(poss).reshape(shape0)

    return  [crbs, vals, errs, poss, times]

def dist(a):
    return np.sqrt((a[...,:3]**2).sum(axis=-1)).mean(axis=-1)

def errs(val, pos):
    v,p = val, pos
    return np.sqrt(((v[...,:3] - p[:,:,None,:])**2).sum(axis=-1)).mean(axis=(1,2))

def doplot(prefix='/media/scratch/peri/brownian-motion', snrs=[20,200]):
    fig = pl.figure()

    for i, snr in enumerate(snrs):
        c = COLORS[i]
        fn = prefix+'-snr'+str(snr)+'.pkl'
        crb, val, err, pos, time = pickle.load(open(fn))

        pl.plot(time, dist(crb), '-', c=c, lw=2, label=r"$\rm{SNR} = %i$ CRB" % snr)
        pl.plot(time, errs(val, pos), 'o', c=c, label=r"$\rm{SNR} = %i$ Error" % snr, ms=12)

    pl.loglog()
    pl.ylim(1e-4, 1e0)
    pl.xlim(0, time[-1])
    pl.legend(loc='best',  prop={'size': 18}, numpoints=1)
    pl.xlabel(r"Exposure time (sec)")
    pl.ylabel(r"CRB, $\bar{\sigma}$")
    pl.grid(False, which='minor', axis='both')
    pl.title("Brownian motion")
