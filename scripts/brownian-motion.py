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

def diffusion(diffusion_constant, exposure_time, samples=100):
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

    for i in xrange(samples):
        offset = np.sqrt(6*diffusion_constant*exposure_time)*np.random.randn(3)
        s0.obj.pos[0] = np.array(s0.image.shape)/2 + offset
        s0.reset()

        finalimage += s0.get_model_image()[sl]

    finalimage /= float(samples)

    # place that into a new image at the expected parameters
    s = init.create_single_particle_state(imsize=4*radius, sigma=0.05,
            radius=radius, psfargs={'params': psfsize, 'error': 1e-6})
    s.reset()

    # measure the true inferred parameters
    return s, finalimage

def crb(state):
    crb = []

    blocks = state.explode(state.block_all())
    for block in blocks:
        tc = np.sqrt(1.0/np.abs(state.fisher_information(blocks=[block])))
        crb.append(tc)

    return np.squeeze(np.array(crb))

def sample(state, im, noise, N=10):
    values, errors = [], []

    for i in xrange(N):
        print '{:=^79}'.format(' %i ' % i)
        set_image(state, im, noise)
        h,l = runner.do_samples(state, 20, 10)

        h = np.array(h)
        values.append(h.mean(axis=0))
        errors.append(h.std(axis=0))

    return np.array(values), np.array(errors)

def dorun():
    """
    we want to display the errors introduced by pixelation so we plot:
        * CRB, sampled error vs exposure time
    """
    times = np.logspace(-6, 1, 20)
    crbs0, vals0, errs0 = [], [], []
    crbs1, vals1, errs1 = [], [], []

    for i,t in enumerate(times):
        print '###### time', i, t
        s,im = diffusion(diffusion_constant=1, exposure_time=t)
        goodstate = s.state.copy()

        # essentially noise-less
        set_image(s, im, 1e-6)
        crbs0.append(crb(s))

        val, err = sample(s, im, 1e-6)
        vals0.append(val)
        errs0.append(err)

        # typical image
        set_image(s, im, 5e-2)
        crbs1.append(crb(s))

        val, err = sample(s, im, 5e-2)
        vals1.append(val)
        errs1.append(err)

    return [np.array(crbs0), np.array(vals0), np.array(errs0),
            np.array(crbs1), np.array(vals1), np.array(errs1)]

def doplot(filename='/media/scratch/peri/pixel-integration.pkl'):
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
