"""
Plot the average positional / radius error vs fraction of self-diffusion time
"""
import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd
from IPython.core.debugger import Tracer
#Tracer()() / %debug after stacktrace

import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import ImageGrid

from cbamf import const, runner, initializers
from cbamf.test import init
from cbamf.states import prepare_image
from cbamf.viz.util import COLORS
from cbamf.viz.plots import lbl

def set_image(state, cg, sigma):
    image = cg + np.random.randn(*cg.shape)*sigma
    image = np.pad(image, const.PAD, mode='constant', constant_values=const.PADVAL)
    state.set_image(image)
    state.sigma = sigma
    state.reset()

def zjitter(jitter=0.0, radius=5):
    """
    scan jitter is in terms of the fractional pixel difference when
    moving the laser in the z-direction
    """
    psfsize = np.array([2.0, 1.0, 3.0])

    # create a base image of one particle
    s0 = init.create_single_particle_state(imsize=4*radius, 
            radius=radius, psfargs={'params': psfsize, 'error': 1e-6})
    sl = np.s_[s0.pad:-s0.pad,s0.pad:-s0.pad,s0.pad:-s0.pad]

    # add up a bunch of trajectories
    finalimage = 0*s0.get_model_image()[sl]
    position = 0*s0.obj.pos[0]

    for i in xrange(finalimage.shape[0]):
        offset = jitter*np.random.randn(3)*np.array([1,0,0])
        s0.obj.pos[0] = np.array(s0.image.shape)/2 + offset
        s0.reset()

        finalimage[i] = s0.get_model_image()[sl][i]
        position += s0.obj.pos[0]

    position /= float(finalimage.shape[0])

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

def dorun(SNR=20, njitters=20, samples=10, noise_samples=10, sweeps=20, burn=10):
    """
    we want to display the errors introduced by pixelation so we plot:
        * CRB, sampled error vs exposure time

    a = dorun(ntimes=10, samples=5, noise_samples=5, sweeps=20, burn=8)
    """
    jitters = np.logspace(-6, np.log10(0.5), njitters)
    crbs, vals, errs, poss = [], [], [], []

    for i,t in enumerate(jitters):
        print '###### jitter', i, t

        for j in xrange(samples):
            print 'image', j, '|', 
            s,im,pos = zjitter(jitter=t)

            # typical image
            set_image(s, im, 1.0/SNR)
            crbs.append(crb(s))

            val, err = sample(s, im, 1.0/SNR, N=noise_samples, sweeps=sweeps, burn=burn)
            poss.append(pos)
            vals.append(val)
            errs.append(err)


    shape0 = (njitters, samples, -1)
    shape1 = (njitters, samples, noise_samples, -1)

    crbs = np.array(crbs).reshape(shape0)
    vals = np.array(vals).reshape(shape1)
    errs = np.array(errs).reshape(shape1)
    poss = np.array(poss).reshape(shape0)

    return  [crbs, vals, errs, poss, jitters]

def dist(a):
    return np.sqrt((a[...,:3]**2).sum(axis=-1)).mean(axis=-1)

def errs(val, pos):
    v,p = val, pos
    return np.sqrt(((v[...,:3] - p[:,:,None,:])**2).sum(axis=-1)).mean(axis=(1,2))

def doplot(prefix='/media/scratch/peri/does_matter/z-jitter', snrs=[20,50,200,500]):
    fig = pl.figure(figsize=(10,10))

    ax = fig.add_axes([0.15, 0.10, 0.80, 0.48])
    gs = ImageGrid(fig, rect=[0.05, 0.63, 0.90, 0.35], nrows_ncols=(1,3), axes_pad=0.05)

    s,im,pos = zjitter(jitter=0.3, radius=5)
    nn = np.s_[:,:,im.shape[2]/2]
    args = {'cmap': 'bone_r', 'vmin': 0, 'vmax': 1}
    figlbl = ['A', 'B', 'C']
    labels = ['Reference', 'Model', 'Difference']
    gs[0].imshow(im[nn], **args)
    gs[1].imshow(s.get_model_image()[s.inner][nn], **args)
    gs[2].imshow((im-s.get_model_image()[s.inner])[nn], **args)

    for i in xrange(3):
        gs[i].set_xticks([])
        gs[i].set_yticks([])
        gs[i].set_xlabel(labels[i])
        lbl(gs[i], figlbl[i])

    symbols = ['o', '^', 'D', '>']
    for i, snr in enumerate(snrs):
        c = COLORS[i]
        fn = prefix+'-snr-'+str(snr)+'.pkl'
        crb, val, err, pos, time = pickle.load(open(fn))

        if i == 0:
            label0 = r"$\rm{SNR} = %i$ CRB" % snr
            label1 = r"$\rm{SNR} = %i$ Error" % snr
        else:
            label0 = r"$%i$, CRB" % snr
            label1 = r"$%i$, Error" % snr

        ax.plot(time, dist(crb), '-', c=c, lw=3, label=label0)
        ax.plot(time, errs(val, pos), symbols[i], ls='--', lw=2, c=c, label=label1, ms=12)

    lbl(ax, 'D')
    ax.loglog()
    ax.set_ylim(1e-4, 1e0)
    ax.set_xlim(0, time[-1])
    ax.legend(loc='best', ncol=2, prop={'size': 18}, numpoints=1)
    ax.set_xlabel(r"$z$-scan NSR")
    ax.set_ylabel(r"Position CRB, Error")
    ax.grid(False, which='both', axis='both')
    gs[1].set_title(r"$z$-scan jitter")
