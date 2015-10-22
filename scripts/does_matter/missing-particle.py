"""
Plot the average positional / radius error vs fraction of self-diffusion time
"""
import sys
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

RADIUS = 5.0

def missing_particle(separation=0.0, radius=RADIUS, SNR=20):
    """ create a two particle state and compare it to featuring using a single particle guess """
    # create a base image of one particle
    s = init.create_two_particle_state(imsize=6*radius+4, axis='x', sigma=1.0/SNR,
            delta=separation, radius=radius, stateargs={'varyn': True}, psfargs={'error': 1e-6})
    s.obj.typ[1] = 0.
    s.reset()

    return s, s.obj.pos.copy()

def crb(state):
    crb = []

    blocks = state.explode(state.block_all())
    for block in blocks:
        tc = np.sqrt(1.0/np.abs(state.fisher_information(blocks=[block])))
        crb.append(tc)

    return np.squeeze(np.array(crb))

def sample(state, N=15, burn=15, sweeps=20):
    bl = state.blocks_particle(0)
    h = runner.sample_state(state, bl, stepout=0.1, N=sweeps)
    h = h.get_histogram()[burn:]

    return h.mean(axis=0), h.std(axis=0)

def dorun(SNR=20, separations=20, noise_samples=12, sweeps=30, burn=15):
    seps = np.logspace(-2, np.log10(2*RADIUS), separations)
    crbs, vals, errs, poss = [], [], [], []

    np.random.seed(10)
    for i,t in enumerate(seps):
        print 'sep', i, t, '|', 

        s,pos = missing_particle(separation=t, SNR=SNR)
        crbs.append(crb(s))
        poss.append(pos)

        for j in xrange(noise_samples):
            print j,
            sys.stdout.flush()

            s,pos = missing_particle(separation=t, SNR=SNR)
            val, err = sample(s, N=noise_samples, sweeps=sweeps, burn=burn)
            vals.append(val)
            errs.append(err)

        print ''
    shape0 = (separations,  -1)
    shape1 = (separations, noise_samples, -1)

    crbs = np.array(crbs).reshape(shape0)
    vals = np.array(vals).reshape(shape1)
    errs = np.array(errs).reshape(shape1)
    poss = np.array(poss).reshape(shape0)

    return  [crbs, vals, errs, poss, seps]

def dist(a):
    return np.sqrt((a[...,:3]**2).sum(axis=-1))

def errs(val, pos):
    v,p = val, pos
    return np.sqrt(((v[:,:,:3] - p[:,None,:3])**2).sum(axis=-1)).mean(axis=1)

def doplot(prefix='/media/scratch/peri/does_matter/missing-particle', snrs=[20,50,200]):
    fig = pl.figure(figsize=(14,7))

    ax = fig.add_axes([0.43, 0.15, 0.52, 0.75])
    gs = ImageGrid(fig, rect=[0.05, 0.05, 0.25, 0.90], nrows_ncols=(2,1), axes_pad=0.25,
            cbar_location='right', cbar_mode='each', cbar_size='10%', cbar_pad=0.04)

    s, pos = missing_particle(separation=0.0, radius=RADIUS, SNR=20)
    s.obj.typ[1] = 1.; s.reset()
    im = s.get_model_image()[s.inner]
    s.obj.typ[1] = 0.; s.reset()

    h,l = runner.do_samples(s, 30,0, quiet=True)
    nn = np.s_[im.shape[0]/2,:,:]

    figlbl, labels = ['A', 'B'], ['Reference', 'Difference']
    diff = (im - s.get_model_image()[s.inner])[nn]
    diffm = 1.0#np.abs(diff).max()
    im0 = gs[0].imshow(im[nn], vmin=0, vmax=1, cmap='bone_r')
    im1 = gs[1].imshow(diff, vmin=-diffm, vmax=diffm, cmap='RdBu')
    cb0 = pl.colorbar(im0, cax=gs[0].cax, ticks=[0,1])
    cb1 = pl.colorbar(im1, cax=gs[1].cax, ticks=[-diffm,diffm]) 
    cb0.ax.set_yticklabels(['0', '1'])
    cb1.ax.set_yticklabels(['-%0.1f' % diffm, '%0.1f' % diffm])

    for i in xrange(2):
        gs[i].set_xticks([])
        gs[i].set_yticks([])
        gs[i].set_ylabel(labels[i])
        #lbl(gs[i], figlbl[i])

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

    ax.loglog()
    ax.set_ylim(1e-3, 2e0)
    ax.set_xlim(0, time[-1])
    ax.legend(loc='best', ncol=3, numpoints=1, prop={'size': 16})
    ax.set_xlabel(r"Surface-to-surface distance")
    ax.set_ylabel(r"Position CRB, Error")
    ax.grid(False, which='minor', axis='both')
    ax.grid(False, which='major', axis='both')
    ax.set_title(r"Missing particle effects")

def doall():
    for snr in [20,50,200]:
        a = dorun(separations=20, noise_samples=10, sweeps=15, burn=5, SNR=snr)
        pickle.dump(a, open('./missing-particle-snr-%i.pkl' % snr, 'w'))
