import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import acor
import pylab as pl
from cbamf.cu import nbl, fields
from cbamf import observers, samplers, models, engines, initializers, states, run
from time import sleep
import itertools
import sys
import pickle
import time

GN = 64
GS = 0.05
PSF = (0.6, 2)
ORDER = (3,3,2)

sweeps = 10
samples = 10
burn = sweeps - samples

import pickle
xstart, rstart = pickle.load(open("/media/scratch/bamf/bamf_ic_16_xr.pkl", 'r'))
bstart = np.zeros(ORDER); bstart[0,0,0] = 1
strue = np.hstack([xstart.flatten(), rstart, np.array(PSF), bstart.ravel(), np.ones(1), np.ones(1)])
s0 = states.ConfocalImagePython(len(rstart), np.zeros((128,128,128)), pad=16,
        order=ORDER, state=strue, threads=1)
s0.set_current_particle()
ipure = s0.create_final_image()
itrue = ipure + np.random.normal(0.0, GS, size=ipure.shape)

s = states.ConfocalImagePython(len(rstart), itrue, pad=16, order=ORDER,
        state=strue, threads=1)

run.renorm(s)

#raise IOError
if True:
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        run.sample_particles(s)
        run.sample_block(s, 'psf', explode=False)
        run.sample_block(s, 'bkg', explode=False)
        run.sample_block(s, 'amp', explode=True)
        #run.sample_block(s, 'zscale', explode=True)

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)

mu = h.mean(axis=0)
std = h.std(axis=0)
pl.figure(figsize=(20,4))
pl.errorbar(xrange(len(mu)), (mu-strue), yerr=5*std/np.sqrt(samples),
        fmt='.', lw=0.15, alpha=0.5)
pl.vlines([0,3*GN-0.5, 4*GN-0.5], -1, 1, linestyle='dashed', lw=4, alpha=0.5)
pl.hlines(0, 0, len(mu), linestyle='dashed', lw=5, alpha=0.5)
pl.xlim(0, len(mu))
pl.ylim(-0.02, 0.02)
pl.show()
#"""
