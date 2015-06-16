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

GS = 0.02
RADIUS = 12.0
PSF = (1, 2)
ORDER = (3,3,2)
PAD = 22

sweeps = 20
samples = 10
burn = sweeps - samples

#itrue = initializers.normalize(initializers.load_tiff("/media/scratch/bamf/brian-frozen.tif", do3d=True)[12:,:256,:256], True)
itrue = initializers.normalize(initializers.load_tiff("/media/scratch/bamf/neil-large-clean.tif", do3d=True)[12:,:128,:128], False)
xstart, proc = initializers.local_max_featuring(itrue, 9)
itrue = initializers.normalize(itrue, True)

rstart = 7*np.ones(xstart.shape[0])
pstart = np.array(PSF)
bstart = np.zeros(np.prod(ORDER))
astart = np.zeros(1)
zstart = np.ones(1)#*1.41
GN = rstart.shape[0]
bstart[0] = 1

itrue = np.pad(itrue, PAD+2, mode='constant', constant_values=-10)
xstart += PAD+2

strue = np.hstack([xstart.flatten(), rstart, pstart, bstart, astart, zstart])
s = states.ConfocalImagePython(GN, itrue, pad=PAD, order=ORDER, state=strue,
        sigma=GS, psftype=states.PSF_ANISOTROPIC_GAUSSIAN, threads=4)

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
