import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf.cu import nbl, fields
from cbamf import observers, samplers, models, engines, initializers, states, run

ORDER = (1,1,1)
sweeps = 20
samples = 10
burn = sweeps - samples

if False:
    GS = 0.02
    PSF = (1.4, 3.0)
    PAD, FSIZE, RAD, INVERT, IMSIZE = 34, 13, 13, True, 256
    raw = initializers.load_tiff("/media/scratch/bamf/brian-frozen.tif", do3d=True)
else:
    GS = 0.02
    PSF = (1.4, 3.0)
    PAD, FSIZE, RAD, INVERT, IMSIZE = 22, 9, 7.3, False, 128
    raw = initializers.load_tiff("/media/scratch/bamf/neil-large-clean.tif", do3d=True)

itrue = initializers.normalize(raw[12:,:IMSIZE,:IMSIZE], INVERT)
xstart, proc = initializers.local_max_featuring(itrue, FSIZE)
itrue = initializers.normalize(itrue, True)

rstart = RAD*np.ones(xstart.shape[0])
pstart = np.array(PSF)
bstart = np.zeros(np.prod(ORDER))
astart = np.ones(1)*0
zstart = np.ones(1)#*1.064
GN = rstart.shape[0]
bstart[0] = 1.0

itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
xstart += PAD

strue = np.hstack([xstart.flatten(), rstart, pstart, bstart, astart, zstart])
s = states.ConfocalImagePython(GN, itrue, pad=PAD, order=ORDER, state=strue.copy(),
        sigma=GS, psftype=states.PSF_ANISOTROPIC_GAUSSIAN, threads=4)

run.renorm(s)

def gd(state, N=1, ratio=1e-1):
    state.set_current_particle()
    for i in xrange(N):
        print state.loglikelihood()
        grad = state.gradloglikelihood()
        n = state.state + 1.0/np.abs(grad).max() * ratio * grad
        state.set_state(n)
        print state.loglikelihood()

def sample(s):
    h = []
    run.renorm(s)
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        run.sample_particles(s, stepout=0.1)
        run.sample_block(s, 'psf', stepout=0.1, explode=False)
        run.sample_block(s, 'bkg', stepout=0.1, explode=False)
        run.sample_block(s, 'amp', stepout=0.1, explode=True)
        #run.sample_block(s, 'zscale', explode=True)

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
    return h
