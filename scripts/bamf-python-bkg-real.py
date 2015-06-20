import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import states, run, initializers
from cbamf.comp import objs, psfs, ilms

ORDER = (3,3,2)
sweeps = 20
samples = 10
burn = sweeps - samples

if False:
    sigma = 0.05
    PSF = (0.9, 2.0)
    PAD, FSIZE, RAD, INVERT, IMSIZE, zscale = 34, 16, 17, True, 256, 1.34
    raw = initializers.load_tiff("/media/scratch/bamf/brian-frozen.tif", do3d=True)
else:
    sigma = 0.02
    PSF = (1.4, 3.0)
    #PAD, FSIZE, RAD, INVERT, IMSIZE, zscale = 22, 9, 7.3, False, 128, 1.06
    #raw = initializers.load_tiff("/media/scratch/bamf/neil-large-clean.tif", do3d=True)

    PAD, FSIZE, RAD, INVERT, IMSIZE, zscale = 16, 5, 5.3, False, 128, 1.06
    raw = initializers.load_tiff("/media/scratch/bamf/neil-large.tif", do3d=True)

itrue = initializers.normalize(raw[12:,:IMSIZE,:IMSIZE], INVERT)
xstart, proc = initializers.local_max_featuring(itrue, FSIZE)
itrue = initializers.normalize(itrue, True)
itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
xstart += PAD
rstart = RAD*np.ones(xstart.shape[0])

imsize = itrue.shape
obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian(PSF, shape=imsize)
ilm = ilms.Polynomial3D(order=ORDER, shape=imsize)
s = states.ConfocalImagePython(itrue, obj=obj, psf=psf, ilm=ilm,
        zscale=zscale, offset=0, pad=16, sigma=sigma)

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
        run.sample_block(s, 'ilm', stepout=0.1, explode=False)
        run.sample_block(s, 'off', stepout=0.1, explode=True)
        run.sample_block(s, 'zscale', explode=True)

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
    return h
