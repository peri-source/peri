import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import states, runner, initializers
from cbamf.comp import objs, psfs, ilms
from cbamf.viz import plots

ORDER = (3,3,2)
sweeps = 30
samples = 20
burn = sweeps - samples

FILE = 2

if FILE == 1:
    sigma = 0.04
    PSF = (2.5, 5.0)
    OFF = 0.2
    BKG = 0.6
    PAD, FSIZE, RAD, INVERT, IMSIZE, zstart, zscale = 24, 8, 14, True, 256, 17, 1.34
    raw = initializers.load_tiff("/media/scratch/bamf/brian-frozen.tif")
if FILE == 2:
    sigma = 0.0141
    PSF = (2.402, 5.151)
    ILM = [ 0.94367202, -0.03018738,  0.02714354, -0.14109688,  0.09263272,
            0.02980459, -0.07640744, -0.07616682, -0.31820937,  0.13269572,
            0.1601496 ,  0.07075362,  0.16516718,  0.02004564, -0.00973342,
            -0.11106133,  0.00270983, -0.27165701]
    PAD, FSIZE, RAD, INVERT, IMSIZE, zstart, zscale = 24, 5, 5.04, False, 128, 5, 1.0717
    raw = initializers.load_tiff("/media/scratch/bamf/neil-large.tif")
if FILE == 3:
    sigma = 0.01
    PSF = (2.5, 5.0)
    PAD, FSIZE, RAD, INVERT, IMSIZE, zstart, zscale = 22, 6, 7.3, False, 128, 12, 1.06
    raw = initializers.load_tiff("/media/scratch/bamf/neil-large-clean.tif")
if FILE == 4:
    sigma = 0.05
    PSF = (2.2, 4.6)
    PAD, FSIZE, RAD, INVERT, IMSIZE, zstart, zscale = 16, 9, 7.3, True, 128, 2, 1.056
    raw = next(initializers.load_tiff_iter("/media/scratch/bamf/p1_N150_1.tif", 70))

feat = initializers.normalize(raw[zstart:,:IMSIZE,:IMSIZE], INVERT)
xstart, proc = initializers.local_max_featuring(feat, FSIZE, FSIZE/3.)
itrue = initializers.normalize(feat, True)
itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
xstart += PAD
rstart = RAD*np.ones(xstart.shape[0])
initializers.remove_overlaps(xstart, rstart)

imsize = itrue.shape
obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian(PSF, shape=imsize)
ilm = ilms.Polynomial3D(order=ORDER, shape=imsize)
s = states.ConfocalImagePython(itrue, obj=obj, psf=psf, ilm=ilm,
        zscale=zscale, pad=16, sigma=sigma)

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
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        runner.sample_particles(s, stepout=0.1)
        runner.sample_block(s, 'ilm', stepout=0.1, explode=False)
        runner.sample_block(s, 'off', stepout=0.1, explode=True)
        runner.sample_block(s, 'psf', stepout=0.1, explode=False)
        runner.sample_block(s, 'zscale', explode=True)

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
    return h
