import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import states, runner, initializers
from cbamf.comp import psfs, ilms, objs
from cbamf.viz import plots
import pickle
import time

sigma = 0.050

sweeps = 30
samples = 20
burn = sweeps - samples

imsize = (128,128,128)
blank = np.zeros(imsize, dtype='float')
xstart, rstart = pickle.load(open("/media/scratch/bamf/bamf_ic_16_xr.pkl", 'r'))
initializers.remove_overlaps(xstart, rstart)

obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian((0.6, 2), shape=imsize)
ilm = ilms.Polynomial3D(order=(1,1,1), shape=imsize)
s = states.ConfocalImagePython(blank, obj=obj, psf=psf, ilm=ilm, pad=16, sigma=sigma)

itrue = s.get_model_image()
itrue += np.random.normal(0.0, sigma, size=itrue.shape)
strue = s.state.copy()
s.set_image(itrue)

def scan_sigma(s, n=200):
    sigmas = np.linspace(np.max(0.01,s.sigma-0.1), sigma+0.1, n)
    lls = []
    for ss in sigmas:
        s.sigma = ss
        s._update_ll_field()
        lls.append(s.loglikelihood())
    return sigmas, np.array(lls)

#raise IOError
if True:
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        runner.sample_particles(s, stepout=0.05)
        runner.sample_block(s, 'psf', stepout=0.1)
        runner.sample_block(s, 'ilm', stepout=0.1)
        runner.sample_block(s, 'off', stepout=0.1)
        runner.sample_block(s, 'zscale', stepout=0.1)

        if i >= burn:
            h.append(s.state.copy())

    h = np.array(h)

mu = h.mean(axis=0)
std = h.std(axis=0)
pl.figure(figsize=(20,4))
pl.errorbar(xrange(len(mu)), (mu-strue), yerr=5*std/np.sqrt(samples),
        fmt='.', lw=0.15, alpha=0.5)
pl.vlines([0,3*s.N-0.5, 4*s.N-0.5], -1, 1, linestyle='dashed', lw=4, alpha=0.5)
pl.hlines(0, 0, len(mu), linestyle='dashed', lw=5, alpha=0.5)
pl.xlim(0, len(mu))
pl.ylim(-0.02, 0.02)
pl.show()
