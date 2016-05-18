import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from peri import states, runner, initializers
from peri.comp import psfs, ilms, objs
from peri.viz import plots
import pickle
import time

sigma = 0.05

sweeps = 30
samples = 20
burn = sweeps - samples

imsize = (128,128,128)
blank = np.zeros(imsize, dtype='float')
xstart, rstart = pickle.load(open("/media/scratch/bamf/bamf_ic_16_xr.pkl", 'r'))
initializers.remove_overlaps(xstart, rstart)

obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian((0.6, 2), shape=imsize, fftw_planning_level=psfs.FFTW_PLAN_FAST)
ilm = ilms.Polynomial3D(order=(3,3,2), shape=imsize)
s = states.ConfocalImagePython(blank, obj=obj, psf=psf, ilm=ilm, pad=16, sigma=sigma)

itrue = s.get_model_image()
itrue += np.random.normal(0.0, sigma, size=itrue.shape)
strue = s.state.copy()
s.set_image(itrue)

blocks = s.explode(s.block_all())
h = []
for i in xrange(sweeps):
    print '{:=^79}'.format(' Sweep '+str(i)+' ')

    np.random.shuffle(blocks)
    for block in blocks:
        runner.sample_state(s, [block], stepout=0.1)

    if i >= burn:
        h.append(s.state.copy())

h = np.array(h)
