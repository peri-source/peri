import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import states, runner
from cbamf.comp import psfs, ilms, objs
from cbamf.viz import plots

PAD = 16
sigma = 0.05

imsize = (64,64,64)
blank = np.zeros(imsize, dtype='float')
xstart, rstart = np.array(imsize).reshape(-1,3)/2.0, np.array([5.0])

obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian((1, 2), shape=imsize)
ilm = ilms.Polynomial3D(order=(1,1,1), shape=imsize)
s = states.ConfocalImagePython(blank, obj=obj, psf=psf, ilm=ilm, pad=PAD, sigma=sigma)

itrue = s.get_model_image()
itrue += np.random.normal(0.0, sigma, size=itrue.shape)
strue = s.state.copy()
s.set_image(itrue)

hess = -s.hessloglikelihood()
fischer = np.linalg.inv(hess)
