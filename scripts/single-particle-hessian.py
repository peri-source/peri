import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import states, runner
from cbamf.comp import psfs, ilms, objs
from cbamf.viz import plots

PAD = 16
sigma = 0.05

imsize = (64,)*3
blank = np.zeros(imsize, dtype='float')
xstart, rstart = np.array(imsize).reshape(-1,3)/2.0, np.array([5.0])

obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian((2, 4), shape=imsize, error=3e-3)
ilm = ilms.Polynomial3D(order=(1,1,1), shape=imsize)
s = states.ConfocalImagePython(blank, obj=obj, psf=psf, ilm=ilm, pad=PAD, sigma=sigma)

np.random.seed(10)
itrue = s.get_model_image()
itrue += np.random.normal(0.0, sigma, size=itrue.shape)
strue = s.state.copy()
s.set_image(itrue)

hess = s.hessloglikelihood()

pl.figure()
pl.imshow(np.log10(np.abs(hess)))
pl.title("Log Hessian matrix")
pl.colorbar()
pl.show()

#h, ll = runner.do_samples(s, 30, 5)
#plots.summary_plot(s, h, truestate=strue)
