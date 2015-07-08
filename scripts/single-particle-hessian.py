import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import states, runner
from cbamf.comp import psfs, ilms, objs
from cbamf.viz import plots
np.random.seed(10)

sigma = 0.05

imsize = np.array((64,)*3)
image, pos, rad = states.prepare_for_state(np.zeros(imsize), imsize.reshape(-1,3)/2.0, 5)

imsize = image.shape
obj = objs.SphereCollectionRealSpace(pos=pos, rad=rad, shape=imsize)
psf = psfs.AnisotropicGaussian((2, 4), shape=imsize, error=3e-3)
ilm = ilms.Polynomial3D(order=(1,1,1), shape=imsize)
s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm, sigma=sigma)
s.model_to_true_image()

hess = s.hessloglikelihood()

pl.figure()
pl.imshow(np.log10(np.abs(hess)))
pl.title("Log Hessian matrix")
pl.colorbar()
pl.show()

#h, ll = runner.do_samples(s, 30, 5)
#plots.summary_plot(s, h, truestate=strue)
