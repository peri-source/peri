import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import states, runner, initializers
from cbamf.comp import psfs, ilms, objs
from cbamf.viz import plots
from cbamf.test import poissondisks
import pickle
import time

radius = 8.0
sigma = 0.050

sweeps = 30
samples = 20
burn = sweeps - samples

imsize = np.array((96,96,96))
blank = np.zeros(imsize, dtype='float')
xstart = poissondisks.DiskCollection(imsize-radius, 2*radius).get_positions()+radius/2
image, pos, rad = states.prepare_for_state(blank, xstart, radius)

obj = objs.SphereCollectionRealSpace(pos=pos, rad=rad, shape=image.shape)
psf = psfs.AnisotropicGaussian((1.6, 3), shape=image.shape)
ilm = ilms.Polynomial3D(order=(1,1,1), shape=image.shape)
s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm, sigma=sigma)
s.model_to_true_image()

strue = s.state.copy()

raise IOError
h, ll = runner.do_samples(s, sweeps, burn)
plots.sample_compare(s.N, h, strue)
