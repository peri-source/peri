import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import states, runner, initializers
from cbamf.comp import objs, psfs, ilms
from cbamf.viz import plots
import scipy.ndimage as nd

order = (3,3,2)
sigma = 0.05
raw = initializers.load_tiff("./0.tif")

feat = initializers.normalize(raw[13:,:256,:256], False)
ilm = ilms.Polynomial3D(order=order, shape=feat.shape)
s = states.IlluminationField(feat, ilm=ilm, sigma=sigma)

def fsmooth(im, sigma):
    kz, ky, kx = np.meshgrid(*[np.fft.fftfreq(i) for i in feat.shape], indexing='ij')
    ksq = kx**2 + ky**2 + kz**2
    kim = np.fft.fftn(im)
    kim *= np.exp(-ksq * sigma**2)
    return np.real(np.fft.ifftn(kim))

print "Featuring the illumination field"
h = runner.sample_state(s, s.explode(s.block_all()), N=5, doprint=True)
l = s.model_image - s.image
#m = fsmooth(l, 2.0)
m = nd.gaussian_filter(l, 2)
q = nd.maximum_filter(m, footprint=initializers.generate_sphere(6))
lbl = nd.label(q == m)[0]
pos = np.array(nd.center_of_mass(q, lbl, np.unique(lbl)))[1:]
