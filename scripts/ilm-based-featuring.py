import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import states, runner, initializers
from cbamf.comp import objs, psfs, ilms
from cbamf.viz import plots
import scipy.ndimage as nd

order = (5,5,4)
sigma = 0.05
raw = initializers.load_tiff("/media/scratch/bamf/frozen-particles/zstack_dx0/0.tif")

feat = initializers.normalize(raw[13:,:256,:256], False)
ilm = ilms.Polynomial3D(order=order, shape=feat.shape)
ilm.from_data(feat)

print "Featuring the illumination field"
l = ilm.get_field() - feat
m = nd.gaussian_filter(l, 2)
q = nd.maximum_filter(m, footprint=initializers.generate_sphere(4))
lbl = nd.label(q == m)[0]
pos = np.array(nd.center_of_mass(m, lbl, np.unique(lbl)))[1:]
