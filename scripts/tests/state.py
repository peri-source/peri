import pickle
import pylab as pl
import numpy as np

from peri import states, runner, util
from peri.comp import psfs, objs, ilms, exactpsf, GlobalScalar, ComponentCollection
from peri.viz import interaction
from peri.test import nbody

im = util.NullImage(shape=(32,)*3)
pos, rad = nbody.create_configuration(3, im.tile)

P = ComponentCollection([
    objs.PlatonicSpheresCollection(pos, rad),
    objs.Slab(2)
], category='obj')

H = psfs.AnisotropicGaussian()
I = ilms.LegendrePoly3D(order=(5,3,3), constval=1.0)
B = ilms.Polynomial3D(order=(3,1,1), category='bkg', constval=0.01)
C = GlobalScalar('offset', 0.0)
    
s = states.ImageState(im, [B, I, H, P, C], pad=16, model_as_data=True)
