import pylab as pl
import numpy as np

from peri import states, runner, util
from peri.comp import psfs, objs, ilms, exactpsf, GlobalScalar
from peri.viz import interaction
from peri.test import nbody

im = util.NullImage(np.zeros((32,)*3))
pos, rad, tile = nbody.create_configuration(30, im.tile)

def make_image_0():
    P = objs.PlatonicSpheresCollection(pos, rad)
    H = psfs.AnisotropicGaussian()
    I = ilms.BarnesStreakLegPoly2P1D(npts=(20,10), local_updates=True)
    B = GlobalScalar('bkg', 0.0)
    C = GlobalScalar('offset', 0.0)
    I.randomize_parameters()
    
    return states.ImageState(im, [B, I, H, P, C], pad=16, model_as_data=True)

def make_image_1():
    P = objs.PlatonicSpheresCollection(pos, rad)
    H = psfs.AnisotropicGaussian()
    I = ilms.LegendrePoly3D(order=(5,3,3), constval=1.0)
    B = ilms.Polynomial3D(order=(3,1,1), category='bkg', constval=0.01)
    C = GlobalScalar('offset', 0.0)
    
    return states.ImageState(im, [B, I, H, P, C], pad=16, model_as_data=True)

s = make_image_1()
j = s.J()
jtj = np.dot(j, j.T)

#pl.imshow(np.log10(np.abs(j[:,::10])), aspect=50)
#pl.imshow(np.log10(np.abs(jtj)+1e-6))
pl.imshow((np.abs(jtj)+1e-6)**0.2)
pl.xticks(np.arange(len(s.params)), s.params, fontsize=8, rotation='vertical')
pl.yticks(np.arange(len(s.params)), s.params, fontsize=8)
pl.tight_layout()
pl.show()
