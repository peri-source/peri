import numpy as np
from peri import states, runner, util
from peri.comp import psfs, objs, ilms, exactpsf, GlobalScalarComponent
from peri.viz import interaction
from peri.test import nbody

im = util.Image(np.zeros((32,)*3))
pos, rad = nbody.create_configuration(10, im.tile)

P = objs.PlatonicSpheresCollection(pos, rad)
H = psfs.Gaussian4D()
#I = ilms.LegendrePoly2P1D(constval=1.0)
I = ilms.BarnesStreakLegPoly2P1D(npts=(10,3))
B = GlobalScalarComponent('bkg', 0.0)
C = GlobalScalarComponent('offset', 0.0)

s = states.ImageState(im, [B, I, H, P, C], pad=16, model_as_data=True)
I.randomize_parameters()
s = states.ImageState(im, [B, I, H, P, C], pad=16, model_as_data=True)
