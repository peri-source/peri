from peri import states, util, models
from peri.comp import psfs, objs, ilms, GlobalScalar, ComponentCollection
from peri.test import nbody

import peri.opt.optimize as opt
import peri.opt.addsubtract as addsub

im = util.NullImage(shape=(32,)*3)
pos, rad, tile = nbody.create_configuration(3, im.tile)

P = ComponentCollection([
    objs.PlatonicSpheresCollection(pos, rad),
    objs.Slab(2)
], category='obj')

H = psfs.AnisotropicGaussian()
I = ilms.BarnesStreakLegPoly2P1D(npts=(25,13,3), zorder=2, local_updates=False)
B = ilms.LegendrePoly2P1D(order=(3,1,1), category='bkg', constval=0.01)
C = GlobalScalar('offset', 0.0)
I.randomize_parameters()

s = states.ImageState(im, [B, I, H, P, C], pad=16, model_as_data=True)

