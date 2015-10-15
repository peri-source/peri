import numpy as np
import scipy as sp
import scipy.ndimage as nd

from cbamf import const, runner
from cbamf.test import init
from cbamf.states import prepare_image

# the factor of coarse-graining, goal particle size, and larger size
f = 8
sigma = 1e-5

goalsize = 8
goalpsf = np.array([2.0, 1.0, 3.0/f])

bigsize = goalsize * f
bigpsf = goalpsf * f

s0 = init.create_single_particle_state(imsize=np.array((4*goalsize, 4*bigsize, 4*bigsize)),
        radius=bigsize, psfargs={'params': bigpsf, 'error': 1e-6}, stateargs={'zscale': 1.0*f})
s0.obj.pos[:,1:] += (f-1.0)/2.0
s0.reset()

# coarse-grained image
sl = np.s_[s0.pad:-s0.pad,s0.pad:-s0.pad,s0.pad:-s0.pad]

m = s0.get_model_image()[sl]
e = m.shape[1]

# indices for coarse-graining
i = np.linspace(0, e/f, e, endpoint=False).astype('int')
j = np.linspace(0, e/f, e/f, endpoint=False).astype('int')
z,y,x = np.meshgrid(*(j,i,i), indexing='ij')
ind = x + e*y + e*e*z

# finally, c-g'ed image
cg = nd.mean(m, labels=ind, index=np.unique(ind)).reshape(e/f, e/f, e/f)
image = cg + np.random.randn(*cg.shape)*sigma
image = np.pad(image, const.PAD, mode='constant', constant_values=const.PADVAL)

s = init.create_single_particle_state(imsize=4*goalsize, sigma=sigma,
        radius=goalsize, psfargs={'params': goalpsf, 'error': 1e-6})
s.set_image(image)
diff = s.image - s.get_model_image()
h,l = runner.do_samples(s, 15, 0)
diff2 = s.image - s.get_model_image()
