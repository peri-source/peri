import numpy as np
import scipy as sp
import scipy.ndimage as nd

from cbamf import const, runner
from cbamf.test import init
from cbamf.states import prepare_image

# the factor of coarse-graining, goal particle size, and larger size
f = 8
sigma = 0.05

goalsize = 8
goalpsf = np.array([2.0, 1.0, 3.0])

bigsize = goalsize * f
bigpsf = goalpsf * f

s0 = init.create_single_particle_state(imsize=4*bigsize,
        radius=bigsize, psfargs={'params': bigpsf, 'error': 1e-6})
s0.obj.pos += (f-1)/2.0
s0.reset()

# coarse-grained image
sl = np.s_[s0.pad:-s0.pad,s0.pad:-s0.pad,s0.pad:-s0.pad]

m = s0.get_model_image()[sl]
e = m.shape[0]

# indices for coarse-graining
i = np.linspace(0, e/f, e, endpoint=False).astype('int')
x,y,z = np.meshgrid(*(i,)*3, indexing='ij')
ind = x + e*y + e*e*z

# finally, c-g'ed image
cg = nd.mean(m, labels=ind, index=np.unique(ind)).reshape(e/f, e/f, e/f)
image = cg + np.random.randn(*cg.shape)*sigma
image = np.pad(image, const.PAD, mode='constant', constant_values=const.PADVAL)

s = init.create_single_particle_state(imsize=4*goalsize, sigma=sigma,
        radius=goalsize, psfargs={'params': goalpsf, 'error': 1e-6})
s.set_image(image)
diff = s.image - s.get_model_image()
h,l = runner.do_samples(s, 30, 0)
