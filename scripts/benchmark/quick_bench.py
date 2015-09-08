import numpy as np
import scipy as sp

from cbamf import initializers, runner
from cbamf.test import init
from trackpy import locate

def bamfpy(state, sweeps=50, burn=10):
    h,l = runner.do_samples(state, sweeps, burn, stepout=0.05, sigma=False)
    return h[:,s.b_pos].mean(axis=0).reshape(-1,3)

def trackpy(state):
    image = totiff(state)
    diameter = int(2*state.state[state.b_rad].mean())
    diameter -= 1 - diameter % 2
    out = locate(image, diameter=diameter, invert=True,
            minmass=100*(diameter/2)**3)
    return np.vstack([out.z, out.y, out.x]).T + state.pad

def nearest(p0, p1):
    ind = []
    for i in xrange(len(p0)):
        dist = np.sqrt(((p0[i] - p1)**2).sum(axis=-1))
        ind.append(dist.argmin())
    return ind

def totiff(state):
    p = state.pad
    q = initializers.normalize(state.image[p:-p, p:-p, p:-p])
    return (q*255).astype('uint8')

def jiggle_particles(state, pos=None, sig=0.5, indices=(0,)):
    if pos is None:
        pos = (np.array(state.image.shape)/2,)*len(indices)

    for i, p in enumerate(indices):
        tpos = pos[i]

        bl = state.explode(state.block_particle_pos(p))
        for j, b in enumerate(bl):
            state.update(b, tpos[j]+np.random.rand()*sig)

    state.model_to_true_image()

def fit_single_particle_rad(radii, samples=100, imsize=64, sigma=0.05):
    errors = []

    for rad in radii:
        s = init.create_single_particle_state(imsize, radius=rad, sigma=0.05)
        p = s.state[s.b_pos].copy()

        for i in xrange(samples):
            jiggle_particles(s, pos=p)


#init.create_single_particle_state()
#init.create_two_particle_state()

