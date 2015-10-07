import numpy as np
import scipy as sp
import pylab as pl
import itertools

from cbamf import initializers, runner
from cbamf.test import analyze
from trackpy import locate

def bamfpy_full(state, sweeps=50, burn=10):
    h,l = runner.do_samples(state, sweeps, burn, stepout=0.05, sigma=False)
    return h[:,state.b_pos].mean(axis=0).reshape(-1,3)

def bamfpy_positions(state, sweeps=20, burn=5):
    blocks = list(itertools.chain.from_iterable([state.explode(state.block_particle_pos(i)) for i in xrange(state.N)]))
    vals = [state.state[bl] for bl in blocks]
    h = runner.sample_state(state, blocks, stepout=0.10, N=burn)
    h = runner.sample_state(state, blocks, stepout=0.10, N=sweeps, doprint=True)
    h = h.get_histogram()

    for bl, val in zip(blocks, vals):
        state.update(bl, val)
    return h.mean(axis=0).reshape(-1,3)

def trackpy(state):
    image = totiff(state)
    diameter = int(2*state.state[state.b_rad].mean())
    diameter -= 1 - diameter % 2
    out = locate(image, diameter=diameter, invert=True,
            minmass=145*(diameter/2)**3)
    return np.vstack([out.z, out.y, out.x]).T + state.pad

def error(state, pos):
    preal = state.state[state.b_pos].reshape(-1,3)
    ind = analyze.nearest(preal, pos)
    return preal - pos[ind]

def totiff(state):
    p = state.pad
    q = initializers.normalize(state.image[p:-p, p:-p, p:-p])
    return (q*255).astype('uint8')

def jiggle_particles(state, pos=None, sig=0.5, indices=None):
    if pos is None:
        pos = state.state[state.b_pos].reshape(-1,3)

    if indices is None:
        indices = xrange(state.N)

    noise = np.random.rand(3)*sig
    for i, p in enumerate(indices):
        tpos = pos[i]

        bl = state.explode(state.block_particle_pos(p))
        for j, b in enumerate(bl):
            state.update(b, tpos[j]+noise[j])#np.random.rand()*sig)

    state.model_to_true_image()


