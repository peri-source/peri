import sys
import numpy as np
from scipy.optimize import minimize

from . import models, samplers, engines, observers
from .cu import nbl

def renorm(s, doprint=1):
    p, r, z = s.state[s.b_pos], s.state[s.b_rad], s.state[s.b_zscale]
    nbl.naive_renormalize_radii(p, r, z[0], 1)
    s.state[s.b_pos], s.state[s.b_rad] = p, r

def sample_state(state, blocks, stepout=1, slicing=True, N=1, doprint=False):
    m = models.PositionsRadiiPSF()

    eng = engines.SequentialBlockEngine(m, state)
    opsay = observers.Printer()
    ohist = observers.HistogramObserver(block=blocks[0])
    eng.add_samplers([samplers.SliceSampler(stepout, block=b) for b in blocks])

    eng.add_likelihood_observers(opsay) if doprint else None
    eng.add_state_observers(ohist)

    eng.dosteps(N)
    return ohist

def sample_ll(state, element, size=0.1, N=1000):
    m = models.PositionsRadiiPSF()
    start = state.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    for val in vals:
        state.update(element, val)
        l = m.loglikelihood(state)
        ll.append(l)
    return vals, np.array(ll)

def scan_noise(image, state, element, size=0.01, N=1000):
    start = state.state[element]

    xs, ys = [], []
    for i in xrange(N):
        print i
        test = image + np.random.normal(0, state.sigma, image.shape)
        x,y = sample_ll(test, state, element, size=size, N=300)
        state.update(element, start)
        xs.append(x)
        ys.append(y)

    return xs, ys

def sample_particles(state, stepout=1):
    print '{:-^39}'.format(' POS / RAD ')
    for particle in xrange(state.obj.N):
        print particle
        sys.stdout.flush()

        renorm(state)

        if state.set_current_particle(particle):
            blocks = state.blocks_particle()
            sample_state(state, blocks, stepout=stepout)

def sample_block(state, blockname, explode=True, stepout=1):
    print '{:-^39}'.format(' '+blockname.upper()+' ')
    state.set_current_particle()
    blocks = [state.create_block(blockname)]

    if explode:
        blocks = state.explode(blocks[0])

    sample_state(state, blocks, stepout)

def build_bounds(state):
    bounds = []

    bound_dict = {
        'pos': (1,512),
        'rad': (0, 20),
        'typ': (0,1),
        'psf': (0, 10),
        'bkg': (-100, 100),
        'amp': (-3, 3),
        'zscale': (0.5, 1.5)
    }

    for i,p in enumerate(state.param_order):
        bounds.extend([bound_dict[p]]*state.param_lengths[i])
    return np.array(bounds)

def loglikelihood(vec, state):
    state.set_state(vec)
    state.set_current_particle()
    state.create_final_image()
    return -state.loglikelihood()

def gradloglikelihood(vec, state):
    state.set_state(vec)
    state.set_current_particle()
    return -state.gradloglikelihood()

def gradient_descent(state, method='L-BFGS-B'):
    bounds = build_bounds(state)
    minimize(loglikelihood, state.state, args=(state,),
            method=method, jac=gradloglikelihood, bounds=bounds)
