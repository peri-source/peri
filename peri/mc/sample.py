from builtins import str, range

import os
import sys
import tempfile
import pickle

import numpy as np

from peri import states, util
from peri.mc import samplers, engines, observers

from peri.logger import log
log = log.getChild("mc.sample")

#=============================================================================
# Sampling methods that run through blocks and sample
#=============================================================================
def sample_state(state, blocks, stepout=1, slicing=True, N=1, doprint=False, procedure='uniform'):
    eng = engines.SequentialBlockEngine(state)
    opsay = observers.Printer()
    ohist = observers.HistogramObserver(block=blocks)
    eng.add_samplers([
        samplers.SliceSampler1D(stepout, block=b, procedure=procedure)
        for b in util.listify(blocks)
    ])

    eng.add_likelihood_observers(opsay) if doprint else None
    eng.add_state_observers(ohist)

    eng.dosteps(N)
    return ohist

def scan_ll(state, element, size=0.1, N=1000):
    start = state.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    for val in vals:
        state.update(element, val)
        l = state.loglikelihood
        ll.append(l)

    state.update(element, start)
    return vals, np.array(ll)

def scan_noise(image, state, element, size=0.01, N=1000):
    start = state.state[element]

    xs, ys = [], []
    for i in range(N):
        log.info('{}'.format(i))
        test = image + np.random.normal(0, state.sigma, image.shape)
        x,y = sample_ll(test, state, element, size=size, N=300)
        state.update(element, start)
        xs.append(x)
        ys.append(y)

    return xs, ys

def sample_particles(state, stepout=1, start=0, quiet=False):
    if not quiet:
        log.info('{:-^39}'.format(' POS / RAD '))
    for particle in state.active_particles():
        if not quiet:
            log.info('{}'.format(particle))
        sys.stdout.flush()

        blocks = state.blocks_particle(particle)
        sample_state(state, blocks, stepout=stepout)

    return state.state.copy()

def sample_particle_pos(state, stepout=1, start=0, quiet=False):
    if not quiet:
        log.info('{:-^39}'.format(' POS '))

    for particle in state.active_particles():
        if not quiet:
            log.info('{}'.format(particle))
        sys.stdout.flush()

        blocks = state.blocks_particle(particle)[:-1]
        sample_state(state, blocks, stepout=stepout)

    return state.state.copy()

def sample_particle_rad(state, stepout=1, start=0, quiet=False):
    if not quiet:
        log.info('{:-^39}'.format(' RAD '))

    for particle in state.active_particles():
        if not quiet:
            log.info('{}'.format(particle))

        sys.stdout.flush()

        blocks = [state.blocks_particle(particle)[-1]]
        sample_state(state, blocks, stepout=stepout)

    return state.state.copy()

def sample_block(state, blockname, explode=True, stepout=0.1, quiet=False):
    if not quiet:
        log.info('{:-^39}'.format(' '+blockname.upper()+' '))

    blocks = [state.create_block(blockname)]

    if explode:
        blocks = state.explode(blocks[0])

    return sample_state(state, blocks, stepout)

def sample_block_list(state, blocklist, stepout=0.1, quiet=False):
    for bl in blocklist:
        sample_block(state, bl, stepout=stepout, quiet=quiet)
    return state.state.copy(), state.loglikelihood

def do_samples(s, sweeps, burn, stepout=0.1, save_period=-1,
        prefix='peri', save_name=None, sigma=True, pos=True, quiet=False, postfix=None):
    h = []
    ll = []
    if not save_name:
        with tempfile.NamedTemporaryFile(suffix='.peri-state.pkl', prefix=prefix) as f:
            save_name = f.name

    for i in range(sweeps):
        if save_period > 0 and i % save_period == 0:
            with open(save_name, 'w') as tfile:
                pickle.dump([s,h,ll], tfile, protocol=2)

        if postfix is not None:
            states.save(s, desc=postfix, extra=[np.array(h),np.array(ll)])

        if not quiet:
            log.info('{:=^79}'.format(' Sweep '+str(i)+' '))

        #sample_particles(s, stepout=stepout)
        if pos:
            sample_particle_pos(s, stepout=stepout, quiet=quiet)
        sample_particle_rad(s, stepout=stepout, quiet=quiet)
        sample_block(s, 'psf', stepout=stepout, quiet=quiet)
        sample_block(s, 'ilm', stepout=stepout, quiet=quiet)
        sample_block(s, 'off', stepout=stepout, quiet=quiet)
        sample_block(s, 'zscale', stepout=stepout, quiet=quiet)

        if s.bkg:
            sample_block(s, 'bkg', stepout=stepout, quiet=quiet)

        if s.slab:
            sample_block(s, 'slab', stepout=stepout, quiet=quiet)

        if sigma and s.nlogs:
            sample_block(s, 'sigma', stepout=stepout/10, quiet=quiet)

        if i >= burn:
            h.append(s.state.copy())
            ll.append(s.loglikelihood)

    if save_period > 0 and save_name:
        os.remove(save_name)

    h = np.array(h)
    ll = np.array(ll)
    return h, ll

def do_blocks(s, blocks, sweeps, burn, stepout=0.1, postfix=None, quiet=False):
    h, ll = [], []

    for i in range(sweeps):
        if postfix is not None:
            states.save(s, desc=postfix, extra=[np.array(h),np.array(ll)])

        if not quiet:
            log.info('{:=^79}'.format(' Sweep '+str(i)+' '))

        sample_state(s, blocks, stepout=stepout, N=1, doprint=~quiet)

        if i >= burn:
            h.append(s.state.copy())
            ll.append(s.loglikelihood)

    h = np.array(h)
    ll = np.array(ll)
    return h, ll


