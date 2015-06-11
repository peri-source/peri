import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import acor
import pylab as pl
from colloids.cu import mc, nbody, fields
from colloids.bamf import observers, sampler, model, engine, initialize, state
from time import sleep
import itertools
import sys
import pickle

GN = 64
GS = 0.01
PHI = 0.47
RADIUS = 8.0
PSF = (1.6, 2)
ORDER = (3,3,3)

sweeps = 20
samples = 10
burn = sweeps - samples

PAD = int(2*RADIUS)
SIZE = int(3*RADIUS)
TPAD = PAD+SIZE/2

def renorm(tstate, doprint=1):
    p, r = tstate.state[tstate.b_pos], tstate.state[tstate.b_rad]
    mc.naive_renormalize_radii(p, r, 1)
    tstate.state[tstate.b_pos], tstate.state[tstate.b_rad] = p, r

#pickle.dump([itrue, xstart, rstart, pstart], open("/media/scratch/bamf_ic.pkl", 'w'))
#itrue, xstart, rstart, pstart = pickle.load(open("/media/scratch/bamf_ic.pkl"))
ipure, itrue, xstart, rstart, pstart = initialize.fake_image_3d(GN, phi=PHI, noise=GS, radius=RADIUS, psf=PSF)
ipure = np.pad(ipure, TPAD, mode='constant', constant_values=-10)
itrue = np.pad(itrue, TPAD, mode='constant', constant_values=-10)
xstart = xstart + TPAD

bkgpoly = np.zeros(np.prod(ORDER))
strue = np.hstack([xstart.flatten(), rstart, pstart, bkgpoly]).copy()
tstate = state.StateXRPB(GN, itrue, pad=2*PAD, state=strue.copy().astype('float64'), order=ORDER)
renorm(tstate)

nbody.setSeed(10)
mc.setSeed(10)
np.random.seed(10)

def sample_state(image, st, blocks, slicing=True, N=1, doprint=False):
    m = model.PositionsRadiiPSF(image, imsig=GS)

    eng = engine.SequentialBlockEngine(m, st)
    opsay = observers.Printer()
    ohist = observers.HistogramObserver(block=blocks[0])
    eng.add_samplers([sampler.SliceSampler(RADIUS/1e1, block=b) for b in blocks])

    eng.add_likelihood_observers(opsay) if doprint else None
    eng.add_state_observers(ohist)

    eng.dosteps(N)
    m.free()
    return ohist

def sample_ll(image, st, element, size=0.1, N=1000):
    m = model.PositionsRadiiPSF(image, imsig=GS)
    start = st.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    for val in vals:
        st.update(element, val)
        l = m.loglikelihood(st)
        ll.append(l)
    m.free()
    return vals, np.array(ll)

def scan_noise(image, st, element, size=0.01, N=1000):
    start = st.state[element]

    xs, ys = [], []
    for i in xrange(N):
        print i
        test = image + np.random.normal(0, GS, image.shape)
        x,y = sample_ll(test, st, element, size=size, N=300)
        st.update(element, start)
        xs.append(x)
        ys.append(y)

    return xs, ys

#"""
#raise IOError
if True:
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        print '{:-^39}'.format(' POS / RAD ')
        for particle in xrange(tstate.N):
            print particle
            sys.stdout.flush()

            renorm(tstate)

            tstate.set_current_particle(particle, max_size=SIZE)
            blocks = tstate.blocks_particle()
            sample_state(itrue, tstate, blocks)

        print '{:-^39}'.format(' PSF ')
        tstate.set_current_particle(max_size=SIZE)
        blocks = tstate.explode(tstate.create_block('psf'))
        sample_state(itrue, tstate, blocks)

        print '{:-^39}'.format(' BKG ')
        tstate.set_current_particle(max_size=SIZE)
        blocks = (tstate.create_block('bkg'),)
        sample_state(itrue, tstate, blocks)

        if i > burn:
            h.append(tstate.state.copy())

    h = np.array(h)
    #return h

#h = cycle(itrue, xstart, rstart, pstart, sweeps, sweeps-samples, size=SIZE)
mu = h.mean(axis=0)
std = h.std(axis=0)
pl.figure(figsize=(20,4))
pl.errorbar(xrange(len(mu)), (mu-strue), yerr=5*std/np.sqrt(samples),
        fmt='.', lw=0.15, alpha=0.5)
pl.vlines([0,3*GN-0.5, 4*GN-0.5, 4*GN+tstate.psfn], -1, 1, linestyle='dashed', lw=4, alpha=0.5)
pl.hlines(0, 0, len(mu), linestyle='dashed', lw=5, alpha=0.5)
pl.xlim(0, len(mu))
pl.ylim(-0.02, 0.02)
pl.show()
#"""
