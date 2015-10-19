import numpy as np
import scipy as sp
import pylab as pl
import itertools

from cbamf import initializers, runner
from cbamf import runner, const
from cbamf.test import init, bench

def fit_edge_z(separation, radius=5.0, samples=100, imsize=64, sigma=0.05):
    terrors = []
    berrors = []
    crbs = []

    for sep in separation:
        print '='*79
        print 'sep =', sep

        s = init.create_two_particle_state(imsize, radius=radius, delta=sep, sigma=0.05,
                axis='z', stateargs={'sigmapad': False, 'pad': const.PAD})

        # move the particles to the edge
        bl = s.blocks_particle(0)
        s.update(bl[0], np.array([s.pad+radius]))

        bl = s.blocks_particle(1)
        s.update(bl[0], np.array([s.pad-radius-sep]))
        s.model_to_true_image()

        # save where the particles were originally so we can jiggle
        p = s.state[s.b_pos].reshape(-1,3).copy()

        # calculate the CRB for this configuration
        bl = s.explode(s.b_pos)
        crbs.append(np.sqrt(np.diag(np.linalg.inv(s.fisher_information(blocks=bl)))).reshape(-1,3))

        # calculate the featuring errors
        tmp_tp, tmp_bf = [],[]
        for i in xrange(samples):
            print i
            bench.jiggle_particles(s, pos=p, sig=0.3)
            t = bench.trackpy(s)
            b = bench.bamfpy_positions(s, sweeps=15)

            tmp_tp.append(bench.error(s, t))
            tmp_bf.append(bench.error(s, b))
        terrors.append(tmp_tp)
        berrors.append(tmp_bf)

    return np.array(crbs), np.array(terrors), np.array(berrors)

def fit_edge_xy(separation, radius=5.0, samples=100, imsize=64, sigma=0.05):
    terrors = []
    berrors = []
    crbs = []

    for sep in separation:
        print '='*79
        print 'sep =', sep

        s = init.create_two_particle_state(imsize, radius=radius, delta=sep, sigma=0.05,
                axis='z', stateargs={'sigmapad': False, 'pad': const.PAD})

        # honestly, i'm not really sure what this does
        d = np.array([0,0.5,0.5])
        s.psf.error = 1e-8
        s.psf._memoize_clear()
        s.obj.pos -= d
        s.reset()

        # move the particles to the edge
        bl = s.blocks_particle(0)
        s.update(bl[0], np.array([s.pad+radius]))

        bl = s.blocks_particle(1)
        s.update(bl[0], np.array([s.pad-radius]))

        bl = s.blocks_particle(1)
        s.update(bl[1], s.state[bl[1]]+sep)
        s.model_to_true_image()

        # save where the particles were originally so we can jiggle
        p = s.state[s.b_pos].reshape(-1,3).copy()

        # calculate the CRB for this configuration
        bl = s.explode(s.b_pos)
        crbs.append(np.sqrt(np.diag(np.linalg.inv(s.fisher_information(blocks=bl)))).reshape(-1,3))

        # calculate the featuring errors
        tmp_tp, tmp_bf = [],[]
        for i in xrange(samples):
            print i
            bench.jiggle_particles(s, pos=p, sig=0.3)
            t = bench.trackpy(s)
            b = bench.bamfpy_positions(s, sweeps=15)

            tmp_tp.append(bench.error(s, t))
            tmp_bf.append(bench.error(s, b))
        terrors.append(tmp_tp)
        berrors.append(tmp_bf)

    return np.array(crbs), np.array(terrors), np.array(berrors)

