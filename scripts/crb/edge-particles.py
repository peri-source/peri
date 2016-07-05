import numpy as np
import scipy as sp
import pylab as pl
import itertools

from peri import initializers, runner
from peri import runner, const
from peri.test import init, bench
from peri.viz.base import COLORS, MARKERS

def fit_edge(separation, radius=5.0, samples=100, imsize=64, sigma=0.05, axis='z'):
    """
    axis is 'z' or 'xy'
    seps = np.linspace(0,2,20) 'z'
    seps = np.linspace(-2,2,20) 'xy'
    """
    terrors = []
    berrors = []
    crbs = []

    for sep in separation:
        print '='*79
        print 'sep =', sep,

        s = init.create_two_particle_state(imsize, radius=radius, delta=sep, sigma=0.05,
                axis='z', psfargs={'params': (2.0,1.0,4.0), 'error': 1e-8},
                          stateargs={'sigmapad': True, 'pad': const.PAD})

        # move off of a pixel edge (cheating for trackpy)
        d = np.array([0,0.5,0.5])
        s.obj.pos -= d
        s.reset()

        # move the particles to the edge
        bl = s.blocks_particle(0)
        s.update(bl[0], np.array([s.pad+radius]))

        bl = s.blocks_particle(1)
        s.update(bl[0], np.array([s.pad-radius]))

        if axis == 'z':
            bl = s.blocks_particle(1)
            s.update(bl[0], s.state[bl[0]]-sep)
            s.model_to_true_image()

        if axis == 'xy':
            bl = s.blocks_particle(1)
            s.update(bl[2], s.state[bl[2]]+sep)
            s.model_to_true_image()

        # save where the particles were originally so we can jiggle
        p = s.state[s.b_pos].reshape(-1,3).copy()

        print p[0], p[1]
        # calculate the CRB for this configuration
        bl = s.explode(s.b_pos)
        crbs.append(np.sqrt(np.diag(np.linalg.inv(s.fisher_information(blocks=bl)))).reshape(-1,3))

        # calculate the featuring errors
        tmp_tp, tmp_bf = [],[]
        for i in xrange(samples):
            print i
            bench.jiggle_particles(s, pos=p, sig=0.3, mask=np.array([1,1,1]))
            t = bench.trackpy(s)
            b = bench.bamfpy_positions(s, sweeps=15)

            tmp_tp.append(bench.error(s, t))
            tmp_bf.append(bench.error(s, b))
        terrors.append(tmp_tp)
        berrors.append(tmp_bf)

    return np.array(crbs), np.array(terrors), np.array(berrors)

def plot_errors(seps, crb, errors, labels=['trackpy', 'peri']):
    fig = pl.figure()
    comps = ['z', 'y', 'x']
    markers = MARKERS
    colors = COLORS

    for i in reversed(xrange(3)):
        pl.plot(seps, crb[:,0,i], lw=2.5, label='CRB-'+comps[i], color=colors[i])

    for c, (error, label) in enumerate(zip(errors, labels)):
        mu = np.sqrt((error[:,:,0,:]**2)).mean(axis=1)#np.sqrt((error**2).mean(axis=1)).mean(axis=0)
        std = np.std(np.sqrt((error**2)), axis=1)

        for i in reversed(xrange(len(mu[0]))):
            pl.plot(seps, mu[:,i], marker=markers[c], color=colors[i], lw=0, label=label+"-"+comps[i], ms=13)

    pl.ylim(1e-3, 8e0)
    pl.semilogy()
    pl.legend(loc='upper left', ncol=3, numpoints=1, prop={"size": 16})
    pl.xlabel(r"Distance to edge (pixels)")
    pl.ylabel(r"CRB / $\Delta$ (pixels)")
