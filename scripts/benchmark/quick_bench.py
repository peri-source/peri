import numpy as np
import scipy as sp
import pylab as pl
import itertools

from cbamf import initializers, runner
from cbamf.test import init, analyze, bench
from cbamf.viz.util import COLORS
from trackpy import locate

def fit_single_particle_rad(radii, samples=100, imsize=64, sigma=0.05):
    terrors = []
    berrors = []
    crbs = []

    for rad in radii:
        print '='*79
        print 'radius =', rad

        s = init.create_single_particle_state(imsize, radius=rad, sigma=0.05)
        p = s.state[s.b_pos].reshape(-1,3).copy()

        bl = s.explode(s.b_pos)
        crbs.append(np.sqrt(np.diag(np.linalg.inv(s.fisher_information(blocks=bl)))).reshape(-1,3))
        tmp_tp, tmp_bf = [],[]
        for i in xrange(samples):
            print i
            bench.jiggle_particles(s, pos=p)
            t = bench.trackpy(s)
            b = bench.bamfpy_positions(s, sweeps=30)

            tmp_tp.append(bench.error(s, t))
            tmp_bf.append(bench.error(s, b))
        terrors.append(tmp_tp)
        berrors.append(tmp_bf)

    return np.array(crbs), np.array(terrors), np.array(berrors)

def fit_single_particle_psf(psf_scale, samples=100, imsize=64, sigma=0.05):
    terrors = []
    berrors = []
    crbs = []

    psf0 = np.array([2.0, 1.0, 4.0])
    for scale in psf_scale:
        print '='*79
        print 'scale =', scale

        s = init.create_single_particle_state(imsize, radius=5.0,
                sigma=0.05, psfargs={'params': scale*psf0})
        p = s.state[s.b_pos].reshape(-1,3).copy()

        bl = s.explode(s.b_pos)
        crbs.append(np.sqrt(np.diag(np.linalg.inv(s.fisher_information(blocks=bl)))).reshape(-1,3))
        tmp_tp, tmp_bf = [],[]
        for i in xrange(samples):
            print i
            bench.jiggle_particles(s, pos=p)
            t = bench.trackpy(s)
            b = bench.bamfpy_positions(s, sweeps=30)

            tmp_tp.append(bench.error(s, t))
            tmp_bf.append(bench.error(s, b))
        terrors.append(tmp_tp)
        berrors.append(tmp_bf)

    return np.array(crbs), np.array(terrors), np.array(berrors)

def fit_two_particle_separation(separation, radius=5.0, samples=100, imsize=64, sigma=0.05):
    terrors = []
    berrors = []
    crbs = []

    for sep in separation:
        print '='*79
        print 'sep =', sep

        s = init.create_two_particle_state(imsize, radius=radius, delta=sep, sigma=0.05, axis='z')
        p = s.state[s.b_pos].reshape(-1,3).copy()

        bl = s.explode(s.b_pos)
        crbs.append(np.sqrt(np.diag(np.linalg.inv(s.fisher_information(blocks=bl)))).reshape(-1,3))
        tmp_tp, tmp_bf = [],[]
        for i in xrange(samples):
            print i
            bench.jiggle_particles(s, pos=p)
            t = bench.trackpy(s)
            b = bench.bamfpy_positions(s, sweeps=30)

            tmp_tp.append(bench.error(s, t))
            tmp_bf.append(bench.error(s, b))
        terrors.append(tmp_tp)
        berrors.append(tmp_bf)

    return np.array(crbs), np.array(terrors), np.array(berrors)

def plot_errors_single(rad, crb, errors, labels=['trackpy', 'cbamf']):
    fig = pl.figure()
    comps = ['z', 'y', 'x']
    markers = ['o', '^', '*']
    colors = COLORS

    for i in reversed(xrange(3)):
        pl.plot(rad, crb[:,0,i], lw=2.5, label='CRB-'+comps[i], color=colors[i])

    for c, (error, label) in enumerate(zip(errors, labels)):
        mu = np.sqrt((error**2).mean(axis=1))[:,0,:]
        std = np.std(np.sqrt((error**2)), axis=1)[:,0,:]

        for i in reversed(xrange(len(mu[0]))):
            pl.plot(rad, mu[:,i], marker=markers[c], color=colors[i], lw=0, label=label+"-"+comps[i], ms=13)

    pl.ylim(1e-3, 8e0)
    pl.semilogy()
    pl.legend(loc='upper left', ncol=3, numpoints=1, prop={"size": 16})
    pl.xlabel(r"radius (pixels)")
    pl.ylabel(r"CRB / $\Delta$ (pixels)")
    """ 
    ax = fig.add_axes([0.6, 0.6, 0.28, 0.28])
    ax.plot(rad, crb[:,0,:], lw=2.5)
    for c, error in enumerate(errors):
        mu = np.sqrt((error**2).mean(axis=1))[:,0,:]
        std = np.std(np.sqrt((error**2)), axis=1)[:,0,:]

        for i in xrange(len(mu[0])):
            ax.errorbar(rad, mu[:,i], yerr=std[:,i], fmt=markers[c], color=colors[i], lw=1)
    ax.set_ylim(-0.1, 1.5)
    ax.grid('off')
    """

def plot_errors_two(rad, crb, errors, labels=['trackpy', 'cbamf']):
    fig = pl.figure()
    comps = ['z', 'y', 'x']
    colors = ['r', 'g', 'b']
    markers = ['o', '^', '*']

    for i in reversed(xrange(3)):
        pl.plot(rad, crb[:,0,i], lw=2.5, label='CRB-'+comps[i], color=colors[i])

    for c, (error, label) in enumerate(zip(errors, labels)):
        mu = np.sqrt((error**2).mean(axis=1))[:,0,:]
        std = np.std(np.sqrt((error**2)), axis=1)[:,0,:]

        for i in reversed(xrange(len(mu[0]))):
            pl.plot(rad, mu[:,i], marker=markers[c], color=colors[i], lw=0, label=label+"-"+comps[i], ms=13)

    pl.ylim(1e-3, 8e0)
    pl.loglog()
    #pl.legend(loc='lower left', ncol=3, numpoints=1, prop={"size": 16})
    pl.xlabel(r"$\Delta z$ (pixels)")
    pl.ylabel(r"CRB / $\Delta$ (pixels)")

def plot_errors_psf(rad, crb, errors, labels=['trackpy', 'cbamf']):
    fig = pl.figure()
    comps = ['z', 'y', 'x']
    colors = ['r', 'g', 'b']
    markers = ['o', '^', '*']

    for i in reversed(xrange(3)):
        pl.plot(rad, crb[:,0,i], lw=2.5, label='CRB-'+comps[i], color=colors[i])

    for c, (error, label) in enumerate(zip(errors, labels)):
        mu = np.sqrt((error**2).mean(axis=1))[:,0,:]
        std = np.std(np.sqrt((error**2)), axis=1)[:,0,:]

        for i in reversed(xrange(len(mu[0]))):
            pl.plot(rad, mu[:,i], marker=markers[c], color=colors[i], lw=0, label=label+"-"+comps[i], ms=13)

    pl.ylim(1e-3, 8e0)
    pl.semilogy()
    #pl.legend(loc='lower left', ncol=3, numpoints=1, prop={"size": 16})
    pl.xlabel(r"$\sigma_z$ (pixels)")
    pl.ylabel(r"CRB / $\Delta$ (pixels)")

def doall():
    r = linspace(2.0, 10.0, 20)
    s = logspace(-3, np.log10(5), 2)
    crb,a,b = fit_single_particle_rad(r, samples=30)
    crb,a,b = fit_two_particle_separation(s, samples=30)

    plot_errors_single(s, crb, [a,b])
