import numpy as np
import pylab as pl

from cbamf import runner
from cbamf.test import init

samples = 50

# the final answers, crbs and stds
crbs = []
stds = []

# options for the size distribution
rads = np.arange(1, 10, 1./5)
rads = np.linspace(1, 10, 39)
rads = np.logspace(0, 1, 10)

# create a single particle state and get pos/rad blocks
s = init.create_single_particle_state(imsize=64, radius=1, sigma=0.05)
blocks = s.blocks_particle(0)

# store the original configuration for a while
posrad = np.array(blocks).any(axis=0)
start = s.state[posrad].copy()

for rad in rads:
    print "Radius", rad
    s.update(posrad, start)
    s.update(blocks[-1], np.array([rad]))

    crb = []
    for block in blocks:
        crb.append( s.fisher_information([block])[0,0] )
    crbs.append(crb)

    h = []
    for i in xrange(samples):
        h.append(runner.sample_particles(s, stepout=0.1))
    h = np.array(h)
    stds.append(h.std(axis=0)) 

crbs = 1.0 / np.sqrt(np.array(crbs))

pl.figure()
pl.loglog(rads, crbs[:,0], 'o-', lw=1, label='crb pos-z')
pl.loglog(rads, crbs[:,1], 'o-', lw=1, label='crb pos-y')
pl.loglog(rads, crbs[:,2], 'o-', lw=1, label='crb pos-x')
pl.loglog(rads, crbs[:,3], 'o-', lw=1, label='crb rad')

pl.loglog(rads, stds[:,0], 'o-', lw=1, label='std pos-z')
pl.loglog(rads, stds[:,1], 'o-', lw=1, label='std pos-y')
pl.loglog(rads, stds[:,2], 'o-', lw=1, label='std pos-x')
pl.loglog(rads, stds[:,3], 'o-', lw=1, label='std rad')

pl.legend(loc='upper right')
pl.xlabel("Radius")
pl.ylabel("Variance")
pl.show()
