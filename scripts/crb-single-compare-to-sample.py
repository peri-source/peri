import numpy as np
import pylab as pl

from cbamf import runner
from cbamf.test import init

samples = 100

# the final answers, crbs and stds
crbs = []
stds = []
hists = []

# options for the size distribution
rads = np.arange(1, 10, 1./5)
rads = np.linspace(1, 10, 39)
rads = np.logspace(0, 1, 10)

for rad in rads:
    print "Radius", rad

    # create a single particle state and get pos/rad blocks
    s = init.create_single_particle_state(imsize=64, radius=rad, sigma=0.05)
    blocks = s.blocks_particle(0)

    crb = []
    for block in blocks:
        crb.append( s.fisher_information([block])[0,0] )
    crbs.append(crb)

    hist = []
    for i in xrange(samples):
        hist.append(runner.sample_particles(s, stepout=0.1))
    hist = np.array(hist)
    hists.append(hist)
    stds.append(hist.std(axis=0)) 

crbs = 1.0 / np.sqrt(np.array(crbs))
stds = np.array(stds)

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
