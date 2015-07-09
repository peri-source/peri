import numpy as np
import pylab as pl

from cbamf import runner
from cbamf.test import init

samples = 10
sweeps = 30


# the final answers, crbs and stds
crbs = []
stds = []
hists = []

# options for the size distribution
rads = np.logspace(0, 1, 50)
rads2 = np.logspace(0, 1, 10)

for rad in rads:
    print "Radius", rad

    # create a single particle state and get pos/rad blocks
    s = init.create_single_particle_state(imsize=64, radius=rad, sigma=0.05)
    blocks = s.blocks_particle(0)

    crb = []
    for block in blocks:
        crb.append( s.fisher_information([block])[0,0] )
    crbs.append(crb)

for rad in rads2:
    print "Radius", rad

    # create a single particle state and get pos/rad blocks
    s = init.create_single_particle_state(imsize=64, radius=rad, sigma=0.05)
    blocks = s.blocks_particle(0)

    hist = []
    for i in xrange(samples):
        print "radius", rad, "sample", i
        s.model_to_true_image()

        thist = []
        for j in xrange(sweeps):
            thist.append(runner.sample_particles(s, stepout=0.1))
        hist.append(thist)
    hist = np.array(hist)
    hists.append(hist)

colors = np.array([
    (15,77,192,255), (250,196,4,255), (50,152,51,255), (241,57,50,255)
]) / 255.0

hists = np.array(hists)

crbs = 1.0 / np.sqrt(np.array(crbs))
stdmu = hists.std(axis=-2).mean(axis=-2)
stderr = hists.std(axis=-2).std(axis=-2)

pl.figure()
pl.loglog(rads, crbs[:,0], '-', lw=1.5, c=colors[0], alpha=0.6, label='CRB pos-z')
pl.loglog(rads, crbs[:,1], '-', lw=1.5, c=colors[1], alpha=0.6, label='CRB pos-y')
pl.loglog(rads, crbs[:,2], '-', lw=1.5, c=colors[2], alpha=0.6, label='CRB pos-x')
pl.loglog(rads, crbs[:,3], '-', lw=1.5, c=colors[3], alpha=0.6, label='CRB rad')

pl.errorbar(rads2, stdmu[:,0], yerr=stderr[:,0], fmt='o', lw=1.5, c=colors[0], label='STD pos-z')
pl.errorbar(rads2, stdmu[:,1], yerr=stderr[:,1], fmt='o', lw=1.5, c=colors[1], label='STD pos-y')
pl.errorbar(rads2, stdmu[:,2], yerr=stderr[:,2], fmt='o', lw=1.5, c=colors[2], label='STD pos-x')
pl.errorbar(rads2, stdmu[:,3], yerr=stderr[:,3], fmt='o', lw=1.5, c=colors[3], label='STD rad')

pl.legend(loc='upper right', numpoints=1, ncol=2, prop={'size': 20})
pl.xlabel("Radius")
pl.ylabel("Variance")
pl.show()
