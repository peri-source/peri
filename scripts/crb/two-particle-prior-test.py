import numpy as np
import pylab as pl

from peri import runner
from peri.test import init

samples = 30
sweeps = 100
sigma = 0.5
radius = 5.0

# the final answers, crbs and stds
crbs = []
stds = []
hists = []

# options for the size distribution
deltas = np.logspace(-3, 0, 10)
deltas2 = np.logspace(-3, 0, 5)

for delta in deltas:
    print "Delta", delta

    # create a single particle state and get pos/rad blocks
    s = init.create_two_particle_state(imsize=64, radius=radius, delta=delta,
            sigma=sigma, psfargs=(3,6), stateargs={"doprior": False})
    print s.obj.pos
    blocks = s.blocks_particle(0)

    crb = []
    for block in blocks:
        crb.append( s.fisher_information([block])[0,0] )
    crbs.append(crb)

for delta in deltas2:
    print "Delta", delta

    # create a single particle state and get pos/rad blocks
    s = init.create_two_particle_state(imsize=64, radius=radius, delta=delta,
            sigma=sigma, stateargs={"doprior": True})

    hist = []
    for i in xrange(samples):
        print "delta", delta, "sample", i
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
pl.loglog(deltas, crbs[:,0], '-', lw=1.5, c=colors[0], alpha=0.6, label='CRB pos-z')
pl.loglog(deltas, crbs[:,1], '-', lw=1.5, c=colors[1], alpha=0.6, label='CRB pos-y')
pl.loglog(deltas, crbs[:,2], '-', lw=1.5, c=colors[2], alpha=0.6, label='CRB pos-x')
pl.loglog(deltas, crbs[:,3], '-', lw=1.5, c=colors[3], alpha=0.6, label='CRB rad')

pl.errorbar(deltas2, stdmu[:,0], yerr=stderr[:,0], fmt='o', lw=1.5, c=colors[0], label='STD pos-z')
pl.errorbar(deltas2, stdmu[:,1], yerr=stderr[:,1], fmt='o', lw=1.5, c=colors[1], label='STD pos-y')
pl.errorbar(deltas2, stdmu[:,2], yerr=stderr[:,2], fmt='o', lw=1.5, c=colors[2], label='STD pos-x')
pl.errorbar(deltas2, stdmu[:,6], yerr=stderr[:,3], fmt='o', lw=1.5, c=colors[3], label='STD rad')

pl.legend(loc='upper right', numpoints=1, ncol=2, prop={'size': 20})
pl.xlabel("Radius")
pl.ylabel("Variance")
pl.show()
