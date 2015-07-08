import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import runner
from cbamf.test import init
from cbamf.viz import plots

crbs = []
rads = np.arange(1, 10, 1./5)
rads = np.linspace(1, 10, 39)
rads = np.logspace(0, 1, 50)

for rad in rads:
    print "Radius", rad
    s = init.create_single_particle_state(imsize=64, radius=rad, sigma=0.05)
    blocks = s.blocks_particle(0)

    crb = []
    for block in blocks:
        crb.append( s.fisher_information([block])[0,0] )
    crbs.append(crb)

crbs = 1.0 / np.sqrt(np.array(crbs))

pl.figure()
pl.loglog(rads, crbs[:,0], 'o-', lw=1, label='pos-z')
pl.loglog(rads, crbs[:,1], 'o-', lw=1, label='pos-y')
pl.loglog(rads, crbs[:,2], 'o-', lw=1, label='pos-x')
pl.loglog(rads, crbs[:,3], 'o-', lw=1, label='rad')
pl.legend(loc='upper right')
pl.xlabel("Radius")
pl.ylabel("CRB")
pl.show()
