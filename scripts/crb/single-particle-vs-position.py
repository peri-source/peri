import numpy as np
import pylab as pl

from cbamf.test import init

radius = 5.0
sigma = 0.1
crbs = []

s = init.create_single_particle_state(imsize=64, radius=radius, sigma=sigma)
positions = np.linspace(s.pad-1.5*radius, s.pad+2*radius, 50)
blocks = s.blocks_particle(0)

for pos in positions:
    print "Position", pos
    s.update(blocks[2], np.array([pos]))

    crb = []
    for block in blocks:
        crb.append( s.fisher_information([block])[0,0] )
    crbs.append(crb)

crbs = 1.0 / np.sqrt(np.array(crbs))

pl.figure()
pl.plot((positions-s.pad)/radius, crbs[:,0], 'o-', lw=1, label='pos-z')
pl.plot((positions-s.pad)/radius, crbs[:,1], 'o-', lw=1, label='pos-y')
pl.plot((positions-s.pad)/radius, crbs[:,2], 'o-', lw=1, label='pos-x')
pl.plot((positions-s.pad)/radius, crbs[:,3], 'o-', lw=1, label='rad')
pl.semilogy()
pl.legend(loc='upper right')
pl.xlabel("Position from edge (in radii)")
pl.ylabel("CRB")
pl.title("CRB for R=%0.2f SNR=%0.2f" % (radius, 1/sigma))
pl.show()
