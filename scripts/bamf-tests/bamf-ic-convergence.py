import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import acor
import pylab as pl
from colloids.cu import mc, nbody, fields
from colloids.bamf import observers, sampler, model, engine, initialize, state

GN = 8
GS = 0.10
PHI = 0.65
RADIUS = 8.0
PSF = (1.2, 2)
mc.setSeed(10)

itrue, xstart, rstart, pstart = initialize.fake_image_2d(GN, phi=PHI, noise=GS, radius=RADIUS, psf=PSF)
strue = np.hstack([xstart.flatten(), rstart, pstart]).copy()
#raise IOError

def perturb_state(im, pos, rad, partial=True):
    N = pos.shape[0]
    pos = pos.flatten().astype('float32')
    rad = rad.flatten().astype('float32')
    typ = nbody.ACTIVE * np.ones(N, dtype='int32')

    mc.propose_particle_positions(pos, rad, typ, 400., 18, pos)
    mc.propose_particle_radius(pos, rad, typ, RADIUS, 2, 1000)
    psf = np.array(PSF, dtype='float32')
    st = np.hstack([pos, rad, psf]).astype('float32')

    ml = model.PositionsRadiiPSF(itrue, imsig=GS)
    sl = state.StateXRP(N, state=st, partial=partial, ccd_size=im.shape[::-1])
    if ml.has_overlaps(sl):
        raise IOError
    return sl

def sample_state(N, st, slicing=True):
    m = model.PositionsRadiiPSF(itrue, imsig=GS)
    bs = s.blocks_const_size(1)
    samplers = [sampler.SliceSampler(RADIUS/1, block=b) for b in bs]

    eng = engine.SequentialBlockEngine(m, st)
    ohist = observers.HistogramObserver()
    opsay = observers.Printer(skip=1)
    eng.add_state_observers(ohist)
    eng.add_likelihood_observers(opsay)
    eng.add_samplers(samplers)

    eng.dosteps(10)
    eng.reset_observers()
    eng.dosteps(50)
    print m.evaluations
    return m, st, ohist.get_histogram()

for l, slicing in enumerate([False]*1):
    s = perturb_state(itrue, xstart, rstart)
    x = np.array(xrange(s.nparams))

    pl.figure(1)
    pl.plot(x + (2*l+0)*s.nparams, s.state-strue, 'o')

    m, st, h = sample_state(GN, s, slicing=slicing)
    mu = np.array([acor.acor(h[:,i])[1] for i in xrange(h[0,:].shape[0])], dtype='float32')
    std = np.array([acor.acor(h[:,i])[2] for i in xrange(h[0,:].shape[0])], dtype='float32')

    pl.figure(1)
    pl.errorbar(x + (2*l+1)*s.nparams, ((mu-strue)/strue), yerr=(std), fmt='o')

    pl.figure()
    #m = model.PositionsRadiiPSF(itrue, GN)
    pl.imshow((itrue-m.docalculate(mu))[10,:,:], cmap=pl.cm.bone, interpolation='nearest')
    pl.colorbar()

pl.figure(1)
pl.hlines(0, 0, (2*l+2)*nparams, linestyle='dashed', lw=4, alpha=0.4)
pl.show()
