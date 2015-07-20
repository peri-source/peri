import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import states
from cbamf.mc import observers, samplers, engines

x = np.linspace(1, 10, 100)
y = 1.0*x + 4 + np.random.normal(0, 0.15, *x.shape)
m = states.LinearFit(x,y)

block = m.block_all()

h1 = samplers.MHSampler(0.1, block=block)
h1 = samplers.SliceSampler(1.0, block=block)
h2 = samplers.SliceSampler(1.0, block=block)
h3 = [samplers.SliceSampler1D(block=b, procedure='overrelaxed') for b in m.explode(block)]

engs = []
samps = [h1, h2, h3]

for sampler in samps:
    ohist = observers.HistogramObserver()
    state_obs = [ohist]

    eng = engines.SequentialBlockEngine(m)
    eng.add_samplers(sampler)
    eng.add_state_observers(state_obs)
    engs.append(eng)


for e in engs:
    m.reset()
    e.dosteps(1250, burnin=True)
    e.dosteps(10000, burnin=False)

for e in engs:
    hh = np.array(e.state_obs[0].dat)
    pl.figure(0)
    pl.hist(hh[:,0], bins=100, histtype='step')
    pl.figure(1)
    pl.hist(hh[:,1], bins=100, histtype='step')
    pl.show()
