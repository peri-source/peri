import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from colloids.bamf import observers, sampler, model, engine, state

x = np.linspace(1, 10, 100)
y = 1.0*x + 4 + np.random.normal(0, 0.15, *x.shape)
m = model.LinearFitModel(x,y)

s = state.State(2)
block = s.block_all()

h1 = sampler.MHSampler(0.1, block=block)
h2 = sampler.SliceSampler(1.0, block=block)
h3 = sampler.HamiltonianMCSampler(block=block)

engines = []
samplers = [h1, h2, h3]

for sampler in samplers:
    ohist = observers.HistogramObserver()
    state_obs = [ohist]

    eng = engine.SequentialBlockEngine(m, s)
    eng.add_samplers(sampler)
    eng.add_state_observers(state_obs)
    engines.append(eng)


for e in engines:
    s.reset()
    e.dosteps(1250, burnin=True)
    e.dosteps(10000, burnin=False)

for e in engines:
    hh = np.array(e.state_obs[0].dat)
    pl.figure(0)
    pl.hist(hh[:,0], bins=100, histtype='step')
    pl.figure(1)
    pl.hist(hh[:,1], bins=100, histtype='step')
    pl.show()
