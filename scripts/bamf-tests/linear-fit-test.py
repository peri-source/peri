import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from colloids.bamf import observers, sampler, model
x = np.linspace(1, 10, 100)
y = 1.0*x + 4 + np.random.normal(0, 1.15, *x.shape)
m = model.LinearFitModel(x,y)

h = sampler.MHSampler(0.1, block=np.s_[:])
h = sampler.SliceSampler(1.0, block=np.s_[:])
h = sampler.HamiltonianMCSampler(block=np.s_[:])

ocovr = observers.CovarianceObserver()
ohist = observers.HistogramObserver()
oauto = observers.TimeAutoCorrelation()
state_obs = [ocovr, ohist]
likel_obs = [oauto]

s = np.array([0.,0.])

ll = None
for i in xrange(250):
    ll,s = h.sample(m, s, ll)

for i in xrange(40000):
    ll,s = h.sample(m, s, ll)

    if i % 1 == 0:
        for o in state_obs:
            o.update(s)
        for o in likel_obs:
            o.update(ll)

hh = np.array(ohist.dat)
pl.figure()
pl.hist(hh[:,0], bins=100, histtype='stepfilled')
pl.figure()
pl.hist(hh[:,1], bins=100, histtype='stepfilled')

pl.figure()
pl.plot(x,y, 'o')
pl.plot(x, m.calculate(ocovr.get_mean()))

pl.figure()
pl.plot(oauto.get_correlation())

pl.figure()
pl.plot(hh[:,0], hh[:,1], 'o-')

#pl.figure()
#i,a,b = pl.histogram2d(hh[:,0], hh[:,1], bins=50)
#pl.imshow(i, cmap=pl.cm.bone, interpolation='nearest', origin='lower')
#pl.show()
