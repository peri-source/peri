import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from colloids.bamf import observers, sampler, model, engine, initialize

itrue, xstart = initialize.local_max_points("./data/colloid-2d-slice.png",
        cutsize=64, offset=64, zlevel=16, radius=3)
itrue, xstart = initialize.fake_image_2d(16)

N = xstart.shape[0]
pos0 = xstart.flatten()
rad0 = 9*np.ones(N)
psf0 = np.array([0.5, 5])
s = np.hstack([pos0, rad0, psf0]).astype('float32')

m = model.PositionsRadiiIsoPSF(itrue, N)
hx = sampler.NBodyMHSampler(N, var=100., nsteps=100, block=m.b_pos)
hr = sampler.MHSampler(0.4, block=m.b_rad)
hp = sampler.MHSampler(0.1, block=m.b_psf)

if True:
    hx = sampler.SliceSampler(0.1, block=m.b_pos)
    hr = sampler.SliceSampler(0.1, block=m.b_rad)
    hp = sampler.SliceSampler(0.1, block=m.b_psf)

engines = []
samplers = [hx, hr, hp]

ocovr = observers.CovarianceObserver()
ohist = observers.HistogramObserver()
oauto = observers.TimeAutoCorrelation()
opsay = observers.Printer(skip=50)
state_obs = [ocovr, ohist, oauto]
likel_obs = [opsay]

eng = engine.SequentialBlockEngine(m, s)
eng.add_samplers(samplers)
eng.add_state_observers(state_obs)
eng.add_likelihood_observers(likel_obs)

