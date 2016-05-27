import numpy as np
import scipy.ndimage as nd

from peri import util, states, initializers
from peri.mc import sample

# poly fit function, because I can
def poly_fit(x, y, order=2, sigma=0.1, N=100, burn=100):
    """
    generate data:
    x = np.linspace(0, 1, 10000)
    y = np.polyval(np.random.randn(4), x) + 0.05*np.random.randn(10000)
    """
    from peri.states import PolyFitState
    s = PolyFitState(x, y, order=order, sigma=sigma)
    h = sample.sample_state(s, s.params, N=burn, doprint=True, procedure='uniform')
    h = sample.sample_state(s, s.params, N=burn, doprint=True, procedure='uniform')

    import pylab as pl
    pl.plot(s.data, 'o')
    pl.plot(s.model, '-')
    return s, h.get_histogram()


