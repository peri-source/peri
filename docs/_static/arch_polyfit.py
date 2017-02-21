import numpy as np
import matplotlib.pyplot as pl

import peri
import peri.states
import peri.opt.optimize as opt

from peri.mc import sample

class PolyFitState(peri.states.State):
    def __init__(self, x, y, order=2, coeffs=None):
        self._data = y
        self._xpts = x

        params = ['c-%i' %i for i in xrange(order)]
        values = coeffs if coeffs is not None else [0.0]*order

        super(PolyFitState, self).__init__(
            params=params, values=values, ordered=False
        )

        self.update(self.params, self.values)

    def update(self, params, values):
        super(PolyFitState, self).update(params, values)
        self._model = np.polyval(self.values, self._xpts)

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

def init_random():
    # noise level
    sigma = 0.3

    # num of coefficients, datapoints
    C, N = 8, 1000

    # generate data
    np.random.seed(159)
    c = 2*np.random.rand(C) - 1
    x = np.linspace(0.0, 2.0, N)
    y = np.polyval(c, x) + sigma*np.random.randn(N)

    # create a state
    s = PolyFitState(x, y, order=C)
    return s

def show_jtj(s):
    fig = pl.figure()
    pl.imshow(s.JTJ(), cmap='bone')
    pl.xticks(np.arange(len(s.params)), s.params)
    pl.yticks(np.arange(len(s.params)), s.params)
    pl.title(r"$J^T J$ for PolyFitState")

def mc_sample(s):
    # burn a number of samples if hadn't optimized yet
    #h = sample.sample_state(s, s.params, N=1000, doprint=True, procedure='uniform')

    # then collect the samples around the true value
    h = sample.sample_state(s, s.params, N=30, doprint=True, procedure='uniform')

    # distribution of fit parameter values
    return h.get_histogram()

def opt_lm(s):
    opt.do_levmarq(s, s.params[1:])

    fig = pl.figure()
    pl.plot(s._xpts, s.data, 'o', label='Data')
    pl.plot(s._xpts, s.model, '-', label='Model')
    pl.xlabel(r"$x$")
    pl.ylabel(r"$y$")
    pl.legend(loc='best')
    pl.title("Best fit model")

if __name__ == '__main__':
    s = init_random()
    show_jtj(s)
    opt_lm(s)
    h = mc_sample(s)
