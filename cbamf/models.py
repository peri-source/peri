import numpy as np
from scipy.special import j1
from .cu import nbl

class Model(object):
    def __init__(self, logpriors=None, gradlogpriors=None):
        self.evaluations = 0
        self.logpriors = logpriors
        self.gradlogpriors = gradlogpriors

    def calculate(self, state):
        self.evaluations += 1
        return self.docalculate(state)

    def update_state(self):
        pass

    def loglikelihood(self, state):
        loglike = self.dologlikelihood(state)
        if self.logpriors is not None:
            loglike += self.logpriors(state)
        return loglike

    def gradloglikelihood(self, state):
        gradloglike = self.dogradloglikelihood(state)
        if self.gradlogpriors is not None:
            gradloglike += self.gradlogpriors(state)
        return gradloglike

    def negloglikelihood(self, state):
        return -self.loglikelihood(state)

    def neggradloglikelihood(self, state):
        return -self.gradloglikelihood(state)

class LinearFitModel(Model):
    def __init__(self, x, y, *args, **kwargs):
        super(LinearFitModel, self).__init__(*args, **kwargs)
        self.dx, self.dy = (np.array(i) for i in zip(*sorted(zip(x, y))))

    def plot(self, state):
        import pylab as pl
        pl.figure()
        pl.plot(self.dx, self.dy, 'o')
        pl.plot(self.dx, self.docalculate(state), '-')
        pl.show()

    def docalculate(self, state):
        return state.state[0]*self.dx + state.state[1]

    def dologlikelihood(self, state):
        return -((self.calculate(state) - self.dy)**2).sum()

    def dogradloglikelihood(self, state):
        pre = -2*(self.calculate(state) - self.dy)
        grad0 = pre.dot(self.dx)
        grad1 = pre.dot(np.ones(self.dx.shape))
        return np.array([grad0, grad1])

class PositionsRadiiPSF(Model):
    def __init__(self, *args, **kwargs):
        super(PositionsRadiiPSF, self).__init__(*args, **kwargs)

    def has_negrad(self, state):
        rad = state.state[state.b_rad]
        return (rad < 0).any()

    def has_overlaps(self, state):
        pos = state.state[state.b_pos]
        rad = state.state[state.b_rad]
        return nbl.naive_overlap(pos, rad, state.state[state.b_zscale][0], 0)

    def docalculate(self, state, docheck=True):
        if docheck:
            if self.has_overlaps(state):
                return -np.inf
            if self.has_negrad(state):
                return -np.inf

        return state.loglikelihood()

    def dologlikelihood(self, state):
        return self.calculate(state)

    def dogradloglikelihood(self, state):
        grad = 0.0*state
        pre = -2*(self.calculate(state) - self.dy)
        grad[self.b_pos] = 0
        grad[self.b_rad] = 0
        return pre*grad
