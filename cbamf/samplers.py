"""
The classes that deal with sampling with respect
to a model and its parameters
"""
from copy import deepcopy
import numpy as np

def createBlock(imin, imax=None, skip=None):
    return np.s_[imin:imax:skip]

class Sampler(object):
    def __init__(self, block=None):
        self.block = block

    def getstate(self, state, substate):
        state.update(self.block, substate)
        return state

    def loglikelihood(self, model, state, substate):
        state.push_change(self.block, substate)
        lg = model.loglikelihood(state)
        state.pop_change()
        return lg

    def gradloglikelihood(self, model, state, substate):
        state.push_change(self.block, substate)
        lg = model.gradloglikelihood(state)
        state.pop_change()
        return lg

    def sample(self, model, state):
        pass

class MHSampler(Sampler):
    def __init__(self, var=None, varsteps=True, *args, **kwargs):
        super(MHSampler, self).__init__(*args, **kwargs)
        self.var = var or 1
        self.loglike = None
        self.varsteps = varsteps

    def sample(self, model, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        tvar = self.var*np.random.rand() if self.varsteps else self.var

        px = curloglike or self.loglikelihood(model, state, x)
        xp = x + tvar*(2*np.random.rand(*size)-1)
        pp = self.loglikelihood(model, state, xp)

        if np.log(np.random.rand()) < (pp - px):
            return pp, self.getstate(state, xp)
        return px, self.getstate(state, x)

class SliceSampler(Sampler):
    def __init__(self, var=None, width=None, *args, **kwargs):
        super(SliceSampler, self).__init__(*args, **kwargs)
        self.width = width or 1

    def sample(self, model, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        px = curloglike or self.loglikelihood(model, state, x)
        up = np.log(np.random.rand()) + px

        r = np.random.rand(*size)
        xl = x - r*self.width
        xr = x + (1-r)*self.width

        pl = self.loglikelihood(model, state, xl)
        pr = self.loglikelihood(model, state, xr)

        if px < -1e90:
            print "Starting with bad state"
            raise IOError

        steps = 0
        while (up < pl):
            xl = xl - self.width
            pl = self.loglikelihood(model, state, xl)
            steps += 1
            if steps > 100:
                print up, pl, self.width
                raise IOError
        steps = 0
        while (up < pr):
            xr = xr + self.width
            pr = self.loglikelihood(model, state, xr)
            steps += 1
            if steps > 100:
                print up, pl, self.width
                raise IOError

        steps = 0
        xr0 = xr
        xl0 = xl
        while True:
            xp = (xr-xl)*np.random.rand(*size) + xl
            pp = self.loglikelihood(model, state, xp)

            if up < pp:
                return pp, self.getstate(state, xp)

            xr[xp > x] = xp[xp > x]
            xl[xp < x] = xp[xp < x]
            steps += 1

            if steps > 100:
                print xl0, xr0, xl, xr, xp, pp, self.width
                raise IOError #return px, self.getstate(state, x)

class HamiltonianMCSampler(Sampler):
    def __init__(self, var=None, steps=1, tau=5, eps=1e-2, *args, **kwargs):
        super(HamiltonianMCSampler, self).__init__(*args, **kwargs)
        self.steps = steps
        self.tau = tau
        self.eps = eps

    def sample(self, model, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        g = -self.gradloglikelihood(model, state, x)
        E = -self.loglikelihood(model, state, x)

        for l in xrange(0, self.steps):
            p = np.random.normal(0, 1, size)
            H = p.dot(p)/2 + E

            xnew, gnew = x, g
            for i in xrange(0, self.tau):
                p = p - self.eps*gnew/2
                xnew = xnew + self.eps*p
                gnew = -self.gradloglikelihood(model, state, xnew)
                p = p - self.eps*gnew/2

            Enew = -self.loglikelihood(model, state, xnew)
            Hnew = p.dot(p)/2 + Enew
            dH = Hnew - H

            if dH < 0:
                accept = 1
            elif np.log(np.random.rand()) < -dH:
                accept = 1
            else:
                accept = 0

            if accept:
                g, x, E = gnew, xnew, Enew

        return E, self.getstate(state, xnew)

