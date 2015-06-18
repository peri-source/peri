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
    def __init__(self, var=None, width=None, maxsteps=50, *args, **kwargs):
        super(SliceSampler, self).__init__(*args, **kwargs)
        self.width = width or 1
        self.maxsteps = maxsteps

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

        stepl = np.floor(self.maxsteps * np.random.rand())
        stepr = self.maxsteps - 1 - stepl
        while stepl > 0 and up < pl:
            xl = xl - self.width
            pl = self.loglikelihood(model, state, xl)
            stepl -= 1

        while stepr > 0 and up < pr:
            xr = xr + self.width
            pr = self.loglikelihood(model, state, xr)
            stepr -= 1

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

            if steps > self.maxsteps:
                return up, self.getstate(state, x)

class SliceSampler1D(Sampler):
    def __init__(self, var=None, width=None, maxsteps=10, procedure='uniform', *args, **kwargs):
        super(SliceSampler, self).__init__(*args, **kwargs)
        self.width = width or 1
        self.maxsteps = maxsteps
        self.procedure = procedure

        self._procedure = ['uniform', 'doubling']
        if self.procedure not in self._procedure:
            raise AttributeError("Stepout procedure '%s' not recognized, must be one of %r" % (self.procedure, self._procedure))

    def stepout_uniform(self, model, state, x0, p0):
        u = np.random.rand()
        xl = x0 - self.width*u
        xr = x0 + self.width*(1-u)

        v = np.random.rand()
        ml = np.floor(self.maxsteps * v)
        mr = (self.maxsteps - 1) - ml

        pl = self.loglikelihood(model, state, xl)
        pr = self.loglikelihood(model, state, xr)

        while ml > 0 and p0 < pl:
            xl = xl - self.width
            ml = ml - 1
            pl = self.loglikelihood(model, state, xl)

        while mr > 0 and p0 < pr:
            xr = xr + self.width
            mr = mr - 1
            pr = self.loglikelihood(model, state, xr)

        return xl, xr

    def stepout_doubling(self, model, state, x0, p0):
        u = np.random.rand()
        xl = x0 - self.width*u
        xr = x0 + self.width*(1-u)

        k = self.maxsteps

        pl = self.loglikelihood(model, state, xl)
        pr = self.loglikelihood(model, state, xr)

        while k > 0 and (p0 < pl and p0 < pr):
            v = np.random.rand()
            if v < 0.5:
                xl = xl - (xr - xl)
            else:
                xr = xr + (xr - xl)
            k = k - 1

        return xl, xr

    def sampling_uniform(self, model, state, xl, xr, x0, p0):
        while True:
            u = np.random.rand()
            x1 = xl + u*(xr - xl)

            p1 = self.loglikelihood(model, state, x1)
            if p0 < p1:
                return p0, x1

            xr[x1 > x0] = x1
            xl[x1 < x0] = x1

    def sampling_doubling(self, model, state, xl, xr, x0, p0):
        size = x0.shape
        if size[0] > 1:
            raise AttributeError("Shrink sampling cannot have multidimensional blocks")

        d = False

        v = np.random.rand()
        x1 = xl + (xr - xl)*v
        p1 = self.loglikelihood(model, state, m)

        while xr - xl > 1.1*self.width:
            m = (xl + xr)/2

            if ((x0 < m and x1 > m) or (x0 >= m and x1 < m)):
                d = True

            if x1 < m:
                r = m
            else:
                l = m

            pl = self.loglikelihood(model, state, xl)
            pr = self.loglikelihood(model, state, xr)

            if d and p0 > pl and p0 > pr:
                return up, x0

        return p1, x1

    def sample(self, model, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        px = curloglike or self.loglikelihood(model, state, x)
        up = np.log(np.random.rand()) + px

        if self.procedure == 'uniform':
            xl, xr = self.stepout_uniform(model, state, x, up)
        if self.procedure == 'doubling':
            xl, xr = self.stepout_doubling(model, state, x, up)


        if self.procedure == 'uniform':
            ll, xn = self.sampling_uniform(model, state, xl, xr, x, up)
        if self.procedure == 'doubling':
            ll, xn = self.sampling_doubling(model, state, xl, xr, x, up)

        return ll, self.getstate(state, xn)

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

