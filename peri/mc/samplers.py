"""
The classes that deal with sampling with respect
to a model and its parameters
"""
from builtins import range, object

from copy import deepcopy
import numpy as np

class Sampler(object):
    def __init__(self, block=None):
        self.block = block

    def getstate(self, state, substate):
        state.update(self.block, substate)
        return state

    def loglikelihood(self, state, substate):
        state.update(self.block, substate)
        return state.loglikelihood

    def gradloglikelihood(self, state, substate):
        state.update(self.block, substate)
        return state.gradloglikelihood()

    def sample(self, state):
        pass

class TWalkSampler(Sampler):
    def __init__(self, *args, **kwargs):
        import pytwalk
        super(TWalkSampler, self).__init__(*args, **kwargs)
        self.walker = pytwalk.pytwalk(n=1, U=self.ll, Supp=self.supp)

    def ll(self, x):
        return -self.loglikelihood(state, x)

    def supp(self, x):
        return True

    def sample(self, state, curloglike=None):
        self.state = state
        self.walker.Run(T=2, x0=self.state[self.block], xp0=self.state[self.block]+0.01)
        xout = self.walker.Output[-1,0]
        return self.loglikelihood(self.state, xout), self.getstate(state, xout)

class MHSampler(Sampler):
    def __init__(self, var=None, varsteps=True, *args, **kwargs):
        super(MHSampler, self).__init__(*args, **kwargs)
        self.var = var or 1
        self.loglike = None
        self.varsteps = varsteps

    def sample(self, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        tvar = self.var*np.random.rand() if self.varsteps else self.var

        px = curloglike or self.loglikelihood(state, x)
        xp = x + tvar*(2*np.random.rand(*size)-1)
        pp = self.loglikelihood(state, xp)

        if np.log(np.random.rand()) < (pp - px):
            return pp, self.getstate(state, xp)
        return px, self.getstate(state, x)

class SliceSampler(Sampler):
    def __init__(self, width=None, maxsteps=50, *args, **kwargs):
        super(SliceSampler, self).__init__(*args, **kwargs)
        self.width = width or 1
        self.maxsteps = maxsteps

    def sample(self, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        px = curloglike or self.loglikelihood(state, x)
        up = np.log(np.random.rand()) + px

        r = np.random.rand(*size)
        xl = x - r*self.width
        xr = x + (1-r)*self.width

        pl = self.loglikelihood(state, xl)
        pr = self.loglikelihood(state, xr)

        stepl = np.floor(self.maxsteps * np.random.rand())
        stepr = self.maxsteps - 1 - stepl
        while stepl > 0 and up < pl:
            xl = xl - self.width
            pl = self.loglikelihood(state, xl)
            stepl -= 1

        while stepr > 0 and up < pr:
            xr = xr + self.width
            pr = self.loglikelihood(state, xr)
            stepr -= 1

        steps = 0
        xr0 = xr
        xl0 = xl
        while True:
            xp = (xr-xl)*np.random.rand(*size) + xl
            pp = self.loglikelihood(state, xp)

            if up < pp:
                return pp, self.getstate(state, xp)

            xr[xp > x] = xp[xp > x]
            xl[xp < x] = xp[xp < x]
            steps += 1

            if steps > self.maxsteps:
                return up, self.getstate(state, x)

class SliceSampler1D(Sampler):
    def __init__(self, width=None, maxsteps=10, procedure='uniform', aparam=6, mixrate=0.1, *args, **kwargs):
        super(SliceSampler1D, self).__init__(*args, **kwargs)
        self.width = width or 1
        self.maxsteps = maxsteps
        self.aparam = aparam
        self.procedure = procedure
        self.mixrate = mixrate

        self._procedure = ['uniform', 'doubling', 'overrelaxed']
        if self.procedure not in self._procedure:
            raise AttributeError("Stepout procedure '%s' not recognized, must be one of %r" % (self.procedure, self._procedure))

        if self.procedure == 'doubling':
            raise AttributeError("Doubling not currently supported")

    def stepout_uniform(self, state, x0, p0):
        u = np.random.rand()
        xl = x0 - self.width*u
        xr = x0 + self.width*(1-u)

        v = np.random.rand()
        ml = np.floor(self.maxsteps * v)
        mr = (self.maxsteps - 1) - ml

        pl = self.loglikelihood(state, xl)
        pr = self.loglikelihood(state, xr)

        while ml > 0 and p0 < pl:
            xl = xl - self.width
            ml = ml - 1
            pl = self.loglikelihood(state, xl)

        while mr > 0 and p0 < pr:
            xr = xr + self.width
            mr = mr - 1
            pr = self.loglikelihood(state, xr)

        return xl, xr

    def stepout_doubling(self, state, x0, p0):
        u = np.random.rand()
        xl = x0 - self.width*u
        xr = x0 + self.width*(1-u)

        k = self.maxsteps

        pl = self.loglikelihood(state, xl)
        pr = self.loglikelihood(state, xr)

        while k > 0 and (p0 < pl or p0 < pr):
            v = np.random.rand()
            if v < 0.5:
                xl = xl - (xr - xl)
                pl = self.loglikelihood(state, xl)
            else:
                xr = xr + (xr - xl)
                pr = self.loglikelihood(state, xr)
            k = k - 1

        return xl, xr

    def sampling_uniform(self, state, xl, xr, x0, p0):
        steps = 0
        while True:
            u = np.random.rand()
            x1 = xl + u*(xr - xl)

            p1 = self.loglikelihood(state, x1)

            if p0 < p1:
                return p1, x1

            if x1 < x0:
                xl = x1
            else:
                xr = x1

            steps += 1

            if steps > self.maxsteps:
                return p0, x0

    def sampling_doubling(self, state, xl, xr, x0, p0):
        size = x0.shape

        x1 = xl + (xr - xl)*np.random.rand()
        p1 = self.loglikelihood(state, x1)

        d = False
        while xr - xl > 1.1*self.width:
            m = (xl + xr)/2
            x1 = xl + (xr - xl)*np.random.rand()
            p1 = self.loglikelihood(state, x1)

            if ((x0 < m and x1 > m) or (x0 >= m and x1 < m)):
                d = True

            if x1 < m:
                r = m
            else:
                l = m

            pl = self.loglikelihood(state, xl)
            pr = self.loglikelihood(state, xr)

            if d and p0 > pl and p0 > pr:
                return p0, x0

        return p1, x1

    def sampling_overrelaxed(self, state, xl, xr, x0, p0):
        a = self.aparam
        w = self.width

        if xr - xl < 1.1*w:
            while True:
                xm = (xl + xr)/2
                pm = self.loglikelihood(state, xm)

                if a == 0 or p0 < pm:
                    break

                if x0 > xm:
                    xl = xm
                else:
                    xr = xm

                a = a - 1
                w = w/2

        while a > 0:
            a = a - 1
            w = w/2

            xls = xl + w
            xrs = xr - w

            pls = self.loglikelihood(state, xls)
            prs = self.loglikelihood(state, xrs)

            if p0 >= pls:
                xl = xls
            if p0 >= prs:
                xr = xrs

        x1 = xl + xr - x0
        p1 = self.loglikelihood(state, x1)

        if x1 < xl or x1 > xr or p0 > p1:
            x1 = x0
            p1 = p0

        return p1, x1

    def sample(self, state, curloglike=None):
        x = state.state[self.block]

        if not isinstance(x, float):
            if np.array(state.state[self.block]).shape[0] > 1:
                raise AttributeError("SliceSampler1D cannot have multidimensional blocks")

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        px = curloglike or self.loglikelihood(state, x)
        up = np.log(np.random.rand()) + px

        if self.procedure == 'uniform' or self.procedure == 'overrelaxed':
            xl, xr = self.stepout_uniform(state, x, up)
        if self.procedure == 'doubling':
            xl, xr = self.stepout_doubling(state, x, up)

        if self.procedure == 'uniform':
            ll, xn = self.sampling_uniform(state, xl, xr, x, up)
        if self.procedure == 'doubling':
            ll, xn = self.sampling_doubling(state, xl, xr, x, up)
        if self.procedure == 'overrelaxed':
            if np.random.rand() < self.mixrate:
                ll, xn = self.sampling_uniform(state, xl, xr, x, up)
            else:
                ll, xn = self.sampling_overrelaxed(state, xl, xr, x, up)

        return ll, self.getstate(state, xn)

class HamiltonianMCSampler(Sampler):
    def __init__(self, steps=1, tau=5, eps=1e-2, *args, **kwargs):
        super(HamiltonianMCSampler, self).__init__(*args, **kwargs)
        self.steps = steps
        self.tau = tau
        self.eps = eps

    def sample(self, state, curloglike=None):
        size = state.state[self.block].shape
        x = state.state[self.block]

        if not isinstance(x, np.ndarray):
            x = np.array([x])

        g = -self.gradloglikelihood(state, x)
        E = -self.loglikelihood(state, x)

        for l in range(0, self.steps):
            p = np.random.normal(0, 1, size)
            H = p.dot(p)/2 + E

            xnew, gnew = x, g
            for i in range(0, self.tau):
                p = p - self.eps*gnew/2
                xnew = xnew + self.eps*p
                gnew = -self.gradloglikelihood(state, xnew)
                p = p - self.eps*gnew/2

            Enew = -self.loglikelihood(state, xnew)
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

