import numpy as np
from scipy.special import j1
from .cu import fields, nbl

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

class NaiveColloids(Model):
    def __init__(self, imtrue, N, Lz=16, *args, **kwargs):
        super(NaiveColloids, self).__init__(*args, **kwargs)
        self.N = N
        self.Lz = Lz
        self.imtrue = imtrue
        self.b_pos = np.s_[:3*N]
        self.b_rad = np.s_[3*N:4*N]
        self.b_psf = np.s_[4*N:]

        self.shape = self.imtrue.shape
        if len(self.shape) == 2:
            self.shape = self.shape + (self.Lz,)
        self.lastimage = 0*self.imtrue

    def _disc1(self, k, R):
        return 2*R*np.sin(k)/k

    def _disc2(self, k, R):
        return 2*np.pi*R**2 * j1(k) / k

    def _disc3(self, k, R):
        return 4*np.pi*R**3 * (np.sin(k)/k - np.cos(k))/k**2

    def _psf_disc(self, k, params):
        return (2*j1(params[0]*k)/(params[0]*k))**2

    def docalculate(self, state, full3d=False):
        # TODO - side-step the big calculation if it turns out that
        # there are particles overlapping.  return -inf
        pos = state[self.b_pos].reshape(-1,3)
        rad = state[self.b_rad]
        psf = state[self.b_psf]

        ccd = np.zeros(self.shape)
        kx = 2*np.pi*np.fft.fftfreq(ccd.shape[0])[:,None,None]
        ky = 2*np.pi*np.fft.fftfreq(ccd.shape[1])[None,:,None]
        kz = 2*np.pi*np.fft.fftfreq(ccd.shape[2])[None,None,:]
        kv = np.array(np.broadcast_arrays(kx,ky,kz)).T
        k = np.sqrt(kx**2 + ky**2 + kz**2)

        iccd = np.fft.fftn(ccd)
        for p0, r0 in zip(pos, rad):
            iccd += self._disc3(k*r0+1e-8, r0)*np.exp(-1.j*(kv*p0).sum(axis=-1)).T
        iccd[0,0,0] = self.N

        iccd *= self._psf_disc(k+1e-8, psf)
        ccd = np.real(np.fft.ifftn(iccd))

        if full3d:
            return cdd - ccd.min()
        return ccd[:,:,ccd.shape[-1]/2] - ccd.min()

    def dologlikelihood(self, state):
        return -((self.calculate(state) - self.imtrue)**2).sum()

    def dogradloglikelihood(self, state):
        grad = 0.0*state
        pre = -2*(self.calculate(state) - self.dy)
        grad[self.b_pos] = 0
        grad[self.b_rad] = 0
        return pre*grad

class PositionsRadiiPSF(Model):
    def __init__(self, imsig=0.1, LZ=32, *args, **kwargs):
        super(PositionsRadiiPSF, self).__init__(*args, **kwargs)
        self.Lz = LZ
        self.imsig = imsig

    def has_negrad(self, state):
        rad = state.state[state.b_rad]
        return (rad < 0).any()

    def has_overlaps(self, state):
        pos = state.state[state.b_pos]
        rad = state.state[state.b_rad]
        return nbl.naive_overlap(pos, rad, state.state[state.b_zscale][0], 0)

    def docalculate(self, state, docheck=True):
        # TODO - side-step the big calculation if it turns out that
        # there are particles overlapping.  return -inf
        if docheck:
            if self.has_overlaps(state):
                return -1e100
            if self.has_negrad(state):
                return -1e101

        state.create_final_image()
        return state.create_differences()

    def dologlikelihood(self, state):
        logl = self.calculate(state)
        if isinstance(logl, float):
            return logl
        return -(logl**2).sum() / self.imsig**2

    def dogradloglikelihood(self, state):
        grad = 0.0*state
        pre = -2*(self.calculate(state) - self.dy)
        grad[self.b_pos] = 0
        grad[self.b_rad] = 0
        return pre*grad
