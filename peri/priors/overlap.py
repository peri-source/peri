from builtins import range, zip, object

import itertools
import numpy as np
import scipy as sp

from peri.logger import log

class HardSphereOverlapNaive(object):
    def __init__(self, pos, rad, zscale=1, prior_type='absolute'):
        self.N = rad.shape[0]
        self.pos = pos
        self.rad = rad
        self.zscale = np.array([zscale, 1, 1])
        self.logpriors = np.zeros_like(rad)

        if prior_type == 'absolute':
            self.prior_func = lambda x: (x < 0)*ZEROLOGPRIOR
        self._calculate()

    def _calculate(self):
        self.logpriors = np.zeros_like(self.rad)

        for i in range(self.N-1):
            o = np.arange(i+1, self.N)

            dist = ((self.zscale*(self.pos[i] - self.pos[o]))**2).sum(axis=-1)
            dist0 = (self.rad[i] + self.rad[o])**2

            update = self.prior_func(dist - dist0)
            self.logpriors[i] += np.sum(update)
            self.logpriors[o] += update

        """
        # This is equivalent
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                d = ((self.zscale*(self.pos[i] - self.pos[j]))**2).sum(axis=-1)
                r = (self.rad[i] + self.rad[j])**2

                cost = self.prior_func(d - r)
                self.logpriors[i] += cost
                self.logpriors[j] += cost
        """

    def update(self, particles, pos, rad, typ):
        self.pos[particles] = pos
        self.rad[particles] = rad

        self._calculate()

    def logprior(self):
        return self.logpriors.sum()


class HardSphereOverlapCell(object):
    def __init__(self, pos, rad, typ, bounds=None, cutoff=None, zscale=1, maxn=30,
            prior_type='absolute'):

        # the mild inflation is to deal with numerical issues
        # at the absolute boundaries
        if bounds is None:
            bounds = (
                pos.min(axis=0)-0.1*np.abs(pos.min(axis=0)),
                pos.max(axis=0)+0.1*np.abs(pos.max(axis=0))
            )
        if cutoff is None:
            cutoff = 2.1 * rad.max()

        # setup the big box that the particles live in
        self.bounds = bounds
        self.bl, self.br = np.array(bounds[0]), np.array(bounds[1])
        self.bdiff = self.br - self.bl

        self.N = rad.shape[0]
        self.cutoff = cutoff
        self.pos = pos.copy()
        self.rad = rad.copy()
        self.typ = typ.copy()
        self.maxn = maxn
        self.zscale = np.array([zscale, 1, 1])
        self.logpriors = np.zeros_like(rad)
        self.inds = [[] for i in range(self.N)]
        self.neighs = [{} for i in range(self.N)]

        if prior_type == 'absolute':
            self.prior_func = lambda x: (x < 0)*ZEROLOGPRIOR
        self._initialize()

    def _initialize(self):
        self.size = (self.bdiff / self.cutoff).astype('int')
        self.size += 1

        self.cells = np.zeros(tuple(self.size) + (self.maxn,), dtype='int') - 1
        self.counts = np.zeros(self.size, dtype='int')

        for i in range(self.N):
            if self.typ[i] == 1:
                self._bin_particle(i)

    def _pos_to_inds(self, pos):
        ind = (self.size * (pos - self.bl) / self.bdiff).astype('int')
        return [[np.s_[ind[0], ind[1], ind[2]], ind]]

    def _unbin_particle(self, index):
        inds = self.inds[index]

        for n in self.neighs[index].keys():
            dlogprior = self.neighs[n].pop(index)
            self.logpriors[n] -= dlogprior

        for ind,_ in inds:
            cell = self.cells[ind]
            p = np.where(cell == index)[0]
            cell[p] = cell[self.counts[ind]-1]
            cell[self.counts[ind]-1] = -1
            self.cells[ind] = cell
            self.counts[ind] -= 1

        self.inds[index] = []
        self.neighs[index] = {}

    def _bin_particle(self, index):
        inds = self._pos_to_inds(self.pos[index])

        for ind,q in inds:
            try:
                self.cells[ind][self.counts[ind]] = index
            except IndexError as e:
                self.inds[index] = []
                self.neighs[index] = {}
                self.logpriors[index] = ZEROLOGPRIOR
                return

            self.counts[ind] += 1

        self.inds[index] = inds

        neighs = self._neighbors(index).astype('int')
        for n in neighs:
            co = self._logprior(index, n)
            self.neighs[index][n] = co
            self.neighs[n][index] = co

            self.logpriors[n] += co

        self.logpriors[index] = np.sum(self.neighs[index].values())

    def _logprior(self, i, j):
        dd = self._dist_diff(self.pos[i], self.pos[j],
                np.array(self.rad[i]+self.rad[j]), self.zscale)
        return self.prior_func(dd)

    def _dist_diff(self, p0, p1, r1r2, zs):
        a = zs*(p0-p1)
        dist = np.dot(a, a)
        dist0 = r1r2*r1r2
        return dist - dist0

    def _dist_diff2(self, p0, p1, r1r2, zs):
        from scipy import weave
        code = """
            double dist = 0.0;
            for (int i=0; i<3; i++){
                double d = zs[i]*(p0[i] - p1[i]);
                dist += d*d;
            }
            o[0] = dist - r1r2[0]*r1r2[0];
        """
        o = np.zeros(1)
        weave.inline(code, ["p0", "p1", "r1r2", "zs", "o"])
        return o[0]

    def _gentiles(self, loc):
        return itertools.product(
                range(max(loc[0]-1,0), min(loc[0]+2, self.size[0])),
                range(max(loc[1]-1,0), min(loc[1]+2, self.size[1])),
                range(max(loc[2]-1,0), min(loc[2]+2, self.size[2]))
            )

    def _neighbors(self, i):
        locs = self.inds[i]

        neighs = []
        for _,loc in locs:
            tiles = self._gentiles(loc)

            for tile in tiles:
                cell = self.cells[tile]
                count = self.counts[tile]
                neighs.extend(cell[:count])

        neighs = np.unique(np.array(neighs))
        neighs = np.delete(neighs, np.where((neighs == i) | (neighs == -1)))
        return neighs

    def update(self, index, pos, rad, typ):
        for i,p,r,t in zip(index, pos, rad, typ):
            if self.typ[i] == 1:
                self._unbin_particle(i)

            self.pos[i] = p
            self.rad[i] = r
            self.typ[i] = t

            if self.typ[i] == 1:
                self._bin_particle(i)

    def logprior(self):
        return self.logpriors.sum()

def test():
    N = 128
    for i in range(50):
        log.info('{}'.format(i))
        x = np.random.rand(N, 3)
        r = 0.05*np.random.rand(N)

        a = HardSphereOverlapNaive(x, r)
        b = HardSphereOverlapCell(x, r)

        assert((a.logpriors == b.logpriors).all())

        for j in range(100):
            l = np.random.randint(N, size=1)
            pp = x[l]#np.random.rand(3)
            rp = 0.05*np.random.rand(1)
            a.update(l, pp, rp)
            b.update(l, pp, rp)

            if not (a.logpriors == b.logpriors).all():
                log.info('{} {} {}'.format(l, pp, rp))
                log.info('{}'.format((a.logpriors - b.logpriors).sum()))
                raise IOError
