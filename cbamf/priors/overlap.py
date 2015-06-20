import itertools
import numpy as np
import scipy as sp
from priors import ZEROLOGPRIOR
from cbamf.util import Tile

class HardSphereOverlapNaive(object):
    def __init__(self, pos, rad, zscale=1):
        self.N = rad.shape[0]
        self.pos = pos
        self.rad = rad
        self.zscale = np.array([zscale, 1, 1])
        self.logpriors = np.zeros_like(rad)

        self._initialize()

    def _calculate_priors(self):
        self.size = (self.bdiff / self.cutoff).astype('int')
        self.size[self.size < 1] = 1

        self.cells = np.zeros(self.size + (self.maxn,))
        self.counts = np.zeros(self.size)

        for i in xrange(self.N):
            self._bin_particle(i)
        self._calculate_all_priors()

    def update(self, pos, rad, index):
        self._unbin_particle(index)

        self.pos[index] = pos
        self.rad[index] = rad

        self._bin_particle(index)

        for n in self._neighbors(index):
            self._calculate_prior(n)

    def logprior(self):
        return self.logpriors.sum()


class HardSphereOverlapCellBased(object):
    def __init__(self, pos, rad, bounds=None, cutoff=None, zscale=1, maxn=30):
        if bounds is None:
            bounds = (pos.min(axis=0)-1, pos.max(axis=0)+1)
        if cutoff is None:
            cutoff = 2 * rad.max()

        self.bounds = bounds
        self.bl, self.br = np.array(bounds[0]), np.array(bounds[1])
        self.bdiff = self.br - self.bl

        self.N = rad.shape[0]
        self.cutoff = cutoff
        self.pos = pos
        self.rad = rad
        self.maxn = maxn
        self.zscale = np.array([zscale, 1, 1])
        self.logpriors = np.zeros_like(rad)
        self.inds = [[]]*self.N

        self.prior_func = lambda x: (x < 0)*ZEROLOGPRIOR
        self._initialize()

    def _initialize(self):
        self.size = (self.bdiff / self.cutoff).astype('int')
        self.size += 1#[self.size < 1] = 1

        self.cells = np.zeros(tuple(self.size) + (self.maxn,), dtype='int')
        self.counts = np.zeros(self.size, dtype='int')

        for i in xrange(self.N):
            self._bin_particle(i)
        self._calculate_all_priors()

    def _pos_to_inds(self, pos):
        # TODO - add to multiple bins if bigger than cutoff
        ind = (self.size * (pos - self.bl) / self.bdiff).astype('int')
        return [[np.s_[ind[0], ind[1], ind[2]], ind]]

    def _unbin_particle(self, index):
        inds = self.inds[index]

        for ind,_ in inds:
            p = np.where(self.cells[ind] == index)[0]
            self.cells[p] = self.cells[self.counts[ind]-1]
            self.counts[ind] -= 1

        self.inds[index] = []

    def _bin_particle(self, index):
        inds = self._pos_to_inds(self.pos[index])

        for ind,q in inds:
            self.cells[ind][self.counts[ind]] = index
            self.counts[ind] += 1

        self.inds[index] = inds

    def _gentiles(self, loc):
        return itertools.product(
                xrange(max(loc[0]-1,0), min(loc[0]+2, self.size[0]-1)),
                xrange(max(loc[1]-1,0), min(loc[1]+2, self.size[1]-1)),
                xrange(max(loc[2]-1,0), min(loc[2]+2, self.size[2]-1))
            )

    def _neighbors(self, i):
        locs = self.inds[i]

        neighs = []

        for _,loc in locs:
            tiles = self._gentiles(loc)

            for tile in tiles:
                cell = self.cells[tile]
                count = self.counts[tile]
                #for cell, count in zip(self.cells[tile.slicer], self.counts[tile.slicer]):
                neighs.extend(cell[:count])
        neighs = np.unique(np.array(neighs))
        neighs = np.delete(neighs, np.where(neighs == i))
        return neighs

    def _calculate_prior(self, i):
        self.logpriors[i] = 0

        n = self._neighbors(i).astype('int')
        dist = ((self.zscale*(self.pos[i] - self.pos[n]))**2).sum(axis=-1)
        dist0 = (self.rad[i] + self.rad[n])**2

        self.logpriors[i] = np.sum(self.prior_func(dist - dist0))

    def _calculate_all_priors(self):
        for i in xrange(self.rad.shape[0]):
            self._calculate_prior(i)

    def update(self, pos, rad, index):
        self._unbin_particle(index)

        self.pos[index] = pos
        self.rad[index] = rad

        self._bin_particle(index)

        for n in self._neighbors(index):
            self._calculate_prior(n)

    def logprior(self):
        return self.logpriors.sum()
