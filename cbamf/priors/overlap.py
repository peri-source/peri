import numpy as np
import scipy as sp

class HardSphereOverlap(object):
    def __init__(self, pos, rad, bounds, cutoff, maxn=30):
        self.bounds = bounds
        self.cutoff = cutoff
        self.pos = pos
        self.rad = rad
        self.maxn = maxn
        self.hasoverlaps = False

        self.bl, self.br = np.array(bounds[0]), np.array(bounds[1])
        self.bdiff = self.br - self.bl
        self._initialize()
        self._calculate_all_overlaps()

    def _initialize(self):
        self.size = (self.bdiff / self.cutoff).astype('int')
        self.size[self.size == 0] = 1

        self.cells = np.zeros(self.size + (self.maxn,))
        self.counts = np.zeros(self.size)

        for i, p in enumerate(self.pos):
            self._bin_particle(p, i)

    def _pos_to_inds(self, pos):
        # TODO - add to multiple bins if bigger than cutoff
        ind = (self.size * (pos - self.bl) / self.bdiff).astype('int')
        return ind

    def _bin_particle(self, pos, index):
        for ind in self._pos_to_inds(pos):
            self.cells[ind] = index
            self.countts[ind] += 1

    def _calculate_all_overlaps(self):
        pass

    def update(self, pos, rad, index):
        pass

    def logprior(self):
        pass
