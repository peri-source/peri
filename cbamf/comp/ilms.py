import numpy as np
from numpy.polynomial.polynomial import polyval3d
from numpy.polynomial.legendre import legval
import scipy.optimize as opt
from itertools import product

from cbamf.util import Tile, cdd

class Polynomial3D(object):
    def __init__(self, shape, coeffs=None, order=(1,1,1), partial_update=True, cache=False):
        self.shape = shape
        self.order = order
        self.cache = cache
        self.partial_update = partial_update

        if coeffs is None:
            self.params = np.zeros(np.prod(order), dtype='float')
            self.params[0] = 1
        else:
            self.params = coeffs.astype('float')

        self._setup_rvecs()
        self._setup_cache()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))

        self.block = np.ones(self.params.shape).astype('bool')
        self.update(self.block, self.params)

    def _poly_orders(self):
        return product(*(xrange(o) for o in self.order))

    def _setup_rvecs(self):
        # normalize all sizes to a strict upper bound on image size
        # so we can transfer ILM between different images
        self.rz, self.ry, self.rx = Tile(self.shape).coords(norm=1024.)

    def _setup_cache(self):
        if self.cache:
            self._poly = []
            for index in self._poly_orders():
                self._poly.append( self._term(index) )
            self._poly = np.rollaxis( np.array(self._poly), 0, len(self.shape)+1 )
        else:
            self._indices = list(self._poly_orders())

    def from_data(self, f, mask=None, dopriors=False, multiplier=1):
        if self.cache:
            self._from_data_cache(f, mask=mask, dopriors=dopriors, multiplier=multiplier)
        else:
            self._from_data(f, mask=mask, dopriors=dopriors, multiplier=multiplier)

    def _score(self, coeffs, f, mask):
        test = np.zeros(f.shape)
        for i, c in enumerate(coeffs):
            self.bkg += c * self._term(self._indices[i])
        return f[mask] - test[mask]

    def _from_data(self, f, mask=None, dopriors=False, multiplier=1):
        # TODO -- add priors to the fit
        if mask is None:
            mask = np.s_[:]
        res = opt.leastsq(self._score, x0=self.params, args=(f, mask))
        self.update(self.block, res[0])

    def _from_data_cache(self, f, mask=None, dopriors=False, multiplier=1):
        # TODO -- add priors to the fit
        if mask is None:
            mask = np.s_[:]
        fit, _, _, _ = np.linalg.lstsq(self._poly[mask].reshape(-1, self.params.shape[0]), f[mask].ravel())
        self.update(self.block, fit)

    def _term(self, index):
        # TODO -- per index cache, so if called multiple times in a row, keep the answer
        i,j,k = index
        return self.rx**i * self.ry**j * self.rz**k

    def initialize(self):
        self.update(self.block, self.params)

    def set_tile(self, tile):
        self.tile = tile

    def update(self, blocks, params):
        if self.cache:
            if self.partial_update and blocks.sum() < self.block.sum()/2:
                self.bkg -= (self._poly[...,blocks] * self.params[blocks]).sum(axis=-1)
                self.params = params
                self.bkg += (self._poly[...,blocks] * self.params[blocks]).sum(axis=-1)
            else:
                self.params = params
                self.bkg = (self._poly * self.params).sum(axis=-1)
        else:
            if self.partial_update and blocks.sum() < self.block.sum()/2:
                for b in np.arange(len(blocks))[blocks]:
                    self.bkg -= self.params[b] * self._term(self._indices[b])
                    self.params[b] = params[b]
                    self.bkg += self.params[b] * self._term(self._indices[b])
            else:
                self.params = params
                self.bkg = np.zeros(self.shape)
                for b in np.arange(len(blocks))[blocks]:
                    self.bkg += self.params[b] * self._term(self._indices[b])

    def get_field(self):
        return self.bkg[self.tile.slicer]

    def get_params(self):
        return self.params

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, ['_poly', 'rx', 'ry', 'rz', 'bkg'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self._setup_rvecs()
        self._setup_cache()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))
        self.update(self.block, self.params)

class LegendrePoly3D(Polynomial3D):
    def __init__(self, shape, coeffs=None, order=(1,1,1), *args, **kwargs):
        super(LegendrePoly3D, self).__init__(*args, shape=shape, coeffs=coeffs, order=order, **kwargs)

    def _setup_rvecs(self):
        o = self.shape
        self.rz, self.ry, self.rx = np.meshgrid(*[np.linspace(-1, 1, i) for i in o], indexing='ij')

    def _term(self, index):
        i,j,k = index
        ci = np.zeros(i+1)
        cj = np.zeros(j+1)
        ck = np.zeros(k+1)
        ci[-1] = 1
        cj[-1] = 1
        ck[-1] = 1
        return legval(self.rx, ci) * legval(self.ry, cj) * legval(self.rz, ck)
