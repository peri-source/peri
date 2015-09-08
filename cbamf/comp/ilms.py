import numpy as np
from numpy.polynomial.polynomial import polyval3d
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
import scipy.optimize as opt
from itertools import product, chain

from cbamf.util import Tile, cdd

#=============================================================================
# Pure 3d functional representations of ILMs
#=============================================================================
class Polynomial3D(object):
    def __init__(self, shape, coeffs=None, order=(1,1,1), partial_update=True, cache=False):
        self.shape = shape
        self.order = order
        self.cache = cache
        self.partial_update = partial_update
        self.nparams = len(list(self._poly_orders()))

        if coeffs is None:
            self.params = np.zeros(self.nparams, dtype='float')
            self.params[0] = 1
        else:
            self.params = coeffs.astype('float')

        self._setup_rvecs()
        self._setup_cache()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))

        self.block = np.ones(self.nparams).astype('bool')
        self.update(self.block, self.params)

    def _poly_orders(self):
        return product(*(xrange(o) for o in self.order))

    def _setup_rvecs(self):
        # normalize all sizes to a strict upper bound on image size
        # so we can transfer ILM between different images
        self.rz, self.ry, self.rx = Tile(self.shape).coords(norm=1024., meshed=False)

    def _setup_cache(self):
        self._last_index = None

        #if not hasattr(self, 'cache'):
        #    self.cache = True

        #if self.cache:
        #    self._poly = []
        #    for index in self._poly_orders():
        #        self._poly.append( self._term(index) )
        #    self._poly = np.rollaxis( np.array(self._poly), 0, len(self.shape)+1 )
        #else:
        self._indices = list(self._poly_orders())

    def from_data(self, f, mask=None, dopriors=False, multiplier=1):
        if self.cache:
            self._from_data_cache(f, mask=mask, dopriors=dopriors, multiplier=multiplier)
        else:
            self._from_data(f, mask=mask, dopriors=dopriors, multiplier=multiplier)

    def _score(self, coeffs, f, mask):
        self.params = coeffs
        test = self._bkg()
        out = (f[mask] - test[mask]).flatten()
        return out

    def _from_data(self, f, mask=None, dopriors=False, multiplier=1):
        if mask is None:
            mask = np.s_[:]
        res = opt.leastsq(self._score, x0=self.params, args=(f, mask))
        self.update(self.block, res[0])

    def _from_data_cache(self, f, mask=None, dopriors=False, multiplier=1):
        if mask is None:
            mask = np.s_[:]
        fit, _, _, _ = np.linalg.lstsq(self._poly[mask].reshape(-1, self.params.shape[0]), f[mask].ravel())
        self.update(self.block, fit)

    def _bkg(self):
        self.bkg = np.zeros(self.shape)

        for order in self._poly_orders():
            ind = self._indices.index(order)
            self.bkg += self.params[ind] * self._term(order)

        return self.bkg

    def _term_ijk(self, index):
        i,j,k = index
        return self.rx**i * self.ry**j * self.rz**k

    def _term(self, index):
        # per index cache, so if called multiple times in a row, keep the answer
        if self._last_index == index:
            return self._last_term

        # otherwise, just calculate this one term
        self._last_index = index
        self._last_term = self._term_ijk(index)
        return self._last_term

    def initialize(self):
        self.update(self.block, self.params)

    def set_tile(self, tile):
        self.tile = tile

    def update(self, blocks, params):
        if self.partial_update and blocks.sum() < self.block.sum()/2:
            for b in np.arange(len(blocks))[blocks]:
                self.bkg -= self.params[b] * self._term(self._indices[b])
                self.params[b] = params[b]
                self.bkg += self.params[b] * self._term(self._indices[b])
        else:
            self.params = params
            self._bkg()

    def get_field(self):
        return self.bkg[self.tile.slicer]

    def get_params(self):
        return self.params

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, ['_poly', 'rx', 'ry', 'rz', 'bkg', '_last_term', '_last_index'])
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
        self.rz, self.ry, self.rx = [np.linspace(-1, 1, i) for i in o]
        self.rz = self.rz[:,None,None]
        self.ry = self.ry[None,:,None]
        self.rx = self.rx[None,None,:]

    def _term_ijk(self, index):
        i,j,k = index
        ci = np.zeros(i+1)
        cj = np.zeros(j+1)
        ck = np.zeros(k+1)
        ci[-1] = 1
        cj[-1] = 1
        ck[-1] = 1
        return legval(self.rx, ci) * legval(self.ry, cj) * legval(self.rz, ck)

#=============================================================================
# 2+1d functional representations of ILMs, p(x,y)+q(z)
#=============================================================================
class Polynomial2P1D(object):
    def __init__(self, shape, order=(1,1,1)):
        self.shape = shape
        self.xyorder = order[:2]
        self.zorder = order[-1]

        self.order = order
        self.nparams = len(list(self._poly_orders()))

        self.params = np.zeros(self.nparams, dtype='float')
        self.params[0] = 1
        self.params[len(list(self._poly_orders_xy()))] = 1

        self._setup()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))

        self.block = np.ones(self.nparams).astype('bool')
        self.initialize()

    def _poly_orders_xy(self):
        return product(*(xrange(o) for o in self.order[:2]))

    def _poly_orders_z(self):
        return product(xrange(self.order[-1]))

    def _poly_orders(self):
        return chain(self._poly_orders_xy(), self._poly_orders_z())

    def _setup_rvecs(self):
        # normalize all sizes to a strict upper bound on image size
        # so we can transfer ILM between different images
        self.rz, self.ry, self.rx = Tile(self.shape).coords(norm=1024., meshed=False)

    def _setup(self):
        self._setup_rvecs()
        self._last_index = None
        self._indices = list(self._poly_orders())
        self._indices_xy = list(self._poly_orders_xy())
        self._indices_z = list(self._poly_orders_z())

    def _bkg(self):
        self.bkg = np.zeros(self.shape)
        self._polyxy = 0*self.bkg
        self._polyz = 0*self.bkg

        for order in self._poly_orders_xy():
            ind = self._indices.index(order)
            self._polyxy += self.params[ind] * self._term(order)

        for order in self._poly_orders_z():
            ind = self._indices.index(order)
            self._polyz += self.params[ind] * self._term(order)

        self.bkg = self._polyxy * self._polyz
        return self.bkg

    def from_data(self, f, mask=None, dopriors=False, multiplier=1):
        if mask is None:
            mask = np.s_[:]
        res = opt.leastsq(self._score, x0=self.params, args=(f, mask))
        self.update(self.block, res[0])

    def _score(self, coeffs, f, mask):
        self.params = coeffs
        test = self._bkg()
        out = (f[mask] - test[mask]).flatten()
        #print np.abs(out).mean()
        return out

    def _term_xy(self, index):
        i,j = index
        return self.rx**i * self.ry**j

    def _term_z(self, index):
        k = index[0]
        return self.rz**k

    def _term(self, index):
        # per index cache, so if called multiple times in a row, keep the answer
        if self._last_index == index:
            return self._last_term

        # otherwise, just calculate this one term
        self._last_index = index
        if len(index) == 2:
            self._last_term = self._term_xy(index)
        if len(index) == 1:
            self._last_term = self._term_z(index)
        return self._last_term

    def initialize(self):
        self.update(self.block, self.params)

    def set_tile(self, tile):
        self.tile = tile

    def update(self, blocks, params):
        if blocks.sum() < self.block.sum()/2:
            for b in np.arange(len(blocks))[blocks]:
                order = self._indices[b]

                if order in self._indices_xy:
                    _term = self._polyxy
                else:
                    _term = self._polyz

                _term -= self.params[b] * self._term(order)
                self.params[b] = params[b]
                _term += self.params[b] * self._term(order)
                self.bkg = self._polyxy * self._polyz
        else:
            self.params = params
            self._bkg()

    def get_field(self):
        return self.bkg[self.tile.slicer]

    def get_params(self):
        return self.params

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, ['_poly', 'rx', 'ry', 'rz', 'bkg', '_last_term', '_last_index'])
        cdd(odict, ['_polyxy', '_polyz'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self._setup()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))
        self.initialize()

class LegendrePoly2P1D(Polynomial2P1D):
    def __init__(self, shape, coeffs=None, order=(1,1,1), *args, **kwargs):
        super(LegendrePoly2P1D, self).__init__(*args, shape=shape, order=order, **kwargs)

    def _setup_rvecs(self):
        o = self.shape
        self.rz, self.ry, self.rx = [np.linspace(-1, 1, i) for i in o]
        self.rz = self.rz[:,None,None]
        self.ry = self.ry[None,:,None]
        self.rx = self.rx[None,None,:]

    def _term_xy(self, index):
        i,j = index
        ci = np.zeros(i+1)
        cj = np.zeros(j+1)
        ci[-1], cj[-1] = 1, 1
        return legval(self.rx, ci) * legval(self.ry, cj)

    def _term_z(self, index):
        k = index[0]
        ck = np.zeros(k+1)
        ck[-1] = 1
        return legval(self.rz, ck)

class ChebyshevPoly2P1D(Polynomial2P1D):
    def __init__(self, shape, coeffs=None, order=(1,1,1), *args, **kwargs):
        super(ChebyshevPoly2P1D, self).__init__(*args,
                shape=shape, order=order, **kwargs)

    def _term_xy(self, index):
        i,j = index
        ci = np.zeros(i+1)
        cj = np.zeros(j+1)
        ci[-1], cj[-1] = 1, 1
        return chebval(self.rx, ci) * chebval(self.ry, cj)

    def _term_z(self, index):
        k = index[0]
        ck = np.zeros(k+1)
        ck[-1] = 1
        return chebval(self.rz, ck)

#=============================================================================
# a complex hidden variable representation of the ILM
# something like (p(x,y)+m(x,y))+q(z) where m is determined by local models
#=============================================================================
class PiecewisePolyStreak2P1D(object):
    def __init__(self, shape, order=(1,1,1), num=30):
        self.shape = shape
        self.xyorder = order[:2]
        self.zorder = order[-1]

        self.order = order
        self.nparams = len(list(self._poly_orders()))

        self.params = np.zeros(self.nparams, dtype='float')
        self.params[0] = 1
        self.params[len(list(self._poly_orders_xy()))] = 1

        self._setup()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))

        self.block = np.ones(self.nparams).astype('bool')
        self.initialize()

    def _poly_orders_xy(self):
        return product(*(xrange(o) for o in self.order[:2]))

    def _poly_orders_z(self):
        return product(xrange(self.order[-1]))

    def _poly_orders(self):
        return chain(self._poly_orders_xy(), self._poly_orders_z())

    def _setup(self):
        # normalize all sizes to a strict upper bound on image size
        # so we can transfer ILM between different images
        self.rz, self.ry, self.rx = Tile(self.shape).coords(norm=1024., meshed=False)

        self._last_index = None
        self._indices = list(self._poly_orders())
        self._indices_xy = list(self._poly_orders_xy())
        self._indices_z = list(self._poly_orders_z())

    def _bkg(self):
        self.bkg = np.zeros(self.shape)
        self._polyxy = 0*self.bkg
        self._polyz = 0*self.bkg

        for order in self._poly_orders_xy():
            ind = self._indices.index(order)
            self._polyxy += self.params[ind] * self._term(order)

        for order in self._poly_orders_z():
            ind = self._indices.index(order)
            self._polyz += self.params[ind] * self._term(order)

        self.bkg = self._polyxy * self._polyz
        return self.bkg

    def from_data(self, f, mask=None, dopriors=False, multiplier=1):
        if mask is None:
            mask = np.s_[:]
        res = opt.leastsq(self._score, x0=self.params, args=(f, mask))
        self.update(self.block, res[0])

    def _score(self, coeffs, f, mask):
        self.params = coeffs
        test = self._bkg()
        out = (f[mask] - test[mask]).flatten()
        #print np.abs(out).mean()
        return out

    def _term_xy(self, index):
        i,j = index
        return self.rx**i * self.ry**j

    def _term_z(self, index):
        k = index[0]
        return self.rz**k

    def _term(self, index):
        # per index cache, so if called multiple times in a row, keep the answer
        if self._last_index == index:
            return self._last_term

        # otherwise, just calculate this one term
        self._last_index = index
        if len(index) == 2:
            self._last_term = self._term_xy(index)
        if len(index) == 1:
            self._last_term = self._term_z(index)
        return self._last_term

    def initialize(self):
        self.update(self.block, self.params)

    def set_tile(self, tile):
        self.tile = tile

    def update(self, blocks, params):
        if blocks.sum() < self.block.sum()/2:
            for b in np.arange(len(blocks))[blocks]:
                order = self._indices[b]

                if order in self._indices_xy:
                    _term = self._polyxy
                else:
                    _term = self._polyz

                _term -= self.params[b] * self._term(order)
                self.params[b] = params[b]
                _term += self.params[b] * self._term(order)
                self.bkg = self._polyxy * self._polyz
        else:
            self.params = params
            self._bkg()

    def get_field(self):
        return self.bkg[self.tile.slicer]

    def get_params(self):
        return self.params

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, ['_poly', 'rx', 'ry', 'rz', 'bkg', '_last_term', '_last_index'])
        cdd(odict, ['_polyxy', '_polyz'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self._setup()
        self.tile = Tile(self.shape)
        self.set_tile(Tile(self.shape))
        self.initialize()
