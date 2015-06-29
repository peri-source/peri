import numpy as np
from itertools import product

from ..util import Tile

class Polynomial3D(object):
    def __init__(self, shape, coeffs=None, order=(1,1,1)):
        self.shape = shape
        self.order = order

        if coeffs is None:
            self.params = np.zeros(np.prod(order), dtype='float')
            self.params[0] = 1
        else:
            self.params = coeffs.astype('float')

        self._setup_rvecs()
        self.tile = Tile(self.shape)

    def _poly_orders(self):
        return product(*(xrange(o) for o in self.order))

    def _setup_rvecs(self):
        o = self.shape
        self.rz, self.ry, self.rx = np.mgrid[0:o[0], 0:o[1], 0:o[2]] / float(max(o))
        self._poly = []

        for i,j,k in self._poly_orders():
            self._poly.append( self.rx**i * self.ry**j * self.rz**k )

        self._poly = np.rollaxis( np.array(self._poly), 0, len(self.shape)+1 )

    def fit_to_data(self, f):
        fit, _, _, _ = np.linalg.lstsq(self._poly.reshape(-1, self.params.shape[0]), f.ravel())
        return fit

    def initialize(self):
        self.update(self.params)

    def set_tile(self, tile):
        self.tile = tile

    def update(self, params):
        self.params = params
        self.bkg = (self._poly * self.params).sum(axis=-1)

    def get_field(self):
        return self.bkg[self.tile.slicer]

    def get_params(self):
        return self.params
