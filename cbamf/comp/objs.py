import numpy as np
from ..util import Tile

class SphereCollectionRealSpace(object):
    def __init__(self, pos, rad, shape, boundary_size=3):
        self.boundary_size = boundary_size
        self.pos = pos.astype('float')
        self.rad = rad.astype('float')
        self.N = rad.shape[0]

        self.shape = shape
        self._setup()

    def _setup(self):
        z,y,x = np.meshgrid(*(xrange(i) for i in self.shape), indexing='ij')
        self.rvecs = np.rollaxis(np.array(np.broadcast_arrays(z,y,x)), 0, 4)
        self.particles = np.zeros(self.shape)

    def _particle(self, pos, rad, zscale, sign=1):
        p = np.round(pos)
        r = np.round(np.array([1.0/zscale,1,1])*np.ceil(rad)+self.boundary_size)

        tile = Tile(p-r, p+r, 0, self.shape)
        subr = self.rvecs[tile.slicer + (np.s_[:],)]
        rvec = (subr - pos)

        # apply the zscale and find the distances to make a ellipsoid
        # note: the appearance of PI in the last line is because of leastsq
        # fits to the correct Fourier version of the sphere, j_{3/2} / r^{3/2}
        # happened to fit right at pi -- what?!
        rvec[...,0] *= zscale
        rdist = np.sqrt((rvec**2).sum(axis=-1))
        self.particles[tile.slicer] += sign/(1.0 + np.exp(np.pi*(rdist - rad)))

    def _update_particle(self, n, p, r, zscale):
        self._particle(self.pos[n], self.rad[n], zscale, -1)

        self.pos[n] = p
        self.rad[n] = r

        self._particle(self.pos[n], self.rad[n], zscale, +1)

    def initialize(self, zscale):
        if len(self.pos.shape) != 2:
            raise AttributeError("Position array needs to be (-1,3) shaped, (z,y,x) order")

        self.particles = np.zeros(self.shape)
        for p0, r0 in zip(self.pos, self.rad):
            self._particle(p0, r0, zscale)

    def set_tile(self, tile):
        self.tile = tile

    def update(self, ns, pos, rad, zscale):
        for n, p, r in zip(ns, pos, rad):
            self._update_particle(n, p, r, zscale)

    def get_field(self):
        return self.particles[self.tile.slicer]

    def get_params(self):
        return np.hstack([self.pos.ravel(), self.rad])

    def get_params_pos(self):
        return self.pos.ravel()

    def get_params_rad(self):
        return self.rad
