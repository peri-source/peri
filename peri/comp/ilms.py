import numpy as np
from numpy.polynomial.polynomial import polyval3d
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
import scipy.optimize as opt

from operator import add, mul
from itertools import product, chain

from peri import util
from peri.comp import Component
from peri.interpolation import BarnesInterpolation1D

#=============================================================================
# Pure 3d functional representations of ILMs
#=============================================================================
class Polynomial3D(Component):
    category = 'ilm'

    def __init__(self, shape=None, order=(1,1,1), tileinfo=None, constval=None):
        """
        A polynomial 3D class for updating large fields of polys.

        Parameters:
        -----------
        shape : tuple
            shape of the field (z,y,x)

        order : tuple
            number of terms in each direction

        tileinfo : tuple of 2 `peri.util.Tile`
            These objects help in the transfer of fields from different
            sections of the same image to new fields. `tileinfo` is a tuple
            containing the Tile representing the entire image as well as the
            Tile representing this particular section of field. (typically
            given by `peri.rawimage.tile`)

        constval : float
            The initial value of the entire field, if a constant.
        """
        self.shape = shape
        self.order = order
        self.tileinfo = tileinfo

        # set up the parameter mappings and values
        params, values = [], []
        self.param_term = {}
        for order in product(*(xrange(o) for o in self.order)):
            p = 'ilm-%i-%i-%i' % order
            self.param_term[p] = order

            params.append(p)
            values.append(0.0)

        if constval:
            values[0] = constval

        super(Polynomial3D, self).__init__(params=params, values=values)

        if self.shape:
            self.initialize()

    def initialize(self):
        self.r = self.rvecs()
        self.set_tile(self.shape)
        self.field = np.zeros(self.shape.shape)
        self.update(self.params, self.values)

    def rvecs(self):
        # normalize all sizes to a strict upper bound on image size
        # so we can transfer ILM between different images
        if self.tileinfo:
            img, inner = self.tileinfo
            vecs = img.coords(norm=img.shape)
            vecs = [v[inner.slicer] for v in vecs]
        else:
            vecs = self.shape.coords(norm=self.shape.shape)
        return vecs

    def term_ijk(self, index):
        i,j,k = index
        return self.r[0]**i * self.r[1]**j * self.r[2]**k

    def term(self, index):
        if self.__dict__.get('_last_index') and index == self._last_index:
            return self._last_term
        else:
            term = self.term_ijk(index)
            self._last_term = term
            self._last_index = index
            return self._last_term

    def set_tile(self, tile):
        self.tile = tile

    def update(self, params, values):
        params = util.listify(params)
        values = util.listify(values)

        if len(params) < len(self.params)/2:
            for p,v1 in zip(params, values):
                v0 = self.get_values(p)

                self.field -= v0 * self.term(self.param_term[p])
                self.set_values(p, v1)
                self.field += v1 * self.term(self.param_term[p])
        else:
            self.field = np.zeros(self.shape.shape)
            for p,v in zip(self.params, self.values):
                self.field += v * self.term(self.param_term[p])

    def get_field(self):
        return self.field[self.tile.slicer]

    def get_params(self):
        return self.params

    def get_update_tile(self, params, values):
        return util.Tile(self.shape.shape)

    def __str__(self):
        return "{} [{}]".format(
            self.__class__.__name__, self.order
        )

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, ['r', 'field', '_last_term', '_last_index'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.initialize()

class LegendrePoly3D(Polynomial3D):
    def __init__(self, shape=None, *args, **kwargs):
        """ Same arguments are Polynomial3D """
        super(LegendrePoly3D, self).__init__(shape=shape, *args, **kwargs)

    def rvecs(self):
        vecs = super(LegendrePoly3D, self).rvecs()
        vecs = [2*v - 1 for v in vecs]
        return vecs

    def term_ijk(self, index):
        i,j,k = index
        ci = np.zeros(i+1)
        cj = np.zeros(j+1)
        ck = np.zeros(k+1)
        ci[-1] = cj[-1] = ck[-1] = 1
        return legval(self.r[0], ci) * legval(self.r[1], cj) * legval(self.r[2], ck)

#=============================================================================
# 2+1d functional representations of ILMs, p(x,y)+q(z)
#=============================================================================
class Polynomial2P1D(Polynomial3D):
    def __init__(self, shape=None, order=(1,1,1), tileinfo=None, constval=None,
            operation='*'):
        """
        A polynomial 2+1D class for updating large fields of polys.  The form
        of these polynomials if P(x,y) () Q(z), separated in the z-direction.

        Parameters:
        -----------
        shape : tuple
            shape of the field (z,y,x)

        order : tuple
            number of terms in each direction

        tileinfo : tuple of 2 `peri.util.Tile`
            These objects help in the transfer of fields from different
            sections of the same image to new fields. `tileinfo` is a tuple
            containing the Tile representing the entire image as well as the
            Tile representing this particular section of field. (typically
            given by `peri.rawimage.tile`)

        constval : float
            The initial value of the entire field, if a constant.

        operation : string
            Type of joining operation between the (x,y) and (z) poly. Can be
            either '*' or '+'
        """

        self.shape = shape
        self.operation = operation
        self.order = order
        self.tileinfo = tileinfo

        # set up the parameter mappings and values
        params, values = [], []
        self.xy_param = {}
        self.z_param = {}

        for order in product(*(xrange(o) for o in self.order[1::-1])):
            p = 'ilm-xy-%i-%i' % order
            self.xy_param[p] = order

            params.append(p)
            values.append(0.0)

        for order in xrange(self.order[0]):
            p = 'ilm-z-%i' % order
            self.z_param[p] = (order,)

            params.append(p)
            values.append(0.0)

        # setup the basics of the component now
        Component.__init__(self, params, values)

        # set up the appropriate zero terms for the supplied constant value
        # parameter if there.
        if constval:
            if operation == '*':
                self.set_values('ilm-xy-0-0', constval)
                self.set_values('ilm-z-0', 1.0)
            else:
                self.set_values('ilm-xy-0-0', constval)

        if self.shape:
            self.initialize()

    def initialize(self):
        self.r = self.rvecs()
        self.field_xy = 0*self.term_ijk((0,0))
        self.field_z = 0*self.term_ijk((0,))
        super(Polynomial2P1D, self).initialize()

    def calc_field(self):
        self.field_xy = 0*self.term_ijk((0,0))
        self.field_z = 0*self.term_ijk((0,))

        for p,v in zip(self.params, self.values):
            if p in self.xy_param:
                order = self.xy_param[p]
                term = self.field_xy
            else:
                order = self.z_param[p]
                term = self.field_z

            term += v * self.term(order)

        op = {'*': mul, '+': add}[self.operation]
        self.field = op(self.field_xy, self.field_z)
        return self.field

    def term_ijk(self, index):
        if len(index) == 2:
            i,j = index
            return self.r[2]**i * self.r[1]**j

        elif len(index) == 1:
            k = index[0]
            return self.r[0]**k

    def update(self, params, values):
        params = util.listify(params)
        values = util.listify(values)

        if len(params) < len(self.params)/2:
            for p,v1 in zip(params, values):
                if p in self.xy_param:
                    order = self.xy_param[p]
                    term = self.field_xy
                else:
                    order = self.z_param[p]
                    term = self.field_z

                v0 = self.get_values(p)
                term -= v0 * self.term(order)
                self.set_values(p,v1)
                term += v1 * self.term(order)

            op = {'*': mul, '+': add}[self.operation]
            self.field = op(self.field_xy, self.field_z)
        else:
            self.set_values(params, values)
            self.field = self.calc_field()

    def get_field(self):
        return self.field[self.tile.slicer]

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, ['r', 'field', 'field_xy', 'field_z'])
        util.cdd(odict, ['_last_term', '_last_index'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.initialize()

class LegendrePoly2P1D(Polynomial2P1D):
    def __init__(self, shape=None, order=(1,1,1), **kwargs):
        super(LegendrePoly2P1D, self).__init__(shape=shape, order=order, **kwargs)

    def rvecs(self):
        vecs = super(LegendrePoly2P1D, self).rvecs()
        vecs = [2*v - 1 for v in vecs]
        return vecs

    def term_ijk(self, index):
        if len(index) == 2:
            i,j = index
            ci = np.diag(np.ones(i+1))[i]
            cj = np.diag(np.ones(j+1))[j]
            return legval(self.r[2], ci) * legval(self.r[1], cj)

        elif len(index) == 1:
            k = index[0]
            ck = np.diag(np.ones(k+1))[k]
            return legval(self.r[0], ck)

class ChebyshevPoly2P1D(Polynomial2P1D):
    def __init__(self, shape=None, order=(1,1,1), **kwargs):
        super(ChebyshevPoly2P1D, self).__init__(shape=shape, order=order, **kwargs)

    def term_ijk(self, index):
        if len(index) == 2:
            i,j = index
            ci = np.diag(np.ones(i+1))[i]
            cj = np.diag(np.ones(j+1))[j]
            return chebval(self.r[2], ci) * chebval(self.r[1], cj)

        elif len(index) == 1:
            k = index[0]
            ck = np.diag(np.ones(k+1))[k]
            return chebval(self.r[0], ck)

#=============================================================================
# a complex hidden variable representation of the ILM
# something like (p(x,y)+m(x,y))*q(z) where m is determined by local models
#=============================================================================
class BarnesStreakLegPoly2P1D(Component):
    category = 'ilm'

    def __init__(self, shape=None, npts=(40,20), zorder=7, op='*', barnes_dist=1.75):
        """
        A Barnes interpolant. This one is of the form

            I = (1 + (\sum b_k(x) (o) L_k(y))) * (1 + q(z)) + c

        where b_k are independent barnes interpolants and L_k are legendre
        polynomials. q is a polynomial strictly in z. Additionally, the
        operation (o) is settable.

        Parameters:
        -----------
        shape : iterable
            size of the field in pixels, needs to be padded shape

        npts : int
            Number of control points used for the Barnes interpolant b_k
            in the x-y sum.

        zorder : integer
            Number of orders for the z-polynomial.

        op : string
            The operation to perform between Barnes and LegPoly, '*' or '+'.

        barnes_dist : float
            Fractional distance to use for the barnes interpolator
        """
        self.shape = shape
        self.barnes_dist = barnes_dist
        self.zorder = zorder
        self.npts = npts
        self.op = op

        # set up the various parameter mappings and out local cache of how to
        # distinguish them quickly from one another
        params, values = ['ilm-scale', 'ilm-off'], [1.0, 0.0]
        self.barnes_params = []
        self.poly_params = {}
        for i, npt in enumerate(npts):
            tparams = ['ilm-b%i-%i' % (i, j) for j in xrange(npt)]
            tvalues = [0.0]*len(tparams)

            params.extend(tparams)
            values.extend(tvalues)

            self.barnes_params.append(tparams)

        # tack on the z-poly parameters on the end
        self.poly_params = {'ilm-z-%i':i for i in xrange(zorder)}
        params.extend(self.poly_params)
        values.extend([0.0]*len(self.poly_params))

        super(BarnesStreakLegPoly2P1D, self).__init__(params=params, values=values)

        if self.shape:
            self.initialize()

    def _setup_rvecs(self):
        o = self.shape.shape
        self.r = [np.linspace(-1, 1, i) for i in o]
        self.r[0] = self.r[0][:,None,None]
        self.r[1] = self.r[1][None,:,None]
        self.r[2] = self.r[2][None,None,:]

        self.b_out = np.squeeze(self.r[2])
        self.b_in = [
            np.linspace(self.b_out.min(), self.b_out.max(), q)
            for q in self.npts
        ]

    def _barnes_poly(self, n=0):
        weights = np.diag(np.ones(n+1))[n]
        return legval(np.squeeze(self.r[1]), weights)[:,None]

    def _barnes(self, y, n=0):
        b_in = self.b_in[n]
        fdst = (b_in[1] - b_in[0])*1.0/self.barnes_dist
        coeffs = self.get_values(self.barnes_params[n])

        b = BarnesInterpolation1D(
            b_in, coeffs, filter_size=fdst, damp=0.9, iterations=3,
            clip=True, clipsize=3
        )
        return b(y)

    def _barnes_val(self, n=0):
        return self._barnes(self.b_out, n=n)[None,:]

    def _barnes_full(self):
        barnes = np.array([
            self._barnes_val(i)*self._barnes_poly(i)
            for i in xrange(len(self.npts))
        ])
        return barnes.sum(axis=0)[None,:,:]

    def _term(self, index):
        weights = np.diag(np.ones(index+1))[index]
        return legval(self.r[0], weights)

    @property
    def off(self):
        return self.get_values('ilm-off')

    @property
    def scale(self):
        return self.get_values('ilm-scale')

    def calc_field(self):
        op = {'*': mul, '+': add}[self.op]
        return self.scale * op(1 + self._barnes_full(), 1 + self.poly) + self.off

    def calc_poly(self):
        return np.sum([
            self.get_values(p) * self._term(i)
            for p, i in self.poly_params.iteritems()
        ], axis=0)

    def initialize(self):
        self._setup_rvecs()
        self.set_tile(self.shape)

        self.poly = self.calc_poly()
        self.field = self.calc_field()

    def set_tile(self, tile):
        self.tile = tile

    def update(self, params, values):
        op = {'*': mul, '+': add}[self.op]

        params = util.listify(params)
        values = util.listify(values)

        if len(params) < len(self.params)/2:
            for p,v1 in zip(params, values):
                if p in self.poly_params:
                    tm = self._term(self.poly_params[p])
                    v0 = self.get_values(p)

                    self.poly += (v1-v0) * tm

            self.set_values(params, values)
            self.field = self.calc_field()
        else:
            self.set_values(params, values)
            self.field = self.calc_field()

    def get_field(self):
        return self.field[self.tile.slicer]

    def get_update_tile(self, params, values):
        params = util.listify(params)
        values = util.listify(values)

        orig_values = self.get_values(params)

        tiles = []
        for p,v in zip(params, values):
            # global parameters get global tiles
            if p in self.poly_params or p == 'ilm-scale' or p == 'ilm-off':
                return self.shape.copy()

            # figure out hte barnes local update size
            for n, grp in enumerate(self.barnes_params):
                if not p in grp:
                    continue

                val0 = self._barnes(self.b_out, n=n)
                self.set_values(p, v)
                val1 = self._barnes(self.b_out, n=n)

                inds = np.arange(self.b_out.shape[0])
                inds = inds[(val1 - val0) > 1e-12]
                if len(inds) <= 1:
                    continue

                l, r = inds.min(), inds.max()

                tile = self.shape.copy()
                tile.l[2] = l
                tile.r[2] = r
                if tile.shape[2] == 0:
                    tile.r[2] = tile.r[2] + 1

                tiles.append(util.Tile(tile.l, tile.r))

        self.set_values(params, orig_values)
        if len(tiles) == 0:
            return util.Tile(0)
        return util.Tile.boundingtile(tiles)

    def randomize_parameters(self, ptp=0.2, fourier=False, vmin=None, vmax=None):
        """
        Create random parameters for this ILM that mimic experiments
        as closely as possible without real assumptions.
        """
        if vmin is not None and vmax is not None:
            ptp = vmax - vmin
            print "Warning: vmin and vmax set, using those for ptp"
        elif vmax is not None and vmin is None:
            vmin = vmax - ptp
        elif vmin is not None and vmax is None:
            vmax = vmin + ptp
        else:
            vmax = 1.0
            vmin = vmax - ptp

        self.update('ilm-scale', 1.0)
        self.update('ilm-off', 0.0)

        for k, v in self.poly_params.iteritems():
            norm = (self.zorder + 1.0)*2
            self.set_values(k, ptp*(np.random.rand() - 0.5) / norm)

        for i, p in enumerate(self.barnes_params):
            N = len(p)
            if fourier:
                t = ((np.random.rand(N)-0.5) + 1.j*(np.random.rand(N)-0.5))/(np.arange(N)+1)
                q = np.real(np.fft.ifftn(t)) / (i+1)
            else:
                t = ptp*np.sqrt(N)*(np.random.rand(N)-0.5)
                q = np.cumsum(t) / (i+1)

            q = ptp * q / q.ptp() / len(self.barnes_params)
            q -= q.mean()
            self.set_values(p, q)

        self.initialize()

        ptp0 = self.get_field().ptp()
        scale = ptp / ptp0
        self.update('ilm-scale', scale)

        min0 = self.get_field().min()
        off = vmin - min0
        self.update('ilm-off', off)

    def __str__(self):
        return "{} [{} {}]".format(
            self.__class__.__name__, self.npts, self.zorder
        )

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, ['poly', 'b_in', 'b_out', 'r', 'field'])
        util.cdd(odict, ['_last_term', '_last_index'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.initialize()


