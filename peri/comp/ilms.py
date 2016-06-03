import numpy as np
from numpy.polynomial.polynomial import polyval3d
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
import scipy.optimize as opt

from collections import OrderedDict
from operator import add, mul
from itertools import product, chain

from peri import util
from peri.comp import Component
from peri.interpolation import BarnesInterpolation1D

#=============================================================================
# Pure 3d functional representations of ILMs
#=============================================================================
class Polynomial3D(Component):
    def __init__(self, order=(1,1,1), tileinfo=None, constval=None,
            category='ilm', shape=None):
        """
        A polynomial 3D class for updating large fields of polys.

        Parameters:
        -----------
        shape : `peri.util.Tile`
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
        self.category = category
        c = category

        # set up the parameter mappings and values
        params, values = [], []
        self.param_term = {}
        for order in product(*(xrange(o) for o in self.order)):
            p = c+'-%i-%i-%i' % order
            self.param_term[p] = order

            params.append(p)
            values.append(0.0)

        if constval:
            values[0] = constval

        super(Polynomial3D, self).__init__(
            params=params, values=values, category=category
        )

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
                tm = self.param_term[p]

                self.field -= v0 * self.term(tm)
                self.set_values(p, v1)
                self.field += v1 * self.term(tm)
        else:
            self.set_values(params, values)
            self.field = np.zeros(self.shape.shape)
            for p,v in zip(self.params, self.values):
                self.field += v * self.term(self.param_term[p])

    def get(self):
        return self.field[self.tile.slicer]

    def get_params(self):
        return self.params

    def get_update_tile(self, params, values):
        return self.shape.copy()

    def nopickle(self):
        return super(Polynomial3D, self).nopickle() + [
            'r', 'field', '_last_term', '_last_index'
        ]

    def __str__(self):
        return "{} [{}]".format(
            self.__class__.__name__, self.order
        )

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, self.nopickle())
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        if self.shape:
            self.initialize()

class LegendrePoly3D(Polynomial3D):
    def __init__(self, *args, **kwargs):
        """ Same arguments are Polynomial3D """
        super(LegendrePoly3D, self).__init__(*args, **kwargs)

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
    def __init__(self, order=(1,1,1), tileinfo=None, constval=None,
            operation='*', category='ilm', shape=None):
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
        self.category = category
        c = self.category

        # set up the parameter mappings and values
        params, values = [], []
        self.xy_param = {}
        self.z_param = {}

        for order in product(*(xrange(o) for o in self.order[1:][::-1])):
            p = c+'-xy-%i-%i' % order
            self.xy_param[p] = order

            params.append(p)
            values.append(0.0)

        for order in xrange(self.order[0]):
            p = c+'-z-%i' % order
            self.z_param[p] = (order,)

            params.append(p)
            values.append(0.0)

        # setup the basics of the component now
        Component.__init__(self, params, values, category=category)

        # set up the appropriate zero terms for the supplied constant value
        # parameter if there.
        if constval:
            if operation == '*':
                self.set_values(c+'-xy-0-0', constval)
                self.set_values(c+'-z-0', 1.0)
            else:
                self.set_values(c+'-xy-0-0', constval)

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

    def get(self):
        return self.field[self.tile.slicer]

    def nopickle(self):
        return super(Polynomial2P1D, self).nopickle() + [
            'r', 'field', 'field_xy', 'field_z',
            '_last_term', '_last_index'
        ]

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, self.nopickle())
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        if self.shape:
            self.initialize()

class LegendrePoly2P1D(Polynomial2P1D):
    def __init__(self, order=(1,1,1), **kwargs):
        super(LegendrePoly2P1D, self).__init__(order=order, **kwargs)

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
    def __init__(self, order=(1,1,1), **kwargs):
        super(ChebyshevPoly2P1D, self).__init__(order=order, **kwargs)

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

    def __init__(self, npts=(40,20), zorder=7, op='*', barnes_dist=1.75,
            barnes_clip_size=3, local_updates=True, category='ilm', shape=None):
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

        local_updates : boolean
            Whether to perform local updates on the ILM
        """
        self.shape = shape
        self.local_updates = local_updates
        self.barnes_clip_size = barnes_clip_size
        self.barnes_dist = barnes_dist
        self.category = category
        self.zorder = zorder
        self.npts = npts
        self.op = op

        c = self.category
        # set up the various parameter mappings and out local cache of how to
        # distinguish them quickly from one another
        params, values = [c+'-scale', c+'-off'], [1.0, 0.0]
        self.barnes_params = []
        for i, npt in enumerate(npts):
            tparams = [c+'-b%i-%i' % (i, j) for j in xrange(npt)]
            tvalues = [0.0]*len(tparams)

            params.extend(tparams)
            values.extend(tvalues)

            self.barnes_params.append(tparams)

        # tack on the z-poly parameters on the end
        self.poly_params = {c+'-z-%i' % i:i for i in xrange(zorder)}
        params.extend(self.poly_params.keys())
        values.extend([0.0]*len(self.poly_params))

        super(BarnesStreakLegPoly2P1D, self).__init__(
            params=params, values=values, category=category
        )

        # this next variable is to allow for randomize_parameters before the
        # object has a shape by leaving a breadcrumb for normalization
        self._norm_stat = None

        if self.shape:
            self.initialize()

    def param_barnes_scales(self):
        return [self.category+'-scale', self.category+'-off']

    def param_barnes_pts(self, ind=None):
        if ind is None:
            return [l for b in self.barnes_params for l in b]
        return self.barnes_params[ind]

    def param_barnes_poly(self):
        return self.poly_params.keys()

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
            clip=self.local_updates, clipsize=self.barnes_clip_size
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
        return self.get_values(self.category+'-off')

    @property
    def scale(self):
        return self.get_values(self.category+'-scale')

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

        if self._norm_stat:
            ptp, vmin = self._norm_stat

            ptp0 = self.get().ptp()
            scale = ptp / ptp0
            self.update(self.category+'-scale', scale)

            min0 = self.get().min()
            off = vmin - min0
            self.update(self.category+'-off', off)

            self._norm_stat = None

    def set_tile(self, tile):
        self.tile = tile

    def update(self, params, values):
        op = {'*': mul, '+': add}[self.op]

        params = util.listify(params)
        values = util.listify(values)

        if len(params) < len(self.params)/2:
            calcpoly = False

            for p,v1 in zip(params, values):
                if p in self.poly_params:
                    calcpoly = True
                    #tm = self._term(self.poly_params[p])
                    #v0 = self.get_values(p)
                    #self.poly += (v1-v0) * tm

            self.set_values(params, values)
            if calcpoly:
                self.poly = self.calc_poly()
            self.field = self.calc_field()
        else:
            self.set_values(params, values)
            self.field = self.calc_field()

    def get(self):
        return self.field[self.tile.slicer]

    def get_update_tile(self, params, values):
        if not self.local_updates:
            return self.shape.copy()

        params = util.listify(params)
        values = util.listify(values)

        c = self.category
        # check for global update requiring parameters:
        for p in params:
            if p in self.poly_params or p == c+'-scale' or p == c+'-off':
                return self.shape.copy()

        # now look for the local update sizes
        orig_values = self.get_values(params)

        tiles = []
        for p,v in zip(params, values):
            # figure out the barnes local update size
            for n, grp in enumerate(self.barnes_params):
                if not p in grp:
                    continue

                val0 = self._barnes(self.b_out, n=n)
                self.set_values(p, v)
                val1 = self._barnes(self.b_out, n=n)

                inds = np.arange(self.b_out.shape[0])
                inds = inds[np.abs(val1 - val0) > 1e-12]
                if len(inds) < 2:
                    continue

                l, r = inds.min(), inds.max()

                tile = self.shape.copy()
                tile.l[2] = l
                tile.r[2] = r
                tiles.append(util.Tile(tile.l, tile.r))

        self.set_values(params, orig_values)
        if len(tiles) == 0:
            return None
        return util.Tile.boundingtile(tiles)

    def randomize_parameters(self, ptp=0.2, fourier=False, vmin=None, vmax=None):
        """
        Create random parameters for this ILM that mimic experiments
        as closely as possible without real assumptions.
        """
        if vmin is not None and vmax is not None:
            ptp = vmax - vmin
        elif vmax is not None and vmin is None:
            vmin = vmax - ptp
        elif vmin is not None and vmax is None:
            vmax = vmin + ptp
        else:
            vmax = 1.0
            vmin = vmax - ptp

        self.set_values(self.category+'-scale', 1.0)
        self.set_values(self.category+'-off', 0.0)

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

        self._norm_stat = [ptp, vmin]

        if self.shape:
            self.initialize()
    
    def nopickle(self):
        return super(BarnesStreakLegPoly2P1D, self).nopickle() + [
            'poly', 'b_in', 'b_out', 'r', 'field',
            '_last_term', '_last_index'
        ]

    def __str__(self):
        return "{} [{} {}]".format(
            self.__class__.__name__, self.npts, self.zorder
        )

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        odict = self.__dict__.copy()
        util.cdd(odict, self.nopickle())
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        if self.shape:
            self.initialize()


