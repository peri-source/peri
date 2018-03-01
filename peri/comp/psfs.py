from builtins import range

import os
import atexit
import pickle
import numpy as np
from multiprocessing import cpu_count
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval

from peri.fft import fft, fftkwargs
from peri.comp import Component
from peri.util import Tile, cdd, memoize, listify

#=============================================================================
# Begin 3-dimensional point spread functions
#=============================================================================
class PSF(Component):
    category = 'psf'

    def __init__(self, params, values, shape=None):
        """
        Point spread function classes must contain the following classes in order
        to interface with the states class:

            get_padding_size(z) : get the psf size at certain z position
            get_params : return the parameters of the psf
            set_tile : set the current update tile size
            update : update the psf based on new parameters
            execute : apply the psf to an image
        """
        self.shape = shape
        super(PSF, self).__init__(params, values, category='psf')

        if self.shape:
            self.initialize()

    def initialize(self):
        self.update(self.params, self.values)
        self.set_tile(self.shape)

    @memoize()
    def calculate_kpsf(self, shape):
        d = ((shape - self.min_support))

        # fix off-by-one issues when going odd to even tile sizes
        o = d % 2
        d = d // 2

        pad = tuple((d[i],d[i]+o[i]) for i in [0,1,2])
        self.rpsf = np.pad(self.min_rpsf, pad, mode='constant', constant_values=0)
        self.rpsf = fft.ifftshift(self.rpsf)
        self.kpsf = fft.fftn(self.rpsf, **fftkwargs)
        self.kpsf /= (np.real(self.kpsf[0,0,0]) + 1e-15)
        return self.kpsf

    def calculate_min_rpsf(self):
        # calculate the minimum supported real-space PSF
        min_support = self.get_padding_size(self.shape).shape
        min_support += min_support % 2
        min_rpsf = self.rpsf_func(self._rvecs(min_support))
        return min_rpsf, min_support

    def set_tile(self, tile):
        if any(tile.shape < self.min_support):
            raise IndexError("PSF tile size is less than minimum support size")

        if not hasattr(self, 'tile') or (self.tile.shape != tile.shape).any():
            self.tile = tile

        self.kpsf = self.calculate_kpsf(self.tile.shape)

    def update(self, params, values):
        self.set_values(params, values)
        self.min_rpsf, self.min_support = self.calculate_min_rpsf()

        # clean out the cache since it is no longer useful
        if hasattr(self, '_memoize_clear'):
            self._memoize_clear()

        if hasattr(self, 'tile'):
            self.set_tile(self.tile)

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if not np.iscomplex(field.ravel()[0]):
            infield = fft.fftn(field, **fftkwargs)
        else:
            infield = field

        return np.real(fft.ifftn(infield * self.kpsf, **fftkwargs))

    def get(self):
        return self

    def get_update_tile(self, params, values):
        return self.shape.copy()

    def get_padding_size(self, tile):
        raise NotImplemented('subclasses must implement `get_padding_size`')

    def nopickle(self):
        return super(PSF, self).nopickle() + [
            '_memoize_clear', '_memoize_caches',
            'rpsf', 'kpsf', 'min_rpsf'
        ]

    def _rvecs(self, shape, centered=True):
        tile = Tile(shape)
        return tile.kvectors(norm=1.0/tile.shape, shift=centered)

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, self.nopickle())
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        if self.shape:
            self.initialize()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.values)

class IdentityPSF(PSF):
    def __init__(self):
        """
        Delta-function PSF; returns the field passed to execute identically. 
        Params is an N-element numpy.ndarray, doesn't do anything. 
        """
        self.min_support = 0
        super(IdentityPSF, self).__init__(params=['psf'], values=[0])

    def set_tile(self, tile):
        if any(tile.shape < self.min_support):
            raise IndexError("PSF tile size is less than minimum support size")

        if not hasattr(self, 'tile') or (self.tile.shape != tile.shape).any():
            self.tile = tile

    def execute(self, field):
        return field
    
    def get_padding_size(self, tile):
        return Tile(np.ones(3))
    
    def update(self, params, values):
        self.set_values(params, values)

class AnisotropicGaussian(PSF):
    def __init__(self, sigmas=(2.0, 1.0), error=1.0/255, shape=None):
        self.error = error
        params = ['psf-sig-z', 'psf-sig-rho'] 
        super(AnisotropicGaussian, self).__init__(
            shape=shape, params=params, values=sigmas
        )

    def rpsf_func(self, vecs):
        rz, ry, rx = vecs
        rhosq = rx**2 + ry**2

        vals = np.array(self.values)/2
        arg = np.exp(-(rhosq/vals[1]**2 + (rz/vals[0])**2)/2)
        return arg * (rhosq <= self.pr**2) * (np.abs(rz) <= self.pz)

    def get_padding_size(self, tile):
        self.pr = np.sqrt(-2*np.log(self.error)*self.values[0]**2)
        self.pz = np.sqrt(-2*np.log(self.error)*self.values[1]**2)
        return Tile(np.ceil([self.pz, self.pr, self.pr]))

class AnisotropicGaussianXYZ(PSF):
    def __init__(self, sigmas=(2.0, 0.5, 1.0), error=1.0/255, shape=None):
        self.error = error
        params = ['psf-sigz', 'psf-sigy', 'psf-sigx']
        super(AnisotropicGaussianXYZ, self).__init__(
            shape=shape, params=params, values=sigmas
        )

    def rpsf_func(self, vecs):
        rz, ry, rx = vecs
        params = np.array(self.values)/2

        arg = np.exp(-(
            (rz/params[0])**2 + (ry/params[1])**2 + (rx/params[2])**2
        )/2)
        return arg * (
            (np.abs(rx) <= self.px) * (np.abs(ry) <= self.py) *
            (np.abs(rz) <= self.pz)
        )

    def get_padding_size(self, tile):
        self.px = np.sqrt(-2*np.log(self.error)*self.values[2]**2)
        self.py = np.sqrt(-2*np.log(self.error)*self.values[1]**2)
        self.pz = np.sqrt(-2*np.log(self.error)*self.values[0]**2)
        return Tile(np.ceil([self.pz, self.py, self.px]))


#=============================================================================
# Begin 4-dimensional point spread functions
#=============================================================================
class PSF4D(PSF):
    def __init__(self, params, values, shape=None):
        """
        4-dimensional Point-Spread-Function (PSF) is implemented by assuming
        that the there is only z-dependence of parameters (so that it can be
        separated into x-y and z parts).  Therefore, we keep track of 2D FFTs
        and X-Y psf functions that get convolved with FFTs.  Then, we manually
        convolve in the z direction

        The key variables are rpsf (2d) and kpsf (2d) which are used for the
        x-y convolution.  The z-convolution cannot be cached.
        """
        super(PSF4D, self).__init__(params=params, values=values, shape=shape)

    def rvecs(self, tile):
        rz, ry, rx = tile.kvectors(norm=1.0/tile.shape)
        return rx, ry

    def _zpos(self, tile):
        return np.arange(tile.l[0], tile.r[0]).astype('float')

    @memoize()
    def _calc_tile_2d_psf(self, tile):
        rpsf = np.zeros(shape=tile.shape)
        kpsf = np.zeros(shape=tile.shape, dtype='complex')

        vecs = self.rvecs(tile)
        zs = self._zpos(tile)

        # calculate each slice from the rpsf_xy function
        for i,z in enumerate(zs):
            rpsf[i] = self.rpsf_xy(vecs, z)

        # calcualte the psf in k-space using 2d ffts
        kpsf = fft.fft2(rpsf, **fftkwargs)

        # need to normalize each x-y slice individually
        for i,z in enumerate(zs):
            kpsf[i] *= 1.0/kpsf[i,0,0]

        return rpsf, kpsf

    def set_tile(self, tile):
        if not hasattr(self, 'tile') or (self.tile != tile).any():
            self.tile = tile

        self.rpsf, self.kpsf = self._calc_tile_2d_psf(self.tile)

    def update(self, params, values):
        # what should we update when the parameters are adjusted for
        # the 4d psf?  Well, for simplicity, let's start with nothing.
        self.set_values(params, values)

        # clean out the cache since it is no longer useful
        if hasattr(self, '_memoize_clear'):
            self._memoize_clear()

        if hasattr(self, 'tile'):
            self.set_tile(self.tile)

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if not np.iscomplexobj(field):
            infield = fft.fft2(field, **fftkwargs)
        else:
            infield = field

        cov2d = np.real(fft.ifft2(infield * self.kpsf, **fftkwargs))
        cov2dT = np.rollaxis(cov2d, 0, 3)

        out = np.zeros_like(cov2d)
        z = self._zpos(self.tile)

        for i in range(len(z)):
            size = self.get_padding_size(tile=None, z=z[i]).shape
            m = (z >= z[i]-size[0]) & (z <= z[i]+size[0])
            g = self.rpsf_z(z[m], z[i])
            out[i] = cov2dT[...,m].dot(g)

        return out

    def rpsf_xy(self, vecs, z):
        """
        Returns the x-y plane real space psf function as a function of z values.
        This function does not necessarily have to be normalized, it will be
        normalized in k-space layer by layer later.
        """
        pass

    def rpsf_z(self, z, zp):
        """
        Returns the z dependence of the PSF.  This section needs to be noramlized.
        """
        pass

class Gaussian4D(PSF4D):
    def __init__(self, sigmas=(2.0,0.5,1.0), order=(1,1,1), error=1.0/255,
            zrange=128, shape=None):
        self.order = order
        self.error = error
        self.zrange = float(zrange)

        self.coeffs = {}
        params, values = [], []
        for i, o in enumerate(order):
            d = ['z', 'y', 'x']
            tparams = ['psf-%s-%i' % (d[i], j) for j in range(o)]
            tvalues = [sigmas[i]] + [0]*(o-1)

            params.extend(tparams)
            values.extend(tvalues)
            self.coeffs[i] = tparams

        super(Gaussian4D, self).__init__(shape=shape, params=params, values=values)

    def _sigma_coeffs(self, d=0):
        return listify(self.get_values(self.coeffs[d]))

    def _poly(self, z, coeffs):
        return np.polyval(coeffs[::-1], z)

    @memoize()
    def _sigma(self, z, d=0):
        return self._poly(z/self.zrange, self._sigma_coeffs(d=d))

    @memoize()
    def get_padding_size(self, tile, z=None):
        if tile is not None:
            tile0 = self.get_padding_size(tile=None, z=tile.l[0])
            tile1 = self.get_padding_size(tile=None, z=tile.r[0])
            return Tile.intersection(tile0, tile1)

        if isinstance(z, np.ndarray) and z.shape[0] > 1:
            z = z[0]

        s = np.array([self._sigma(z, 0), self._sigma(z, 1), self._sigma(z, 2)])
        self.pz = np.max([np.sqrt(-2*np.log(self.error)*s[0]**2), 2.1*np.ones_like(s[0])], axis=0)
        self.py = np.max([np.sqrt(-2*np.log(self.error)*s[1]**2), 2.1*np.ones_like(s[1])], axis=0)
        self.px = np.max([np.sqrt(-2*np.log(self.error)*s[2]**2), 2.1*np.ones_like(s[2])], axis=0)
        return Tile(np.ceil([self.pz, self.py, self.px]))

    @memoize()
    def rpsf_z(self, z, zp):
        s = self._sigma(zp, 0)
        size = self.get_padding_size(tile=None, z=zp).shape
        out = np.exp(-(z-zp)**2 / (2*s**2)) * (np.abs(z-zp) <= size[0])
        return out / out.sum()

    def rpsf_xy(self, vecs, zp):
        rx, ry = vecs
        size = self.get_padding_size(tile=None, z=zp).shape
        mask = (np.abs(rx) <= size[2]) * (np.abs(ry) <= size[1])

        sx = self._sigma(zp, 2)
        sy = self._sigma(zp, 1)
        gauss = np.exp(-(rx/sx)**2/2-(ry/sy)**2/2)

        return gauss * mask

class Gaussian4DPoly(Gaussian4D):
    def __init__(self, sigmas=(2.0,0.5,1.0), order=(1,1,1), shape=None,
            error=1.0/255, zrange=128):
        super(Gaussian4DPoly, self).__init__(
            shape=shape, sigmas=sigmas, order=order, error=error, zrange=zrange
        )

    def _sigma_coeffs(self, d=0):
        return listify(self.get_values(self.coeffs[d]))

    def _poly(self, z, coeffs):
        return np.polyval(coeffs[::-1], z)

    @memoize()
    def _sigma(self, z, d=0):
        return self._poly(z/self.zrange, self._sigma_coeffs(d=d))

class Gaussian4DLegPoly(Gaussian4DPoly):
    def __init__(self, sigmas=(2.0,0.5,1.0), order=(1,1,1), error=1.0/255,
            zrange=128, shape=None):
        super(Gaussian4DLegPoly, self).__init__(
            shape=shape, sigmas=sigmas, order=order, error=error, zrange=zrange
        )

    def _poly(self, z, coeffs):
        return legval(z, coeffs)

class GaussianMomentExpansion(PSF4D):
    def __init__(self, sigmas=(2.0,0.5,1.0), order=(1,1,1),
            moment_order=(3,3), error=1.0/255, zrange=128, shape=None):
        """
        3+1D PSF that is of the form:
            (1+a*(3x-x^3) + b*(3-6*x^2+x^4))*exp(-(x/s)^2/2)
        where s is sigma, the scale factor.  
        """
        self.order = order
        self.morder = moment_order
        self.error = error
        self.zrange = float(zrange)

        self.poly_coeffs = {}
        self.moment_coeffs = {}

        params, values = [], []
        for i, o in enumerate(order):
            d = ['z', 'y', 'x']
            tparams = ['psf-%s-%i' % (d[i], j) for j in range(o)]
            tvalues = [sigmas[i]] + [0]*(o-1)

            params.extend(tparams)
            values.extend(tvalues)
            self.poly_coeffs[i] = tparams

        for i, o in enumerate(moment_order):
            t = ['skew', 'kurt']
            self.moment_coeffs[t[i]] = {}

            for j, d in enumerate(['z', 'y', 'x']):
                tparams = [
                    'psf-%s-%s-%i' % (t[i], d, k) for k in range(o)
                ]
                tvalues = [0]*(o)

                params.extend(tparams)
                values.extend(tvalues)
                self.moment_coeffs[t[i]][j] = tparams

        super(GaussianMomentExpansion, self).__init__(
            shape=shape, params=params, values=values
        )

    def _sigma_coeffs(self, d=0):
        return listify(self.get_values(self.poly_coeffs[d]))

    def _poly(self, z, coeffs):
        return np.polyval(coeffs[::-1], z)

    @memoize()
    def _sigma(self, z, d=0):
        return self._poly(z/self.zrange, self._sigma_coeffs(d=d))

    @memoize()
    def _moment(self, x, z, d=0):
        return (1+self._skew(x,z,d=d)+self._kurtosis(x, z, d=d))

    @memoize()
    def _kurtosis_coeffs(self, d):
        return listify(self.get_values(self.moment_coeffs['kurt'][d]))

    @memoize()
    def _skew_coeffs(self, d):
        return listify(self.get_values(self.moment_coeffs['skew'][d]))

    @memoize()
    def _skew(self, x, z, d=0):
        """ returns the kurtosis parameter for direction d, d=0 is rho, d=1 is z """
        # get the top bound determined by the kurtosis
        kval = (np.tanh(self._poly(z, self._kurtosis_coeffs(d)))+1)/12.
        bdpoly = np.array([
            -1.142468e+04,  3.0939485e+03, -2.0283568e+02,
            -2.1047846e+01, 3.79808487e+00, 1.19679781e-02
        ])
        top = np.polyval(bdpoly, kval)

        # limit the skewval to be 0 -> top val
        skew = self._poly(z, self._skew_coeffs(d))
        skewval = top*(np.tanh(skew) + 1) - top

        return skewval*(3*x - x**3)

    @memoize()
    def _kurtosis(self, x, z, d=0):
        """ returns the kurtosis parameter for direction d, d=0 is rho, d=1 is z """
        val = self._poly(z, self._kurtosis_coeffs(d))
        return (np.tanh(val)+1)/12.*(3 - 6*x**2 + x**4)

    @memoize()
    def get_padding_size(self, tile, z=None):
        if tile is not None:
            tile0 = self.get_padding_size(tile=None, z=tile.l[0])
            tile1 = self.get_padding_size(tile=None, z=tile.r[0])
            return Tile.intersection(tile0, tile1)

        if isinstance(z, np.ndarray) and z.shape[0] > 1:
            z = z[0]

        s = np.array([self._sigma(z, 0), self._sigma(z, 1), self._sigma(z, 2)])
        self.pz = np.max([np.sqrt(-2*np.log(self.error)*s[0]**2), 2.1*np.ones_like(s[0])], axis=0)
        self.py = np.max([np.sqrt(-2*np.log(self.error)*s[1]**2), 2.1*np.ones_like(s[1])], axis=0)
        self.px = np.max([np.sqrt(-2*np.log(self.error)*s[2]**2), 2.1*np.ones_like(s[2])], axis=0)
        return Tile(np.ceil([self.pz, self.py, self.px]))

    @memoize()
    def rpsf_z(self, z, zp):
        s = self._sigma(zp, 0)
        size = self.get_padding_size(tile=None, z=zp).shape
        pref = self._moment((z-zp)/s, zp, d=1)
        out = pref*np.exp(-(z-zp)**2 / (2*s**2)) * (np.abs(z-zp) <= size[0])
        return out / out.sum()

    def rpsf_xy(self, vecs, zp):
        rx, ry = vecs
        size = self.get_padding_size(tile=None, z=zp).shape
        mask = (np.abs(rx) <= size[2]) * (np.abs(ry) <= size[1])

        sx = self._sigma(zp, 2)
        sy = self._sigma(zp, 1)
        rho = np.sqrt((rx/sx)**2 + (ry/sy)**2)
        gauss = self._moment(rho, zp, d=0)*np.exp(-rho**2/2)

        return gauss * mask

#=============================================================================
# Array-based specification of PSF
#=============================================================================
class FromArray(PSF):
    def __init__(self, array, *args, **kwargs):
        """
        Only thing to pass is the values of the point spread function (does not
        need to be normalized) in the form of a numpy ndarray of shape

        (z', z, y, x)

        so if there are 50 layers in the image and the psf is at most 16 wide
        then it must be shaped (50,16,16,16). The values of the psf must
        be centered in this array. Hint: np.fft.fftfreq provides the correct
        ordering of values for both even and odd lattices.
        """
        self.array = array
        self.support = np.array(array.shape[1:])
        super(FromArray, self).__init__(*args, params=['dummy'], values=[0], **kwargs)

    def set_tile(self, tile):
        if (self.tile != tile).any():
            self.tile = tile

    def _pad(self, field):
        if any(self.tile.shape < self.get_padding_size().shape):
            raise IndexError("PSF tile size is less than minimum support size")

        d = self.tile.shape - self.get_padding_size().shape

        # fix off-by-one issues when going odd to even tile sizes
        o = d % 2
        d = np.floor_divide(d, 2)

        pad = tuple((d[i],d[i]+o[i]) for i in [0,1,2])
        rpsf = np.pad(field, pad, mode='constant', constant_values=0)
        rpsf = fft.ifftshift(rpsf)
        kpsf = fft.fftn(rpsf, **fftkwargs)
        kpsf /= (np.real(kpsf[0,0,0]) + 1e-15)
        return kpsf

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if not np.iscomplex(field.ravel()[0]):
            infield = fft.fftn(field, **fftkwargs)
        else:
            infield = field

        outfield = np.zeros_like(infield, dtype='float')

        for i in range(field.shape[0]):
            z = int(self.tile.l[0] + i)
            kpsf = self._pad(self.array[z])
            outfield[i] = np.real(fft.ifftn(infield * kpsf, **fftkwargs))[i]

        return outfield

    def get_padding_size(self, tile, z=None):
        return Tile(self.support)
