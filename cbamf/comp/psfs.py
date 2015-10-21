import os
import atexit
import cPickle as pickle
import numpy as np
from multiprocessing import cpu_count
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval

from cbamf.util import Tile, cdd, memoize
from cbamf.conf import get_wisdom

try:
    import pyfftw
    from pyfftw.builders import fftn, ifftn, fft2, ifft2
    hasfftw = True
except ImportError as e:
    print "*WARNING* pyfftw not found, switching to numpy.fft (20x slower)"
    hasfftw = False

def load_wisdom():
    WISDOM_FILE = get_wisdom()
    try:
        pyfftw.import_wisdom(pickle.load(open(WISDOM_FILE)))
    except IOError as e:
        save_wisdom()

def save_wisdom():
    WISDOM_FILE = get_wisdom()
    pickle.dump(pyfftw.export_wisdom(), open(WISDOM_FILE, 'w'), protocol=-1)

@atexit.register
def goodbye():
    if hasfftw:
        save_wisdom()

if hasfftw:
    load_wisdom()

FFTW_PLAN_FAST = 'FFTW_ESTIMATE'
FFTW_PLAN_NORMAL = 'FFTW_MEASURE'
FFTW_PLAN_SLOW = 'FFTW_PATIENT'

#=============================================================================
# Begin 3-dimensional point spread functions
#=============================================================================
class PSF(object):
    def __init__(self, params, shape, fftw_planning_level=FFTW_PLAN_NORMAL, threads=-1):
        self.shape = shape
        self.params = np.array(params).astype('float')

        self.threads = threads if threads > 0 else cpu_count()
        self.fftw_planning_level = fftw_planning_level

        self.tile = Tile((0,0,0))
        self.update(self.params)
        self.set_tile(Tile(self.shape))

    def _setup_ffts(self):
        if hasfftw:
            self._fftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._ifftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._fftn = fftn(self._fftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)
            self._ifftn = ifftn(self._ifftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)

    def _setup_kvecs(self):
        sp = self.tile.shape
        kz = 2*np.pi*np.fft.fftfreq(sp[0])[:,None,None]
        ky = 2*np.pi*np.fft.fftfreq(sp[1])[None,:,None]
        kx = 2*np.pi*np.fft.fftfreq(sp[2])[None,None,:]
        self._kx, self._ky, self._kz = kx, ky, kz
        self._kvecs = np.rollaxis(np.array(np.broadcast_arrays(kz,ky,kx)), 0, 4)
        self._klen = np.sqrt(kx**2 + ky**2 + kz**2)

    def _setup_rvecs(self, shape, centered=True):
        sp = shape
        mx = np.max(sp)

        rz = sp[0]*np.fft.fftfreq(sp[0])[:,None,None]
        ry = sp[1]*np.fft.fftfreq(sp[1])[None,:,None]
        rx = sp[2]*np.fft.fftfreq(sp[2])[None,None,:]

        if centered:
            rz = np.fft.fftshift(rz)
            ry = np.fft.fftshift(ry)
            rx = np.fft.fftshift(rx)

        self._rx, self._ry, self._rz = rx, ry, rz
        self._rvecs = np.rollaxis(np.array(np.broadcast_arrays(rz,ry,rx)), 0, 4)
        self._rlen = np.sqrt(rx**2 + ry**2 + rz**2)

    @memoize()
    def _min_to_tile(self, shape):
        d = ((shape - self._min_support))

        # fix off-by-one issues when going odd to even tile sizes
        o = d % 2
        d /= 2

        pad = tuple((d[i],d[i]+o[i]) for i in [0,1,2])
        self.rpsf = np.pad(self._min_rpsf, pad, mode='constant', constant_values=0)
        self.rpsf = np.fft.fftshift(self.rpsf)
        self.kpsf = self.fftn(self.rpsf)
        self.kpsf /= (np.real(self.kpsf[0,0,0]) + 1e-15)
        return self.kpsf

    def fftn(self, arr):
        if hasfftw:
            self._fftn_data[:] = arr
            self._fftn.execute()
            return self._fftn.get_output_array().copy()
        else:
            return np.fft.fftn(arr)

    def ifftn(self, arr):
        if hasfftw:
            self._ifftn_data[:] = arr
            self._ifftn.execute()
            v = 1.0/self._fftn_data.size
            return self._ifftn.get_output_array() * v
        else:
            return np.fft.ifftn(arr)

    def set_tile(self, tile):
        if any(tile.shape < self._min_support):
            raise IndexError("PSF tile size is less than minimum support size")

        if (self.tile.shape != tile.shape).any():
            self.tile = tile
            self._setup_ffts()

        self.kpsf = self._min_to_tile(self.tile.shape)

    def update(self, params):
        self.params = params

        # calculate the minimum supported real-space PSF
        self._min_support = np.ceil(self.get_support_size()).astype('int')
        self._min_support += self._min_support % 2
        self._setup_rvecs(self._min_support)
        self._min_rpsf = self.rpsf_func()

        # clean out the cache since it is no longer useful
        if hasattr(self, '_memoize_clear'):
            self._memoize_clear()

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if not np.iscomplex(field.ravel()[0]):
            infield = self.fftn(field)
        else:
            infield = field

        return np.real(self.ifftn(infield * self.kpsf))

    def get_params(self):
        return self.params

    def get_support_size(self, z=None):
        return np.zeros(3)

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, ['_rx', '_ry', '_rz', '_rvecs', '_rlen'])
        cdd(odict, ['_kx', '_ky', '_kz', '_kvecs', '_klen'])
        cdd(odict, ['_fftn', '_ifftn', '_fftn_data', '_ifftn_data'])
        cdd(odict, ['_memoize_clear', '_memoize_caches'])
        cdd(odict, ['rpsf', 'kpsf'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.tile = Tile((0,0,0))
        self.update(self.params)
        self.set_tile(Tile(self.shape))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__class__.__name__)+" {} ".format(self.params)

class AnisotropicGaussian(PSF):
    def __init__(self, params, shape, error=1.0/255, *args, **kwargs):
        self.error = error
        super(AnisotropicGaussian, self).__init__(*args, params=params, shape=shape, **kwargs)

    def rpsf_func(self):
        params = self.params/2
        rt2 = np.sqrt(2)
        rhosq = self._rx**2 + self._ry**2
        arg = np.exp(-(rhosq)/(rt2*params[0])**2 - (self._rz/(rt2*params[1]))**2)
        return arg * (rhosq <= self.pr**2) * (np.abs(self._rz) <= self.pz)

    def get_support_size(self, z=None):
        self.pr = np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        return np.array([self.pz, self.pr, self.pr])

class AnisotropicGaussianXYZ(PSF):
    def __init__(self, params, shape, error=1.0/255, *args, **kwargs):
        self.error = error
        super(AnisotropicGaussianXYZ, self).__init__(*args, params=params, shape=shape, **kwargs)

    def rpsf_func(self):
        params = self.params/2
        rt2 = np.sqrt(2)
        arg = np.exp(-(self._rx/(rt2*params[0]))**2 - (self._ry/(rt2*params[1]))**2 - (self._rz/(rt2*params[2]))**2)
        return arg * (np.abs(self._rx) <= self.px) * (np.abs(self._ry) <= self.py) * (np.abs(self._rz) <= self.pz)

    def get_support_size(self, z=None):
        self.px = np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.py = np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        self.pz = np.sqrt(-2*np.log(self.error)*self.params[2]**2)
        return np.array([self.pz, self.py, self.px])

class GaussianPolynomialPCA(PSF):
    def __init__(self, cov_mat_file, mean_mat_file, shape, gaussian=(2,4),
            components=5, error=1.0/255, *args, **kwargs):
        self.cov_mat_file = cov_mat_file
        self.mean_mat_file = mean_mat_file
        self.comp = components
        self.error = error

        self._setup_from_files()
        params0 = np.hstack([gaussian, np.zeros(self.comp)])

        super(GaussianPolynomialPCA, self).__init__(*args, params=params0, shape=shape, **kwargs)

    def _setup_from_files(self):
        covm = np.load(self.cov_mat_file)
        mean = np.load(self.mean_mat_file)
        self.poly_shape = (np.round(np.sqrt(mean.shape[0])),)*2

        vals, vecs = np.linalg.eig(covm)
        self._psf_vecs = np.real(vecs[:,:self.comp])
        self._psf_mean = np.real(mean)

        # TODO -- proper calculation? it seems there is almost no opportunity to
        # cache these values, so perhaps we won't be doing this
        #self._polys = [np.polynomial.polynomial.polyval2d(
        #    rho, z, (self._psf_vecs[:,i] + self._psf_mean).reshape(*self.poly_shape)
        #) for i in xrange(self.comp)]
        #poly = (self._polys * self.params[2:]).sum(axis=-1)

    def rpsf_func(self):
        coeff = self.params
        coeff[:2] /= 2

        rho = np.sqrt(self._rx**2 + self._ry**2) / coeff[0]
        z = self._rz / coeff[1]

        polycoeffs = self._psf_vecs.dot(coeff[2:]) + self._psf_mean
        poly = np.polynomial.polynomial.polyval2d(rho, z, polycoeffs.reshape(*self.poly_shape))
        return poly * np.exp(-rho**2) * np.exp(-z**2) * (rho <= self.pr/coeff[0]) * (np.abs(z) <= self.pz/coeff[1])

    def get_support_size(self, z=None):
        self.pr = 1.4*np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = 1.4*np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        return np.array([self.pz, self.pr, self.pr])

class GaussianPolynomialPCA_XYZ(GaussianPolynomialPCA):
    def __init__(self, *args, **kwargs):
        super(GaussianPolynomialPCA_XYZ, self).__init__(*args, **kwargs)

    def rpsf_func(self):
        coeff = self.params
        coeff[:2] /= 2

        sigma_rho = np.sqrt(coeff[0]**2 + coeff[1]**2)
        rho = np.sqrt(self._rx**2 + self._ry**2) / sigma_rho
        x = self._rx / coeff[0]
        y = self._ry / coeff[1]
        z = self._rz / coeff[2]

        polycoeffs = self._psf_vecs.dot(coeff[3:]) + self._psf_mean
        poly = np.polynomial.polynomial.polyval2d(rho, z, polycoeffs.reshape(*self.poly_shape))
        mask = (np.abs(x) < self.px/coeff[0]) * (np.abs(y) < self.py/coeff[1]) * (np.abs(z) < self.pz/coeff[2])
        return poly * np.exp(-x**2) * np.exp(-y**2) * np.exp(-z**2) * mask

    def get_support_size(self, z=None):
        self.px = 1.4*np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.py = 1.4*np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        self.pz = 1.4*np.sqrt(-2*np.log(self.error)*self.params[2]**2)
        return np.array([self.pz, self.py, self.px])

class ASymmetricGaussianPolynomialPCA(PSF):
    def __init__(self, symm_cov_file, symm_mean_file, asymm_cov_file, asymm_mean_file, shape, gaussian=(2,4),
            components=5, error=1.0/255, *args, **kwargs):
        self.symm_cov_file = symm_cov_file
        self.symm_mean_file = symm_mean_file
        self.asymm_cov_file = asymm_cov_file
        self.asymm_mean_file = asymm_mean_file

        self.comp = components
        self.error = error

        self._setup_from_files()
        params0 = np.hstack([gaussian, np.zeros(2*self.comp)])

        super(ASymmetricGaussianPolynomialPCA, self).__init__(*args, params=params0, shape=shape, **kwargs)

    def _setup_from_files(self):
        symm_covm = np.load(self.symm_cov_file)
        symm_mean = np.load(self.symm_mean_file)
        asymm_covm = np.load(self.asymm_cov_file)
        asymm_mean = np.load(self.asymm_mean_file)

        self.poly_shape = (np.round(np.sqrt(symm_mean.shape[0])),)*2

        symm_vals, symm_vecs = np.linalg.eig(symm_covm)
        self._psf_symm_vecs = np.real(symm_vecs[:,:self.comp])
        self._psf_symm_mean = np.real(symm_mean)

        asymm_vals, asymm_vecs = np.linalg.eig(asymm_covm)
        self._psf_asymm_vecs = np.real(asymm_vecs[:,:self.comp])
        self._psf_asymm_mean = np.real(asymm_mean)

    def rpsf_func(self):
        coeff = self.params
        coeff[:2] /= 2

        rho = np.sqrt(self._rx**2 + self._ry**2) / coeff[0]
        z = self._rz / coeff[1]

        symm = coeff[2:2+self.comp]
        symm_polycoeffs = self._psf_symm_vecs.dot(symm) + self._psf_symm_mean
        symm_poly = np.polynomial.polynomial.polyval2d(rho, z, symm_polycoeffs.reshape(*self.poly_shape))

        asymm = coeff[2+self.comp:]
        asymm_polycoeffs = self._psf_asymm_vecs.dot(asymm) + self._psf_asymm_mean
        asymm_poly = np.polynomial.polynomial.polyval2d(rho, z, asymm_polycoeffs.reshape(*self.poly_shape))

        phi = np.arctan2(self._ry, self._rx)
        out = (symm_poly + np.cos(2*phi)*asymm_poly) * np.exp(-rho**2) * np.exp(-z**2)
        return out * (rho <= self.pr/coeff[0]) * (np.abs(z) <= self.pz/coeff[1])

    def get_support_size(self, z=None):
        self.pr = 1.4*np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = 1.4*np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        return np.array([self.pz, self.pr, self.pr])


#=============================================================================
# Begin 4-dimensional point spread functions
#=============================================================================
class PSF4D(PSF):
    """
    4-dimensional Point-Spread-Function (PSF) is implemented by assuming that
    the there is only z-dependence of parameters (so that it can be separated
    into x-y and z parts).  Therefore, we keep track of 2D FFTs and X-Y psf
    functions that get convolved with FFTs.  Then, we manually convolve in the
    z direction

    The key variables are rpsf (2d) and kpsf (2d) which are used for the x-y
    convolution.  The z-convolution cannot be cached.
    """
    def __init__(self, params, shape, *args, **kwargs):
        super(PSF4D, self).__init__(*args, params=params, shape=shape, **kwargs)

    def _setup_ffts(self):
        if hasfftw:
            self._fftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._ifftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._fftn = fft2(self._fftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)
            self._ifftn = ifft2(self._ifftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)

    def _setup_rvecs(self, shape, centered=False):
        sp = shape
        mx = np.max(sp)

        ry = sp[1]*np.fft.fftfreq(sp[1])[:,None]
        rx = sp[2]*np.fft.fftfreq(sp[2])[None,:]

        self._rx, self._ry = rx, ry

    def _zpos(self):
        return np.arange(self.tile.l[0], self.tile.r[0]).astype('float')

    @memoize()
    def _calc_tile_2d_psf(self, tile):
        self.rpsf = np.zeros(shape=self.tile.shape)
        zs = self._zpos()

        # calculate each slice from the rpsf_xy function
        for i,z in enumerate(zs):
            self.rpsf[i] = self.rpsf_xy(z)

        # calcualte the psf in k-space using 2d ffts
        self.kpsf = self.fftn(self.rpsf)

        # need to normalize each x-y slice individually
        for i,z in enumerate(zs):
            self.kpsf[i] /= self.kpsf[i,0,0]

        return self.kpsf

    def set_tile(self, tile):
        if (self.tile.shape != tile.shape).any():
            self.tile = tile
            self._setup_ffts()

        self._setup_rvecs(self.tile.shape, centered=False)
        self.kpsf = self._calc_tile_2d_psf(self.tile)

    def update(self, params):
        # what should we update when the parameters are adjusted for
        # the 4d psf?  Well, for simplicity, let's start with nothing.
        self.params = params

        # clean out the cache since it is no longer useful
        if hasattr(self, '_memoize_clear'):
            self._memoize_clear()

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if not np.iscomplex(field.ravel()[0]):
            infield = self.fftn(field)
        else:
            infield = field

        cov2d = np.real(self.ifftn(infield * self.kpsf)) * self.tile.shape[0]
        cov2dT = np.rollaxis(cov2d, 0, 3)

        out = np.zeros_like(cov2d)
        z = self._zpos()

        for i in xrange(len(z)):
            size = self.get_support_size(z=z[i])
            m = (z >= z[i]-size[0]) & (z <= z[i]+size[0])
            g = self.rpsf_z(z[m], z[i])
            out[i] = cov2dT[...,m].dot(g)

        return out

    def rpsf_xy(self, z):
        """
        Returns the x-y plane real space psf function as a function of z values.
        This function does not necessarily have to be normalized, it will be
        normalized in k-space layer by layer later.
        """
        pass

    def rpsf_z(self, z):
        """
        Returns the z dependence of the PSF.  This section needs to be noramlized.
        """
        pass

class Gaussian4D(PSF4D):
    def __init__(self, shape, params=(1.0,0.5,2.0), order=(1,1,1), error=1.0/255, zrange=128, *args, **kwargs):
        self.order = order
        self.error = error
        self.zrange = float(zrange)
        params = np.hstack([np.array(params)[:3], np.zeros(np.sum(order))])
        super(Gaussian4D, self).__init__(params=params, shape=shape, *args, **kwargs)

    def _setup_ffts(self):
        if hasfftw:
            self._fftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._ifftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._fftn = fft2(self._fftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)
            self._ifftn = ifft2(self._ifftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)

    def _sigma_coeffs(self, d=0):
        s = 3 + np.sum(self.order[:d+0])
        e = 3 + np.sum(self.order[:d+1])
        return np.hstack([1, self.params[s:e]])

    def _poly(self, z, coeffs):
        return np.polyval(coeffs[::-1], z)

    @memoize()
    def _sigma(self, z, d=0):
        return self.params[d]*self._poly(z/self.zrange, self._sigma_coeffs(d=d))

    @memoize()
    def get_support_size(self, z):
        if isinstance(z, np.ndarray) and z.shape[0] > 1:
            z = z[0]
        s = np.array([self._sigma(z, 0), self._sigma(z, 1), self._sigma(z, 2)])
        self.px = np.max([np.sqrt(-2*np.log(self.error)*s[0]**2), 2.1*np.ones_like(s[0])], axis=0)
        self.py = np.max([np.sqrt(-2*np.log(self.error)*s[1]**2), 2.1*np.ones_like(s[1])], axis=0)
        self.pz = np.max([np.sqrt(-2*np.log(self.error)*s[2]**2), 2.1*np.ones_like(s[2])], axis=0)
        size = np.array([self.pz, self.py, self.px])
        return size

    @memoize()
    def rpsf_z(self, z, zp):
        s = self._sigma(zp, 2)
        size = self.get_support_size(z=zp)
        return 1.0/np.sqrt(2*np.pi*s**2) * np.exp(-(z-zp)**2 / (2*s**2)) * (np.abs(z-zp) <= size[0])

    def rpsf_xy(self, zp):
        size = self.get_support_size(z=zp)
        mask = (np.abs(self._rx) <= size[2]) * (np.abs(self._ry) <= size[1])

        sx = self._sigma(zp, 0)
        sy = self._sigma(zp, 1)
        gauss = np.exp(-(self._rx/sx)**2/2-(self._ry/sy)**2/2)

        return gauss * mask

class Gaussian4DPoly(Gaussian4D):
    def __init__(self, shape, params=(1.0,0.5,2.0), order=(1,1,1),
            error=1.0/255, zrange=128, *args, **kwargs):
        super(Gaussian4DPoly, self).__init__(shape=shape, params=params,
                order=order, error=error, zrange=zrange, *args, **kwargs)

    def _sigma_coeffs(self, d=0):
        s = 3 + np.sum(self.order[:d+0])
        e = 3 + np.sum(self.order[:d+1])
        return np.hstack([self.params[d], self.params[s:e]])

    def _poly(self, z, coeffs):
        return np.polyval(coeffs[::-1], z)

    @memoize()
    def _sigma(self, z, d=0):
        return self._poly(z/self.zrange, self._sigma_coeffs(d=d))

class Gaussian4DLegPoly(Gaussian4DPoly):
    def __init__(self, shape, params=(1.0,0.5,2.0), order=(1,1,1),
            error=1.0/255, zrange=128, *args, **kwargs):
        super(Gaussian4DLegPoly, self).__init__(shape=shape, params=params,
                order=order, error=error, zrange=zrange, *args, **kwargs)

    def _poly(self, z, coeffs):
        return legval(z, coeffs)

class GaussianMomentExpansion(PSF4D):
    def __init__(self, shape, params=(1.0,0.5,2.0), order=(1,1,1),
            moment_order=(3,3), error=1.0/255, zrange=128, *args, **kwargs):
        """
        3+1D PSF that is of the form  (1+a*(3x-x^3) + b*(3-6*x^2+x^4))*exp(-(x/s)^2/2) where s
        is sigma, the scale factor.  
        """
        self.order = order
        self.morder = moment_order
        self.error = error
        self.zrange = float(zrange)
        params = np.hstack([np.array(params)[:3], np.zeros(np.sum(order)), np.zeros(2*np.sum(moment_order))])
        super(GaussianMomentExpansion, self).__init__(params=params, shape=shape, *args, **kwargs)

    def _setup_ffts(self):
        if hasfftw:
            self._fftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._ifftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
            self._fftn = fft2(self._fftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)
            self._ifftn = ifft2(self._ifftn_data, overwrite_input=False,
                    planner_effort=self.fftw_planning_level, threads=self.threads)

    def _sigma_coeffs(self, d=0):
        s = 3 + np.sum(self.order[:d+0])
        e = 3 + np.sum(self.order[:d+1])
        return np.hstack([1, self.params[s:e]])

    def _poly(self, z, coeffs):
        return np.polyval(coeffs[::-1], z)

    @memoize()
    def _sigma(self, z, d=0):
        return self.params[d]*self._poly(z/self.zrange, self._sigma_coeffs(d=d))

    @memoize()
    def _moment(self, x, z, d=0):
        return (1+self._skew(x,z,d=d)+self._kurtosis(x, z, d=d))

    @memoize()
    def _kurtosis_coeffs(self, d):
        i0 = 3 + np.sum(self.order) + np.sum(self.morder[:d+0])
        i1 = 3 + np.sum(self.order) + np.sum(self.morder[:d+1])
        coeffs = self.params[i0:i1]
        return coeffs

    @memoize()
    def _skew_coeffs(self, d):
        o = np.sum(self.morder[:d+1])
        i0 = 3 + np.sum(self.order) + o + np.sum(self.morder[:d+0])
        i1 = 3 + np.sum(self.order) + o + np.sum(self.morder[:d+1])
        coeffs = self.params[i0:i1]
        return coeffs

    @memoize()
    def _skew(self, x, z, d=0):
        """ returns the kurtosis parameter for direction d, d=0 is rho, d=1 is z """
        # get the top bound determined by the kurtosis
        kval = (np.tanh(self._poly(z, self._kurtosis_coeffs(d)))+1)/12.
        bdpoly = np.array([-1.142468e+04, 3.0939485e+03, -2.0283568e+02, -2.1047846e+01, 3.79808487e+00, 1.19679781e-02])
        top = np.polyval(bdpoly, kval)

        # limit the skewval to be 0 -> top val
        skew = self._poly(z, self._skew_coeffs(d))
        skewval = top*(np.tanh(skew) + 1) - top

        return skewval*(3 - 6*x**2 + x**4)

    @memoize()
    def _kurtosis(self, x, z, d=0):
        """ returns the kurtosis parameter for direction d, d=0 is rho, d=1 is z """
        val = self._poly(z, self._kurtosis_coeffs(d))
        return (np.tanh(val)+1)/12.*(3 - 6*x**2 + x**4)

    @memoize()
    def get_support_size(self, z):
        if isinstance(z, np.ndarray) and z.shape[0] > 1:
            z = z[0]
        s = np.array([self._sigma(z, 0), self._sigma(z, 1), self._sigma(z, 2)])
        self.px = np.max([np.sqrt(-2*np.log(self.error)*s[0]**2), 2.1*np.ones_like(s[0])], axis=0)
        self.py = np.max([np.sqrt(-2*np.log(self.error)*s[1]**2), 2.1*np.ones_like(s[1])], axis=0)
        self.pz = np.max([np.sqrt(-2*np.log(self.error)*s[2]**2), 2.1*np.ones_like(s[2])], axis=0)
        size = np.array([self.pz, self.py, self.px])
        return size

    @memoize()
    def rpsf_z(self, z, zp):
        s = self._sigma(zp, 2)
        size = self.get_support_size(z=zp)
        pref = self._moment((z-zp)/s, zp, d=1)
        out = pref*np.exp(-(z-zp)**2 / (2*s**2)) * (np.abs(z-zp) <= size[0])
        return out / out.sum()

    def rpsf_xy(self, zp):
        size = self.get_support_size(z=zp)
        mask = (np.abs(self._rx) <= size[2]) * (np.abs(self._ry) <= size[1])

        sx = self._sigma(zp, 0)
        sy = self._sigma(zp, 1)
        rho = np.sqrt((self._rx/sx)**2 + (self._ry/sy)**2)
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
        self.param_shape = array.shape
        self.support = np.array(array.shape[1:])
        super(FromArray, self).__init__(*args, params=array.flatten(), **kwargs)

    def set_tile(self, tile):
        if (self.tile.shape != tile.shape).any():
            self.tile = tile
            self._setup_ffts()

    def _pad(self, field):
        if any(self.tile.shape < self.get_support_size()):
            raise IndexError("PSF tile size is less than minimum support size")

        d = self.tile.shape - self.get_support_size()

        # fix off-by-one issues when going odd to even tile sizes
        o = d % 2
        d /= 2

        pad = tuple((d[i],d[i]+o[i]) for i in [0,1,2])
        rpsf = np.pad(field, pad, mode='constant', constant_values=0)
        rpsf = np.fft.fftshift(rpsf)
        kpsf = np.fft.fftn(rpsf)
        kpsf /= (np.real(kpsf[0,0,0]) + 1e-15)
        return kpsf

    def update(self, params):
        pass

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if not np.iscomplex(field.ravel()[0]):
            infield = self.fftn(field)
        else:
            infield = field

        outfield = np.zeros_like(infield, dtype='float')

        for i in xrange(field.shape[0]):
            z = int(self.tile.l[0] + i)
            kpsf = self._pad(self.params.reshape(self.param_shape)[z])
            outfield[i] = np.real(self.ifftn(infield * kpsf))[i]

        return outfield

    def get_params(self):
        return self.params

    def get_support_size(self, z=None):
        return self.support
