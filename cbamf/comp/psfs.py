import os
import atexit
import cPickle as pickle
import numpy as np
from multiprocessing import cpu_count

from cbamf.util import Tile, cdd
from cbamf.conf import get_wisdom

try:
    import pyfftw
    from pyfftw.builders import fftn, ifftn
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

class PSF(object):
    def __init__(self, params, shape, fftw_planning_level=FFTW_PLAN_NORMAL, threads=-1,
            cache_tiles=True, cache_max_size=1e9):
        self.cache_max_size = cache_max_size
        self.cache_tiles = cache_tiles
        self._cache = {}
        self._cache_size = 0
        self._cache_hits = 0
        self._cache_misses = 0

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

        rz = 2*sp[0]*np.fft.fftfreq(sp[0])[:,None,None]
        ry = 2*sp[1]*np.fft.fftfreq(sp[1])[None,:,None]
        rx = 2*sp[2]*np.fft.fftfreq(sp[2])[None,None,:]

        if centered:
            rz = np.fft.fftshift(rz)
            ry = np.fft.fftshift(ry)
            rx = np.fft.fftshift(rx)

        self._rx, self._ry, self._rz = rx, ry, rz
        self._rvecs = np.rollaxis(np.array(np.broadcast_arrays(rz,ry,rx)), 0, 4)
        self._rlen = np.sqrt(rx**2 + ry**2 + rz**2)

    def _min_to_tile(self):
        d = ((self.tile.shape - self._min_support)/2)

        pad = tuple((d[i],d[i]) for i in [0,1,2])
        self.rpsf = np.pad(self._min_rpsf, pad, mode='constant', constant_values=0)
        self.rpsf = np.fft.fftshift(self.rpsf)
        self.kpsf = self.fftn(self.rpsf)
        self.normalize_kpsf()

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

    def normalize_kpsf(self):
        self.kpsf /= np.real(self.kpsf[0,0,0])
        pass

    def set_tile(self, tile):
        if any(tile.shape < self._min_support):
            raise IndexError("PSF tile size is less than minimum support size")

        if (self.tile.shape != tile.shape).any():
            self.tile = tile
            self._setup_ffts()

        key = tuple(self.tile.shape)
        if key in self._cache:
            self.kpsf = self._cache[key]
            self._cache_hits += 1
        else:
            self._min_to_tile()

            # if we have cache space left, keep this kpsf around
            newsize = self.kpsf.nbytes + self._cache_size
            if self.cache_tiles and newsize < self.cache_max_size:
                self._cache[key] = self.kpsf.copy()
                self._cache_size = newsize
                self._cache_misses += 1

    def update(self, params):
        self.params = params

        # calculate the minimum supported real-space PSF
        self._min_support = np.ceil(self.get_support_size()).astype('int')
        self._min_support += self._min_support % 2
        self._setup_rvecs(self._min_support)
        self._min_rpsf = self.rpsf_func()

        # clean out the cache since it is no longer useful
        self._cache = {}
        self._cache_size = 0

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

    def get_support_size(self):
        return np.zeros(3)

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, ['_rx', '_ry', '_rz', '_rvecs', '_rlen'])
        cdd(odict, ['_kx', '_ky', '_kz', '_kvecs', '_klen'])
        cdd(odict, ['_fftn', '_ifftn', '_fftn_data', '_ifftn_data'])
        cdd(odict, ['rpsf', 'kpsf'])
        odict['_cache'] = {}
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.tile = Tile((0,0,0))
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
        params = self.params
        rt2 = np.sqrt(2)
        rhosq = self._rx**2 + self._ry**2
        arg = np.exp(-(rhosq)/(rt2*params[0])**2 - (self._rz/(rt2*params[1]))**2)
        return arg * (rhosq <= self.pr**2) * (np.abs(self._rz) <= self.pz)

    def get_support_size(self):
        self.pr = np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        return np.array([self.pz, self.pr, self.pr])

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

        rho = np.sqrt(self._rx**2 + self._ry**2) / coeff[0]
        z = self._rz / coeff[1]

        polycoeffs = self._psf_vecs.dot(coeff[2:]) + self._psf_mean
        poly = np.polynomial.polynomial.polyval2d(rho, z, polycoeffs.reshape(*self.poly_shape))
        return poly * np.exp(-rho**2) * np.exp(-z**2) * (rho <= self.pr/coeff[0]) * (np.abs(z) <= self.pz/coeff[1])

    def get_support_size(self):
        self.pr = 1.4*np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = 1.4*np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        return np.array([self.pz, self.pr, self.pr])

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

    def get_support_size(self):
        self.pr = 1.4*np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = 1.4*np.sqrt(-2*np.log(self.error)*self.params[1]**2)
        return np.array([self.pz, self.pr, self.pr])

