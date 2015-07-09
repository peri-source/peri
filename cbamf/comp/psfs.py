import os
import cPickle as pickle
import numpy as np

try:
    import pyfftw
    from pyfftw.builders import fftn, ifftn
    hasfftw = True
except ImportError as e:
    print "*WARNING* pyfftw not found, switching to numpy.fft (20x slower)"
    hasfftw = False

from multiprocessing import cpu_count

from cbamf.util import Tile, cdd
from cbamf.conf import get_wisdom

if hasfftw:
    WISDOM_FILE = get_wisdom()
    def save_wisdom():
        pickle.dump(pyfftw.export_wisdom(), open(WISDOM_FILE, 'w'), protocol=-1)

    try:
        with open(WISDOM_FILE) as wisdom:
            pyfftw.import_wisdom(pickle.load(open(WISDOM_FILE)))
    except IOError as e:
        save_wisdom()

FFTW_PLAN_FAST = 'FFTW_ESTIMATE'
FFTW_PLAN_NORMAL = 'FFTW_MEASURE'
FFTW_PLAN_SLOW = 'FFTW_PATIENT'

class PSF(object):
    _fourier_space = True

    def __init__(self, params, shape, fftw_planning_level=FFTW_PLAN_NORMAL, threads=-1):
        self._cache = {}
        self.shape = shape
        self.params = np.array(params).astype('float')

        self.threads = threads if threads > 0 else cpu_count()
        self.fftw_planning_level = fftw_planning_level

        self.tile = Tile((0,0,0))
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

    def _setup_rvecs(self, centered=False):
        sp = self.tile.shape
        mx = np.max(sp)

        if not centered:
            rz = 2*sp[0]*np.fft.fftfreq(sp[0])[:,None,None]
            ry = 2*sp[1]*np.fft.fftfreq(sp[1])[None,:,None]
            rx = 2*sp[2]*np.fft.fftfreq(sp[2])[None,None,:]
        else:
            rz = (np.arange(sp[0], dtype='float')[:,None,None] - sp[0]/2)
            ry = (np.arange(sp[1], dtype='float')[None,:,None] - sp[1]/2)
            rx = (np.arange(sp[2], dtype='float')[None,None,:] - sp[2]/2)

        self._rx, self._ry, self._rz = rx, ry, rz
        self._rvecs = np.rollaxis(np.array(np.broadcast_arrays(rz,ry,rx)), 0, 4)
        self._rlen = np.sqrt(rx**2 + ry**2 + rz**2)

    def _set_tile_precalc(self):
        pass

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
            return (self._ifftn.get_output_array() / self._fftn_data.size).copy()
        else:
            return np.fft.ifftn(arr)

    def normalize_kpsf(self):
        self.kpsf /= np.real(self.kpsf[0,0,0])
        pass

    def set_tile(self, tile):
        if any(self.tile.shape != tile.shape):
            self.tile = tile
            self._setup_ffts()

            key = tuple(self.tile.shape)
            if key in self._cache:
                self.kpsf = self._cache[key]
            else:
                if self._fourier_space:
                    self._setup_kvecs()
                else:
                    self._setup_rvecs()

                self._set_tile_precalc()
                self.update(self.params, clearcache=False)
                # TODO - caching not working, figure it out
                #self._cache[key] = self.kpsf.copy()

    def update(self, params, clearcache=True):
        self.params = params

        if self._fourier_space:
            self.kpsf = self.kpsf_func(self.params)
        else:
            self.rpsf = self.rpsf_func(self.params)
            self.kpsf = self.fftn(self.rpsf)

        self.normalize_kpsf()

        if clearcache:
            self._cache = {}

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

    def __del__(self):
        save_wisdom()

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

class AnisotropicGaussian(PSF):
    _fourier_space = False

    def __init__(self, params, shape, error=1.0/255, *args, **kwargs):
        self.error = error
        super(AnisotropicGaussian, self).__init__(*args, params=params, shape=shape, **kwargs)

    def _set_tile_precalc(self):
        self.pr = np.sqrt(-2*np.log(self.error)*self.params[0]**2)
        self.pz = np.sqrt(-2*np.log(self.error)*self.params[1]**2)

    def rpsf_func(self, params):
        rt2 = np.sqrt(2)
        rhosq = self._rx**2 + self._ry**2
        arg = np.exp(-(rhosq)/(rt2*params[0])**2 - (self._rz/(rt2*params[1]))**2)
        return arg * (rhosq <= self.pr**2) * (np.abs(self._rz) <= self.pz)

    def get_support_size(self):
        self._set_tile_precalc()
        return np.array([self.pz, self.pr, self.pr])

class AnisotropicGaussianKSpace(PSF):
    def __init__(self, params, shape, support_factor=1.4, *args, **kwargs):
        """ Do not set support_factor to an integral value """
        self.support_factor = support_factor
        super(AnisotropicGaussianKSpace, self).__init__(*args, params=params, shape=shape, **kwargs)

    def kpsf_func(self, params):
        return np.exp(-(self._kx*params[0])**2 - (self._ky*params[0])**2 - (self._kz*params[1])**2)

    def get_support_size(self):
        return self.support_factor*np.array([self.params[1], self.params[0], self.params[0]])

class GaussianPolynomialPCA(PSF):
    _fourier_space = False

    def __init__(self, cov_mat_file, mean_mat_file, shape, gaussian=(1,1), components=5, *args, **kwargs):
        self.cov_mat_file = cov_mat_file
        self.mean_mat_file = mean_mat_file
        self.comp = components

        self._setup_from_files()
        params0 = np.hstack([gaussian, np.zeros(self.comp)])

        super(GaussianPolynomialPCA, self).__init__(*args, params=params0, shape=shape, **kwargs)

    def _set_tile_precalc(self):
        # normalize the vectors to be small and managable
        mx = np.max(self.tile.shape)
        self._rx = self._rx / mx
        self._ry = self._ry / mx
        self._rz = self._rz / mx

    def _setup_from_files(self):
        covm = np.load(self.cov_mat_file)
        mean = np.load(self.mean_mat_file)
        self.poly_shape = (np.round(np.sqrt(mean.shape[0])),)*2

        vals, vecs = np.linalg.eig(covm)
        self._psf_vecs = np.real(vecs[:,:self.comp])
        self._psf_mean = np.real(mean)

        # TODO -- proper calculation?
        #self._polys = [np.polynomial.polynomial.polyval2d(
        #    rho, z, (self._psf_vecs[:,i] + self._psf_mean).reshape(*self.poly_shape)
        #) for i in xrange(self.comp)]
        #poly = (self._polys * self.params[2:]).sum(axis=-1)

    def rpsf_func(self):
        coeff = self.params
        rvec = self._rvecs

        rho = np.sqrt(rvec[...,1]**2 + rvec[...,2]**2) / coeff[0]
        z = rvec[...,0] / coeff[1]

        polycoeffs = self._psf_vecs.dot(coeff[2:]) + self._psf_mean
        poly = np.polynomial.polynomial.polyval2d(rho, z, polycoeffs.reshape(*self.poly_shape))
        return poly * np.exp(-rho**2) * np.exp(-z**2)

    def get_support_size(self):
        self._set_tile_precalc()
        return np.array([self.pz, self.pr, self.pr])

