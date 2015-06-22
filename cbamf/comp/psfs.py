import os
import pickle
import numpy as np

import pyfftw
from pyfftw.builders import fftn, ifftn
from multiprocessing import cpu_count

from cbamf.util import Tile

WISDOM_FILE = os.path.join(os.path.expanduser("~"), ".fftw_wisdom.pkl")

def save_wisdom():
    pickle.dump(pyfftw.export_wisdom(), open(WISDOM_FILE, 'w'))

try:
    with open(WISDOM_FILE) as wisdom:
        pyfftw.import_wisdom(pickle.load(open(WISDOM_FILE)))
except IOError as e:
    save_wisdom()

FFTW_PLAN_FAST = 'FFTW_ESTIMATE'
FFTW_PLAN_NORMAL = 'FFTW_MEASURE'
FFTW_PLAN_SLOW = 'FFTW_PATIENT'

class PSF(object):
    def __init__(self, params, shape, fftw_planning_level=FFTW_PLAN_NORMAL, threads=-1):
        self.shape = shape
        self.params = params

        self.threads = threads if threads > 0 else cpu_count()
        self.fftw_planning_level = fftw_planning_level

        self.tile = Tile((0,0,0))
        self.set_tile(Tile(self.shape))

    def _setup_ffts(self):
        self._fftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
        self._ifftn_data = pyfftw.n_byte_align_empty(self.tile.shape, 16, dtype='complex')
        self._fftn = fftn(self._fftn_data, overwrite_input=True,
                planner_effort=self.fftw_planning_level, threads=self.threads)
        self._ifftn = ifftn(self._ifftn_data, overwrite_input=True,
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
            rz = 2*sp[0]*np.fft.fftfreq(sp[0])[:,None,None] / mx
            ry = 2*sp[1]*np.fft.fftfreq(sp[1])[None,:,None] / mx
            rx = 2*sp[2]*np.fft.fftfreq(sp[2])[None,None,:] / mx
        else:
            rz = (np.arange(sp[0], dtype='float')[:,None,None] - sp[0]/2)/mx
            ry = (np.arange(sp[1], dtype='float')[None,:,None] - sp[1]/2)/mx
            rx = (np.arange(sp[2], dtype='float')[None,None,:] - sp[2]/2)/mx

        self._rx, self._ry, self._rz = rx, ry, rz
        self._rvecs = np.rollaxis(np.array(np.broadcast_arrays(rz,ry,rx)), 0, 4)
        self._rlen = np.sqrt(rx**2 + ry**2 + rz**2)

    def set_tile(self, tile):
        if any(self.tile.shape != tile.shape):
            self.tile = tile
            self._setup_kvecs()
            self._setup_ffts()
            self.update(self.params)

    def update(self, params):
        self.params = params
        self.kpsf = self.psf_func(self.params)

    def execute(self, field):
        if any(field.shape != self.tile.shape):
            raise AttributeError("Field passed to PSF incorrect shape")

        if (not pyfftw.is_n_byte_aligned(self._fftn_data, 16) or
            not pyfftw.is_n_byte_aligned(self._ifftn_data, 16)):
            raise AttributeError("FFT arrays became misaligned")

        if not np.iscomplex(field.ravel()[0]):
            self._fftn_data[:] = field
            self._fftn.execute()
            infield = self._fftn.get_output_array()
        else:
            infield = field

        self._ifftn_data[:] = infield * self.kpsf / self._fftn_data.size
        self._ifftn.execute()

        return np.real(self._ifftn.get_output_array())

    def get_params(self):
        return self.params

    def __del__(self):
        save_wisdom()

class AnisotropicGaussian(PSF):
    def __init__(self, params, shape, *args, **kwargs):
        super(AnisotropicGaussian, self).__init__(*args, params=params, shape=shape, **kwargs)

    def psf_func(self, params):
        return np.exp(-(self._kx*params[0])**2 - (self._ky*params[0])**2 - (self._kz*params[1])**2)

class IsotropicDisc(PSF):
    def __init__(self, params, shape, *args, **kwargs):
        super(IsotropicDisc, self).__init__(*args, **kwargs)

    def psf_func(self, params):
        return (1.0 + np.exp(-params[1]*params[0])) / (1.0 + np.exp(params[1]*(self._klen - params[0])))

class GaussianPolynomialPCA(PSF):
    def __init__(self, cov_mat_file, mean_mat_file, shape, gaussian=(1,1), components=5, *args, **kwargs):
        self.cov_mat_file = cov_mat_file
        self.mean_mat_file = mean_mat_file
        self.comp = components

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

    def _evaluate_real_space(self):
        coeff = self.params
        rvec = self._rvecs

        rho = np.sqrt(rvec[...,1]**2 + rvec[...,2]**2) / coeff[0]
        z = rvec[...,0] / coeff[1]

        polycoeffs = self._psf_vecs.dot(coeff[2:]) + self._psf_mean
        poly = np.polynomial.polynomial.polyval2d(rho, z, polycoeffs.reshape(*self.poly_shape))
        return poly * np.exp(-rho**2) * np.exp(-z**2)

    def _evaluate_fourier_space(self, arr):
        self._fftn_data[:] = arr
        self._fftn.execute()
        return self._fftn.get_output_array()

    def set_tile(self, tile):
        if any(self.tile.shape != tile.shape):
            self.tile = tile
            self._setup_rvecs(centered=False)
            self._setup_kvecs()
            self._setup_ffts()
            self.update(self.params)

    def update(self, params):
        self.params = params
        self.rpsf = self._evaluate_real_space()
        self.kpsf = self._evaluate_fourier_space(self.rpsf)

