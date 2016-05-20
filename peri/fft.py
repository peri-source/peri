import pickle
import numpy as np

from multiprocessing import cpu_count

from peri.conf import get_wisdom
from peri.util import Tile
from peri.logger import log
log = log.getChild('fft')

try:
    import pyfftw
    hasfftw = True
except ImportError as e:
    log.warning(
        'FFTW not found, which can improve speed by 20x. '
        'Try `pip install pyfftw`.'
    )
    hasfftw = False
    
FFTW_PLAN_FAST = 'FFTW_ESTIMATE'
FFTW_PLAN_NORMAL = 'FFTW_MEASURE'
FFTW_PLAN_SLOW = 'FFTW_PATIENT'

class FFTBase(object):
    def __init__(self, shape=None, real=False):
        self.real = real
        self.shape = shape

    def kvectors(self, shape=None, *args, **kwargs):
        if shape is not None:
            self.set_shape(shape)
        return Tile(self.shape).kvectors(*args, **kwargs)

    def set_shape(self, shape):
        self.shape = shape

    def __getstate__(self):
        pass

    def __setstate__(self):
        pass

    def __getinitargs__(self):
        return (self.shape, self.real)


class FFTNPY(FFTBase):
    def __init__(self, shape=None, real=False):
        super(FFTNPY, self).__init__(shape=shape, real=real)

    def fft2(self, a):
        if self.real:
            return np.fft.rfft2(a)
        return np.fft.fft2(a)

    def ifft2(self, a, shape=None):
        if self.real:
            if self.shape is None:
                self.shape = shape
            if self.shape is None:
                raise AttributeError("IRFFTN shape unspecified, could be odd or even")
            return np.fft.irfft2(a, s=self.shape)
        return np.fft.ifft2(a)

    def fftn(self, a):
        if self.real:
            return np.fft.rfftn(a)
        return np.fft.fftn(a)

    def ifftn(self, a, shape=None):
        if self.real:
            if self.shape is None:
                self.shape = shape
            if self.shape is None:
                raise AttributeError("IRFFTN shape unspecified, could be odd or even")
            return np.fft.irfftn(a, s=self.shape)
        return np.fft.ifftn(a)


class FFTW(FFTBase):
    def __init__(self, shape=None, real=False, plan=FFTW_PLAN_NORMAL, threads=-1):
        """
        A faster FFT option, given that pyfftw is installed. It takes a `shape`
        and creates an object that has methods fftn, ifftn. `shape` can be any
        dimension.
        """
        self.threads = threads if threads > 0 else cpu_count()
        self.shape = None
        self.real = real
        self.plan = plan
        self.set_shape(shape)

    def set_shape(self, shape):
        if shape is None:
            return

        self.shape = shape
        if self.real:
            self._fftn_data = pyfftw.n_byte_align_empty(self.shape, 16, dtype='double')
            self._fftn = pyfftw.builders.rfftn(self._fftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=self.shape)

            oshape = self.fftn(np.zeros(shape)).shape
            self._ifftn_data = pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifftn = pyfftw.builders.irfftn(self._ifftn_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)

            self._fft2_data = pyfftw.n_byte_align_empty(self.shape, 16, dtype='double')
            self._fft2 = pyfftw.builders.rfft2(self._fft2_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=self.shape)

            oshape = self.fft2(np.zeros(shape)).shape
            self._ifft2_data = pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifft2 = pyfftw.builders.irfftn(self._ifft2_data, threads=self.threads,
                    planner_effort=self.fftw_planning_level, s=shape)
        else:
            self._fftn_data = pyfftw.n_byte_align_empty(shape, 16, dtype='complex')
            self._fftn_func = pyfftw.builders.fftn(self._fftn_data, overwrite_input=False,
                    planner_effort=self.plan, threads=self.threads)

            self._ifftn_data = pyfftw.n_byte_align_empty(shape, 16, dtype='complex')
            self._ifftn_func = pyfftw.builders.ifftn(self._ifftn_data, overwrite_input=False,
                    planner_effort=self.plan, threads=self.threads)

            self._fft2_data = pyfftw.n_byte_align_empty(shape, 16, dtype='complex')
            self._fft2_func = pyfftw.builders.fft2(self._fftn_data, overwrite_input=False,
                    planner_effort=self.plan, threads=self.threads)

            self._ifft2_data = pyfftw.n_byte_align_empty(shape, 16, dtype='complex')
            self._ifft2_func = pyfftw.builders.ifft2(self._ifft2_data, overwrite_input=False,
                    planner_effort=self.plan, threads=self.threads)

    def _exec(self, prefix, field):
        func = self.__dict__.get('_%s_func' % prefix)
        data = self.__dict__.get('_%s_data' % prefix)

        if (self.shape is None) or (data is None) or (field.shape != data.shape):
            self.set_shape(field.shape)
            func = self.__dict__.get('_%s_func' % prefix)
            data = self.__dict__.get('_%s_data' % prefix)

        data[:] = field
        func.execute()
        return func.get_output_array().copy()

    def fft2(self, a):
        return self._exec('fft2', a)

    def ifft2(self, a, shape=None):
        normalization = 1.0/a.shape
        return normalization * self._exec('ifft2', a)

    def fftn(self, a):
        return self._exec('fftn', a)

    def ifftn(self, a, shape=None):
        normalization = 1.0/a.shape
        return normalization * self._exec('ifftn', a)

    def load_wisdom(self, wisdomfile):
        try:
            pyfftw.import_wisdom(pickle.load(open(wisdomfile)))
        except IOError as e:
            self.save_wisdom(wisdomfile)
        self.wisdomfile = wisdomfile

    def save_wisdom(self, wisdomfile):
        pickle.dump(pyfftw.export_wisdom(), open(wisdomfile, 'wb'), protocol=-1)

    def __del__(self):
        wisdomfile = self.__dict__.get('wisdomfile')
        if wisdomfile:
            self.save_wisdom(wisdomfile)

    def __getinitargs__(self):
        return (self.shape, self.real, self.plan, self.threads)

if hasfftw:
    fft = FFTW()
    rfft = FFTW(real=True)

    wisdom = get_wisdom()
    fft.load_wisdom(wisdom)
    rfft.load_wisdom(wisdom)
else:
    fft = FFTNPY()
    rfft = FFTNPY(real=True)
