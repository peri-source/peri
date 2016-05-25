import atexit
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

        self.shape = tuple(shape)
        if self.real:
            k = {
                'threads': self.threads,
                'planner_effort': self.plan,
                's': self.shape,
            }

            # FIXME -- fft2 / ifft2 in this case. can't figure it out.
            self._fftn_data = pyfftw.n_byte_align_empty(self.shape, 16, dtype='double')
            self._fftn_func = pyfftw.builders.rfftn(self._fftn_data, **k)

            oshape = self.fftn(np.zeros(shape)).shape
            self._ifftn_data = pyfftw.n_byte_align_empty(oshape, 16, dtype='complex')
            self._ifftn_func = pyfftw.builders.irfftn(self._ifftn_data, **k)
        else:
            k = {
                'threads': self.threads,
                'planner_effort': self.plan,
                'overwrite_input': False
            }
            def _alloc(name):
                data = pyfftw.n_byte_align_empty(shape, 16, dtype='complex')
                func = getattr(pyfftw.builders, name)(data, **k)
                return data, func

            # FIXME -- too much memory?
            self._fftn_data, self._fftn_func = _alloc('fftn')
            self._ifftn_data, self._ifftn_func = _alloc('ifftn')
            self._fft2_data, self._fft2_func = _alloc('fft2')
            self._ifft2_data, self._ifft2_func = _alloc('ifft2')

    def _prefix_to_obj(self, prefix):
        func = self.__dict__.get('_%s_func' % prefix)
        data = self.__dict__.get('_%s_data' % prefix)
        return func, data

    def _exec(self, prefix, field):
        func, data = self._prefix_to_obj(prefix)

        if (self.shape is None) or (data is None) or (field.shape != data.shape):
            self.set_shape(field.shape)
            func, data = self._prefix_to_obj(prefix)

        data[:] = field
        func.execute()
        return func.get_output_array().copy()

    def fft(self, a):
        return self._exec('fft', a)

    def ifft(self, a, shape=None):
        normalization = 1.0/np.prod(self.shape[2:])
        return normalization * self._exec('ifft', a)

    def fft2(self, a):
        return self._exec('fft2', a)

    def ifft2(self, a, shape=None):
        normalization = 1.0/np.prod(self.shape[1:])
        return normalization * self._exec('ifft2', a)

    def fftn(self, a):
        return self._exec('fftn', a)

    def ifftn(self, a, shape=None):
        normalization = 1.0/np.prod(self.shape)
        return normalization * self._exec('ifftn', a)

    def load_wisdom(self, wisdomfile):
        if wisdomfile is None:
            return

        try:
            pyfftw.import_wisdom(pickle.load(open(wisdomfile)))
        except IOError as e:
            log.warn("No wisdom present, generating some at %r" % wisdomfile)
            self.save_wisdom(wisdomfile)
        self.wisdomfile = wisdomfile

    def save_wisdom(self, wisdomfile):
        if wisdomfile is None:
            return

        if wisdomfile:
            pickle.dump(
                pyfftw.export_wisdom(), open(wisdomfile, 'wb'),
                protocol=-1
            )

    def __getinitargs__(self):
        return (self.shape, self.real, self.plan, self.threads)

if hasfftw:
    fft = FFTW()
    rfft = FFTW(real=True)

    wisdom = get_wisdom()
    fft.load_wisdom(wisdom)
    rfft.load_wisdom(wisdom)

    @atexit.register
    def goodbye():
        fft.save_wisdom(wisdom)
else:
    fft = FFTNPY()
    rfft = FFTNPY(real=True)
