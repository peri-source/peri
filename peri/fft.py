import atexit
import pickle
import numpy as np

from multiprocessing import cpu_count

from peri import conf
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

def load_wisdom(wisdomfile):
    if wisdomfile is None:
        return

    try:
        pyfftw.import_wisdom(pickle.load(open(wisdomfile)))
    except IOError as e:
        log.warn("No wisdom present, generating some at %r" % wisdomfile)
        save_wisdom(wisdomfile)

def save_wisdom(wisdomfile):
    if wisdomfile is None:
        return

    if wisdomfile:
        pickle.dump(
            pyfftw.export_wisdom(), open(wisdomfile, 'wb'),
            protocol=-1
        )

if hasfftw:
    _var = conf.load_conf()
    effort = _var['fftw-planning-effort']
    threads = _var['fftw-threads']
    threads = threads if threads > 0 else cpu_count()

    fftkwargs = {
        'planner_effort': effort,
        'threads': threads,
        'overwrite_input': False,
        'auto_align_input': True,
        'auto_contiguous': True
    }

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)

    fft = pyfftw.interfaces.numpy_fft
    load_wisdom(conf.get_wisdom())

    @atexit.register
    def goodbye():
        save_wisdom(conf.get_wisdom())
else:
    fftkwargs = {}
    fft = np.fft
