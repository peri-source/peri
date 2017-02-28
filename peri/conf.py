"""

The default values for the global package configuration are provided in ``peri.conf.default_conf``.
The configuration variables are described here:

========================= ====================== =============================================================
Variable name             Default value          Description
========================= ====================== =============================================================
``fftw-threads``          -1                     Number of threads for fftw to use, -1 indicates all available
``fftw-planning-effort``  ``FFTW_MEASURE``       One of (``FFTW_ESTIMATE``, ``FFTW_MEASURE``, ``FFTW_PATIENT``)
                                                 where options to the right take longer the first time but
                                                 are faster in subsequent evaluations.
``fftw-wisdom``           ``~/.peri-wisdom.pkl`` Location of file in which to store wisdom. Wisdom is the results
                                                 of fftw benchmarking itself, allowing it to run as fast as possible.
``log-filename``          ``~/.peri.log``        Name of file for logging.
``log-to-file``           False                  Whether or not to actually save logs to a file as well
``log-colors``            False                  Display logs in color (supported by xterm256)
``verbosity``             vvv                    Level of verbosity for logs, the more v's the more verbose
========================= ====================== =============================================================
"""
from future.utils import iteritems

import os
import json
import copy

default_conf = {
    "fftw-threads": -1,
    "fftw-planning-effort": "FFTW_MEASURE",
    "fftw-wisdom": os.path.join(os.path.expanduser("~"), ".peri-wisdom.pkl"),
    "log-filename": os.path.join(os.path.expanduser("~"), '.peri.log'),
    "log-to-file": False,
    "log-colors": False,
    "verbosity": 'vvv',
}

def get_conf_filename():
    """
    The configuration file either lives in ~/.peri.json or is specified on the
    command line via the environment variables PERI_CONF_FILE
    """
    default = os.path.join(os.path.expanduser("~"), ".peri.json")
    return os.environ.get('PERI_CONF_FILE', default)

def transform(v):
    """
    Translate environment variables to ones corresponding to keys in the
    configuration.  In particular, env variables may be made with
    "PERI_"+key_name: fftw-threads = PERI_FFTW_THREADS. Each env var
    is later checked to see if it has to do with PERI
    """

    return v.lower().replace('_', '-').replace('peri-', '')

def read_environment():
    """ Read all environment variables to see if they contain PERI """
    out = {}
    for k,v in iteritems(os.environ):
        if transform(k) in default_conf:
            out[transform(k)] = v
    return out

def create_default_conf():
    """ Dump the default_conf to the configuration file """
    with open(get_conf_filename(), 'w') as f:
        json.dump(default_conf, f)

def load_conf():
    """
    Load the configuration with the priority:
        1. environment variables
        2. configuration file
        3. defaults here (default_conf)
    """
    try:
        conf = copy.copy(default_conf)
        conf.update(json.load(open(get_conf_filename())))
        conf.update(read_environment())
        return conf
    except IOError as e:
        create_default_conf()
        return load_conf()

def get_wisdom():
    conf = load_conf()
    return conf['fftw-wisdom']

def get_logfile():
    conf = load_conf()
    return conf['logfile']
