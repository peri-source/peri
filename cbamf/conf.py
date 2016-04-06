import os
import json
import copy

from cbamf.const import CONF_FILE

default_conf = {
    "fftw_wisdom": os.path.join(os.path.expanduser("~"), ".fftw_wisdom.pkl"),
    "log-filename": os.path.join(os.path.expanduser("~"), '.cbamf.log'),
    "log-to-file": False,
    "log-colors": False,
    "verbosity": 'vvv',
}

def transform(v):
    return v.lower().replace('_', '-')

def read_environment():
    out = {}
    for k,v in os.environ.iteritems():
        if transform(k) in default_conf:
            out[transform(k)] = v
    return out

def create_default_conf():
    with open(CONF_FILE, 'w') as f:
        json.dump(default_conf, f)

def load_conf():
    try:
        conf = copy.copy(default_conf)
        conf.update(json.load(open(CONF_FILE)))
        conf.update(read_environment())
        return conf
    except IOError as e:
        create_default_conf()
        return load_conf()

def get_wisdom():
    conf = load_conf()
    return conf['fftw_wisdom']

def get_logfile():
    conf = load_conf()
    return conf['logfile']
