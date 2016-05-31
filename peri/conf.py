import os
import json
import copy

CONF_FILE = os.path.join(os.path.expanduser("~"), ".peri.json")

default_conf = {
    "fftw_threads": -1,
    "fftw_wisdom": os.path.join(os.path.expanduser("~"), ".fftw_wisdom.pkl"),
    "log-filename": os.path.join(os.path.expanduser("~"), '.peri.log'),
    "log-to-file": False,
    "log-colors": False,
    "verbosity": 'vvv',
}

# Each of the variables above can be defined on the command line
def transform(v):
    return v.lower().replace('_', '-')

def read_environment():
    out = {}
    for k,v in os.environ.iteritems():
        if transform(k) in default_conf:
            out[transform(k)] = v
    return out

# variables also defined in the conf file
def create_default_conf():
    with open(CONF_FILE, 'w') as f:
        json.dump(default_conf, f)

def load_conf():
    """
    Load the configuration with the priority:
        1. environment variables
        2. configuration file
        3. defaults here
    """
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
