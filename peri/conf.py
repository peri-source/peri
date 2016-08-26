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
    default = os.path.join(os.path.expanduser("~"), ".peri.json")
    return os.environ.get('PERI_CONF_FILE', default)

# Each of the variables above can be defined on the command line
def transform(v):
    return v.lower().replace('_', '-').replace('peri-', '')

def read_environment():
    out = {}
    for k,v in os.environ.iteritems():
        if transform(k) in default_conf:
            out[transform(k)] = v
    return out

# variables also defined in the conf file
def create_default_conf():
    with open(get_conf_filename(), 'w') as f:
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
