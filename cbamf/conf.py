import os
import json

from cbamf.const import CONF_FILE

default_conf = {
    "fftw_wisdom": os.path.join(os.path.expanduser("~"), ".fftw_wisdom.pkl")
}

def create_default_conf():
    with open(CONF_FILE, 'w') as f:
        json.dump(default_conf, f)

def load_conf():
    try:
        return json.load(open(CONF_FILE))
    except IOError as e:
        create_default_conf()
        return load_conf()

def get_wisdom():
    conf = load_conf()
    return conf['fftw_wisdom']
