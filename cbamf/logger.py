"""
A simple but pretty logging interface for configurable logs across all packages.
To use, simply import the base log and (maybe) tack on a child context:
    
    from cbamf.logger import log
    log = log.getChild("<child name>") # optional
    
    The possible options are:
    log.{info,debug,warn,error,fatal}(...)
    Call with:
    log.info('something'), log.error('bad thing')
    You can set the level of information displayed, for example, to only 
    display critical errors. 
    The order is debug, info, warn, error, fatal
"""
import logging
import logging.handlers

from cbamf import conf

BWHandler = logging.StreamHandler
LogHandler = logging.handlers.RotatingFileHandler

try:
    from cbamf.logger_colors import PygmentHandler
except ImportError as e:
    PygmentHandler = BWHandler

types = {
    'console-bw': BWHandler,
    'console-color': PygmentHandler,
    'rotating-log': LogHandler
}

levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'fatal': logging.FATAL
}

formatters = {
    'quiet': '%(message)s',
    'minimal': '%(levelname)s:%(name)s - %(message)s',
    'standard': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    'verbose': '%(filename)s:%(lineno)d _ %(asctime)s - %(levelname)s - %(name)s - %(message)s'
}

v2l = {
    '':     'fatal',
    'v':    'error',
    'vv':   'warn',
    'vvv':  'info',
    'vvvv': 'debug',
    'vvvvv':'debug'
}

v2f = {
    '':      'quiet',
    'v':     'minimal',
    'vv':    'minimal',
    'vvv':   'standard',
    'vvvv':  'standard',
    'vvvvv': 'verbose'
}

def get_handler(name='console-color'):
    for h in log.handlers:
        if isinstance(h, types[name]):
            return h
    return None

def set_level(level='info', handlers=None):
    if handlers is None:
        handlers = log.handlers
    else:
        handlers = [get_handler(h) for h in handlers]

    for h in handlers:
        h.setLevel(levels[level])

def set_formatter(formatter='standard', handlers=None):
    if handlers is None:
        handlers = log.handlers
    else:
        handlers = [get_handler(h) for h in handlers]

    for h in handlers:
        h.setFormatter(formatters[formatter])

def add_handler(log, name='console-color', level='info', formatter='standard', **kwargs):
    handler = types[name](**kwargs)
    handler.setLevel(levels[level])
    handler.setFormatter(logging.Formatter(formatters[formatter]))
    log.addHandler(handler)

def sanitize(v):
    num = len(v)
    num = min(max([0, num]), 5)
    return 'v'*num

def initialize_loggers(log):
    cf = conf.load_conf()
    verbosity = sanitize(cf.get('verbosity'))
    level = v2l.get(verbosity, 'info')
    form  = v2f.get(verbosity, 'standard')
    color = 'console-color' if cf.get('log-colors') else 'console-bw'

    if cf.get('log-to-file'):
        add_handler(log, name='rotating-log', level=level, formatter=form)
    add_handler(log, name=color, level=level, formatter=form)

log = logging.getLogger('cbamf')
log.setLevel(1)
initialize_loggers(log)
