"""
A simple but pretty logging interface for configurable logs across all packages.
To use, simply import the base log and (maybe) tack on a child context:

.. code-block:: python

    from peri.logger import log
    clog = log.getChild("<child name>") # optional

    The possible options are:
    clog.{info,debug,warn,error,fatal}(...)

Call with:

.. code-block:: python

    log.info('something %r' % object)
    log.error('bad thing')

You can set the level of information displayed, for example, to only display
critical errors. The order is debug, info, warn, error, fatal

.. code-block:: python

    log.set_level('info')
"""
from future import standard_library
standard_library.install_aliases()
from future.utils import iteritems
from builtins import object

import logging
import logging.handlers
from io import StringIO
from contextlib import contextmanager

from peri import conf

class Logger(object):
    def __init__(self, verbosity='vvv', colorlogs=False, logtofile=False,
            logfilename=''):
        """
        Create a new logger class. Since the logging interface is actually global,
        any new logs will create even more clutter on the screen. Therefore,
        only create one! (as is created at the bottom of this file)
        """
        self.log = logging.getLogger('peri')
        self.log.setLevel(1)
        self.handlers = {}

        self.verbosity = sanitize(verbosity)
        self.logfilename = logfilename
        level = v2l.get(verbosity, 'info')
        form  = v2f.get(verbosity, 'standard')
        color = 'console-color' if colorlogs else 'console-bw'

        if logtofile:
            self.add_handler(
                name='rotating-log', level=level, formatter=form,
                filename=self.logfilename
            )
        self.add_handler(name=color, level=level, formatter=form)

    def get_handler(self, name='console-color'):
        return self.handlers.get(name)

    def get_handlers(self, names=None):
        if names is None:
            names = list(self.handlers.keys())
        names = listify(names)
        return [self.get_handler(name) for name in names]

    def set_level(self, level='info', handlers=None):
        """
        Set the logging level (which types of logs are actually printed / recorded)
        to one of ['debug', 'info', 'warn', 'error', 'fatal'] in that order
        of severity
        """
        for h in self.get_handlers(handlers):
            h.setLevel(levels[level])

    def set_formatter(self, formatter='standard', handlers=None):
        """
        Set the text format of messages to one of the pre-determined forms,
        one of ['quiet', 'minimal', 'standard', 'verbose']
        """
        for h in self.get_handlers(handlers):
            h.setFormatter(logging.Formatter(formatters[formatter]))

    def add_handler(self, name='console-color', level='info', formatter='standard', **kwargs):
        """
        Add another handler to the logging system if not present already.
        Available handlers are currently: ['console-bw', 'console-color', 'rotating-log']
        """
        # make sure the the log file has a name
        if name == 'rotating-log' and 'filename' not in kwargs:
            kwargs.update({'filename': self.logfilename})

        # make sure the the log file has a name
        if name == 'stringio' and 'stringio' not in kwargs:
            kwargs.update({'stringio': StringIO.StringIO()})

        handler = types[name](**kwargs)
        self.add_handler_raw(handler, name, level=level, formatter=formatter)

    def add_handler_raw(self, handler, name, level='info', formatter='standard'):
        if name in self.handlers:
            return

        handler.setLevel(levels[level])
        handler.setFormatter(logging.Formatter(formatters[formatter]))
        self.log.addHandler(handler)
        self.handlers[name] = handler

    def remove_handler(self, name):
        """
        Remove handler from the logging system if present already.
        Available handlers are currently: ['console-bw', 'console-color', 'rotating-log']
        """
        if name in self.handlers:
            self.log.removeHandler(self.handlers[name])

    @contextmanager
    def noformat(self):
        """ Temporarily do not use any formatter so that text printed is raw """
        try:
            formats = {}
            for h in self.get_handlers():
                formats[h] = h.formatter
            self.set_formatter(formatter='quiet')
            yield
        except Exception as e:
            raise
        finally:
            for k,v in iteritems(formats):
                k.formatter = v

    def set_verbosity(self, verbosity='vvv', handlers=None):
        """
        Set the verbosity level of a certain log handler or of all handlers.

        Parameters
        ----------
        verbosity : 'v' to 'vvvvv'
            the level of verbosity, more v's is more verbose

        handlers : string, or list of strings
            handler names can be found in ``peri.logger.types.keys()``
            Current set is::

                ['console-bw', 'console-color', 'rotating-log']
        """
        self.verbosity = sanitize(verbosity)
        self.set_level(v2l[verbosity], handlers=handlers)
        self.set_formatter(v2f[verbosity], handlers=handlers)

    def debug(self, *args, **kwargs):
        self.log.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.log.info(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.log.warn(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.log.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        self.log.exception(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self.log.critical(*args, **kwargs)

    def fatal(self, *args, **kwargs):
        self.log.fatal(*args, **kwargs)

    def getChild(self, name):
        return self.log.getChild(name)

BWHandler = logging.StreamHandler
LogHandler = logging.handlers.RotatingFileHandler

try:
    from peri.logger_colors import PygmentHandler
except ImportError as e:
    PygmentHandler = BWHandler

try:
    from loggr import LoggrHandler
except ImportError as e:
    LoggrHandler = BWHandler

types = {
    'stringio': BWHandler,
    'console-bw': BWHandler,
    'console-color': PygmentHandler,
    'rotating-log': LogHandler,
    'loggr': LoggrHandler,
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
    'verbose': '%(filename)s:%(lineno)d | %(asctime)s - %(levelname)s - %(name)s - %(message)s'
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

def listify(a):
    if not isinstance(a, (list, tuple)):
        return [a]
    return a

def sanitize(v):
    num = len(v)
    num = min(max([0, num]), 5)
    return 'v'*num

cf = conf.load_conf()
log = Logger(
    cf.get('verbosity'), cf.get('log-colors'),
    cf.get('log-to-file'), cf.get('log-filename')
)
