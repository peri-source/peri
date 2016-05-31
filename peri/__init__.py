# always monkey-patch the matplotlibrc for peri!
from peri.viz import base

# trackpy loggers interfere with our own. stop that here.
try:
    import trackpy
    import logging

    if len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])
except ImportError as e:
    pass

__version__ = "0.1.1"
