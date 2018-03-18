# trackpy loggers interfere with our own. stop that here.
try:
    import trackpy
    import logging

    if len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])
except ImportError as e:
    pass

import pkg_resources
__version__ = pkg_resources.require("peri")[0].version
