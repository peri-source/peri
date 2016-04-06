"""
A simple but pretty logging interface for configurable logs across all packages.
To use, simply import the base log and (maybe) tack on a child context:
    
    from logger import log
    log = log.getChild("<child name>")

    log.{info,debug,warn,error}(...)
"""
import os
import re
import sys

from pygments.lexer import RegexLexer, include
from pygments.token import Punctuation, Text, Comment, Keyword, Name, String, \
     Generic, Operator, Number, Whitespace, Literal, Error, Token
from pygments import highlight
from pygments.formatters import get_formatter_by_name
from pygments.style import Style

import logging
import logging.handlers

from cbamf import conf

FILELEVEL = logging.DEBUG

class LogStyle(Style):
    background_color = "#000000"
    highlight_color = "#222222"
    default_style = "#cccccc"

    styles = {
        Token:                     "#cccccc",
        Whitespace:                "",
        Comment:                   "#000080",
        Comment.Preproc:           "",
        Comment.Special:           "bold #2BB537",

        Keyword:                   "#cdcd00",
        Keyword.Declaration:       "#00cd00",
        Keyword.Namespace:         "#cd00cd",
        Keyword.Pseudo:            "bold #00cd00",
        Keyword.Type:              "#00cd00",

        Operator:                  "#3399cc",
        Operator.Word:             "#cdcd00",

        Name:                      "",
        Name.Class:                "#00cdcd",
        Name.Builtin:              "#cd00cd",
        Name.Exception:            "bold #666699",
        Name.Variable:             "#00cdcd",

        String:                    "#cd0000",
        Number:                    "#cd00cd",

        Punctuation:               "nobold #FFF",
        Generic.Heading:           "nobold #FFF",
        Generic.Subheading:        "#800080",
        Generic.Deleted:           "nobold #cd3",
        Generic.Inserted:          "#00cd00",
        Generic.Error:             "bold #FF0000",
        Generic.Emph:              "bold #FFFFFF",
        Generic.Strong:            "bold #FFFFFF",
        Generic.Prompt:            "bold #3030F0",
        Generic.Output:            "#888",
        Generic.Traceback:         "bold #04D",

        Error:                     "bg:#FF0000 bold #FFF"
    }


class LogLexer(RegexLexer):
    name = 'Logging.py Logs'
    aliases = ['log']
    filenames = ['*.log']
    mimetypes = ['text/x-log']

    flags = re.VERBOSE
    _logger = r'-\s(pipeline)(\.([a-z._\-0-9]+))*\s-'
    _uuid   = r"([A-Z]{2}_[0-9]{12}_[0-9]{3}-and-[A-Z]{2}_[0-9]{12}_[0-9]{3}-[0-9]{5,})"
    _kimid  = r"((?:[_a-zA-Z][_a-zA-Z0-9]*?_?_)?[A-Z]{2}_[0-9]{12}(?:_[0-9]{3})?)"
    _path   = r'(?:[a-zA-Z0-9_-]{0,}/{1,2}[a-zA-Z0-9_\.-]+)+'
    _debug  = r'DEBUG'
    _info   = r'INFO'
    _pass   = r'PASS'
    _warn   = r'WARNING'
    _error  = r'ERROR'
    _crit   = r'CRITICAL'
    _date   = r'\d{4}-\d{2}-\d{2}'
    _time   = r'\d{2}:\d{2}:\d{2},\d{3}'
    _ws     = r'(?:\s|//.*?\n|/[*].*?[*]/)+'
    _json   = r'{.*}'

    tokens = {
        'whitespace': [
            (_ws, Text),
            (r'\n', Text),
            (r'\s+', Text),
            (r'\\\n', Text),
            (r'\s-\s', Text)
        ],
        'root': [
            include('whitespace'),
            (_uuid, Comment.Special),
            (_kimid, Generic.Prompt),
            (_logger, Generic.Emph),
            (_date, Generic.Output),
            (_time, Generic.Output),
            (_path, Generic.Subheading),
            (_json, Generic.Deleted),
            (_warn, Generic.Strong),
            (_info, Generic.Traceback),
            (_error, Generic.Error),
            (_pass, Keyword.Pseudo),
            (_crit, Error),
            (r'[0-9]+', Generic.Heading),
            ('[a-zA-Z_][a-zA-Z0-9_]*', Generic.Heading),
            (r'[{}`()\"\[\]@.,:-\\]', Punctuation),
            (r'[~!%^&*+=|?:<>/-]', Punctuation),
            (r"'", Punctuation)
        ]
    }


def pygmentize(text, formatter='256', outfile=sys.stdout, style=LogStyle):
    fmtr = get_formatter_by_name(formatter, style=style)
    highlight(text, lexer, fmtr, outfile)

class PygmentHandler(logging.StreamHandler):
    """ A beanstalk logging handler """
    def __init__(self):
        super(PygmentHandler,self).__init__()

    def emit(self,record):
        """ Send the message """
        err_message = self.format(record)
        pygmentize(err_message)

BWHandler = logging.StreamHandler
LogHandler = logging.handlers.RotatingFileHandler

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

def get_logger(name='console-color'):
    for h in log.handlers:
        if isinstance(h, types[name]):
            return h
    return None

def set_level(level='info', handlers=None):
    if handlers is None:
        handlers = log.handlers
    for h in handlers:
        h.setLevel(levels[level])

def set_formatter(formatter='standard', handlers=None):
    if handlers is None:
        handlers = log.handlers
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

lexer = LogLexer()
log = logging.getLogger('cbamf')
log.setLevel(1)

conf = conf.load_conf()
verbosity = sanitize(conf.get('verbosity'))
level = v2l.get(verbosity, 'info')
form  = v2f.get(verbosity, 'standard')
color = 'console-color' if conf.get('log-colors') else 'console-bw'

if conf.get('log-to-file'):
    add_handler(log, name='rotating-log', level=level, formatter=form)
add_handler(log, name=color, level=level, formatter=form)
