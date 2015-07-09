import sys
import numpy as np
import code, traceback, signal

#=============================================================================
# Tiling utilities
#=============================================================================
def amin(a, b):
    return np.vstack([a, b]).min(axis=0)

def amax(a, b):
    return np.vstack([a, b]).max(axis=0)

class Tile(object):
    def __init__(self, left, right=None, mins=None, maxs=None):
        if right is None:
            right = left
            left = np.zeros(len(left), dtype='int')

        if not hasattr(left, '__iter__'):
            left = np.array([left]*3, dtype='int')

        if not hasattr(right, '__iter__'):
            right = np.array([right]*3, dtype='int')

        if mins is not None:
            if not hasattr(mins, '__iter__'):
                mins = np.array([mins]*3)
            left = amax(left, mins)

        if maxs is not None:
            if not hasattr(maxs, '__iter__'):
                maxs = np.array([maxs]*3)
            right = amin(right, maxs)

        l, r = left, right
        self.l = np.array(l)
        self.r = np.array(r)
        self.bounds = (l, r)
        self.shape = self.r - self.l
        self.slicer = np.s_[l[0]:r[0], l[1]:r[1], l[2]:r[2]]

def cdd(d, k):
    """ Conditionally delete key (or list of keys) 'k' from dict 'd' """
    if not isinstance(k, list):
        k = [k]
    for i in k:
        if d.has_key(i):
            d.pop(i)

#=============================================================================
# Progress bar
#=============================================================================
class ProgressBar(object):
    def __init__(self, num, label='Progress', value=0, dobar=True, symbol='='):
        self.num = num
        self.value = value
        self.percent = 0

        self.label = label
        self.dobar = dobar
        self.symbol = symbol

        # the space available for the progress bar is
        # 79 (size of screen) - (label) - 3 (number) - 2 ([]) - 2 (space)
        self._bar_size = 79 - len(self.label) - 3 - 2 - 2
        self._formatstr = '{} [{: <'+str(self._bar_size)+'}] {: >3}\r'

        self._calc_precent()
        self.draw()

    def _calc_precent(self):
        self.percent = int(100 * self.value / self.num)

    def draw(self):
        bar = np.round(self.percent / 100. * self._bar_size)
        print self._formatstr.format(self.label, '='*bar, self.percent),
        sys.stdout.flush()

    def update(self, value=0):
        curr_percent = self.percent
        self.value = value
        self._calc_precent()

        if curr_percent != self.percent:
            self.draw()

    def end(self):
        print '\r\n'

#=============================================================================
# debugging / python interpreter / logging
#=============================================================================
def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

def listen():
    # register handler
    signal.signal(signal.SIGUSR1, debug)
