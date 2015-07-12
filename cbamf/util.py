import sys
import time
import datetime
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
    def __init__(self, num, label='Progress', value=0, screen=79,
            time_remaining=True, bar=True, bar_symbol='=', bar_caps='[]',
            bar_decimals=2):
        """
        ProgressBar class which creates a dynamic ASCII progress bar of two
        different varieties:
            1) A bar chart that looks like the following:
                Progress [================      ]  63.00%

            2) A simple number completed look:
                Progress :   17 / 289

        Parameters:
        -----------
        num : integer
            The number of tasks that need to be completed

        label : string [default: 'Progress']
            The label for this particular progress indicator,

        value : integer [default: 0]
            Starting value

        screen : integer [default: 79]
            Size the screen to use for the progress bar

        time_remaining : boolean [default: True]
            Display estimated time remaining

        bar : boolean [default: True]
            Whether or not to display the bar chart

        bar_symbol : char [default: '=']
            The character to use to fill in the bar chart

        bar_caps : string [default: '[]']
            Characters to use as the end caps of the.  The string will be split in
            half and each half put on a side of the chart

        bar_decimals: integer [default: 2]
            Number of decimal places to include in the _percentage
        """
        # TODO -- add estimated time remaining
        self.num = num
        self.value = value
        self._percent = 0
        self.time_remaining = time_remaining
        self._deltas = []

        self.label = label
        self.bar = bar
        self._bar_symbol = bar_symbol
        self._bar_caps = bar_caps
        self._decimals = bar_decimals
        self.screen = screen

        if len(self._bar_caps) % 2 != 0:
            raise AttributeError("End caps must be even number of symbols")

        if self.bar:
            # 3 digit _percent + decimal places + '.'
            self._numsize = 3 + self._decimals + 1

            # end caps size calculation
            self._cap_len = len(self._bar_caps)/2
            self._capl = self._bar_caps[:self._cap_len]
            self._capr = self._bar_caps[self._cap_len:]

            # time remaining calculation for space
            self._time_space = 11 if self.time_remaining else 0

            # the space available for the progress bar is
            # 79 (screen) - (label) - (number) - 2 ([]) - 2 (space) - 1 (%)
            self._barsize = (
                    self.screen - len(self.label) - self._numsize -
                    len(self._bar_caps) - 2 - 1 - self._time_space
                )

            self._formatstr = '\r{label} {_capl}{_bars:<{_barsize}}{_capr} {_percent:>{_numsize}.{_decimals}f}%'

            self._percent = 0
            self._dt = '--:--:--'
            self._bars = ''

            if self.time_remaining:
                self._formatstr += " ({_dt})"
        else:
            self._digits = str(int(np.ceil(np.log10(self.num))))
            self._formatstr = '\r{label} : {value:>{_digits}} / {num:>{_digits}}'

            self._dt = '--:--:--'
            if self.time_remaining:
                self._formatstr += " ({_dt})"

        self.update()

    def _estimate_time(self):
        if len(self._deltas) < 3:
            self._dt = '--:--:--'
        else:
            dt = np.diff(self._deltas[-25:]).mean() * (self.num - self.value)
            self._dt = time.strftime('%H:%M:%S', time.gmtime(dt))

    def _draw(self):
        """ Interal draw method, simply prints to screen """
        print self._formatstr.format(**self.__dict__),
        sys.stdout.flush()

    def increment(self):
        self.update(self.value + 1)

    def update(self, value=0):
        """
        Update the value of the progress and update progress bar.

        Parameters:
        -----------
        value : integer
            The current iteration of the progress
        """
        self._deltas.append(time.time())

        self.value = value
        self._percent = 100.0 * self.value / self.num

        if self.bar:
            self._bars = self._bar_symbol*int(np.round(self._percent / 100. * self._barsize))

        if (len(self._deltas) < 2) or (self._deltas[-1] - self._deltas[-2]) > 1e-1:
            self._estimate_time()
            self._draw()

        if self.value == self.num:
            self.end()

    def end(self):
        print '\r{lett:>{screen}}'.format(**{'lett':'', 'screen': self.screen})

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
