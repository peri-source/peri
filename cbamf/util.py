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
    def __init__(self, num, label='Progress', value=0, screen=79,
            bar=True, bar_symbol='=', bar_caps='[]', bar_decimals=2):
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

        bar : boolean [default: True]
            Whether or not to display the bar chart

        bar_symbol : char [default: '=']
            The character to use to fill in the bar chart

        bar_caps : string [default: '[]']
            Characters to use as the end caps of the.  The string will be split in
            half and each half put on a side of the chart

        bar_decimals: integer [default: 2]
            Number of decimal places to include in the percentage
        """
        self.num = num
        self.value = value
        self.percent = 0

        self.label = label
        self.bar = bar
        self.bar_symbol = bar_symbol
        self.bar_caps = bar_caps
        self.bar_decimals = bar_decimals
        self.screen = screen

        if len(self.bar_caps) % 2 != 0:
            raise AttributeError("End caps must be even number of symbols")

        if self.bar:
            # 3 digit percent + decimal places + '.'
            self._num_size = 3 + self.bar_decimals + 1

            # end caps size calculation
            self._cap_len = len(self.bar_caps)/2
            self._cap_l = self.bar_caps[:self._cap_len]
            self._cap_r = self.bar_caps[self._cap_len:]

            # the space available for the progress bar is
            # 79 (screen) - (label) - (number) - 2 ([]) - 2 (space) - 1 (%)
            self._bar_size = (
                    self.screen - len(self.label) - self._num_size -
                    len(self.bar_caps) - 2 - 1
                )

            self.pdict = {
                "label": self.label,
                "bars": '',
                "barsize": self._bar_size,
                "numsize": self._num_size,
                "decs": self.bar_decimals,
                "capl": self._cap_l,
                "capr": self._cap_r,
                "percent": '',
            }
            self._formatstr = '\r{label} {capl}{bars:<{barsize}}{capr} {percent:>{numsize}.{decs}f}%'
        else:
            self._digits = str(int(np.ceil(np.log10(self.num))))
            self._formatstr = '\r{label} : {value:>{_digits}} / {num:>{_digits}}'

        self.update()

    def _draw(self):
        """ Interal draw method, simply prints to screen """
        if self.bar:
            self.pdict.update({"percent": self.percent, "bars": self.bar_symbol*self.bars})
            print self._formatstr.format(**self.pdict),
        else:
            print self._formatstr.format(**self.__dict__),
        sys.stdout.flush()

    def update(self, value=0):
        """
        Update the value of the progress and update progress bar.

        Parameters:
        -----------
        value : integer
            The current iteration of the progress
        """
        self.value = value
        self.percent = 100.0 * self.value / self.num

        if self.bar:
            self.bars = int(np.round(self.percent / 100. * self._bar_size))
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
