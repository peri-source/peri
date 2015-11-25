import sys
import time
import datetime
import numpy as np
import code, traceback, signal

from cbamf import const, initializers

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

    def center(self, norm=1.0):
        return (self.r + self.l)/2.0 / norm

    def coords(self, norm=1.0, meshed=True):
        z = np.arange(self.l[0], self.r[0]) / norm
        y = np.arange(self.l[1], self.r[1]) / norm
        x = np.arange(self.l[2], self.r[2]) / norm
        if meshed:
            return np.meshgrid(z, y, x, indexing='ij')
        return z[:,None,None], y[None,:,None], x[None,None,:]

    def coord_vector(self, *args, **kwargs):
        """ Creates a coordinate vector from self.coords """
        z,y,x = self.coords()
        return np.rollaxis(np.array(np.broadcast_arrays(z,y,x)),0,4)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__class__.__name__)+" {} -> {} ({})".format(
                list(self.l), list(self.r), list(self.shape))

class RawImage(object):
    def __init__(self, filename, tile=None, zstart=None, zstop=None,
            xysize=None, invert=False):
        """
        An image object which stores information about desired region and padding,
        etc.  There are number of ways to create an image object:

        filename : str
            path of the image file.  recommended that you supply a relative path
            so that transfer between computers is possible

        tile : `cbamf.util.Tile`

        """
        self.filename = filename
        self.invert = invert

        try:
            self.image = initializers.load_tiff(self.filename)
            self.image = initializers.normalize(self.image, invert=self.invert)
        except IOError as e:
            print "Could not find image '%s'" % self.filename
            raise e

        if tile is not None:
            self.tile = tile
        else:
            zstart = zstart or 0
            left = (zstart, 0, 0)
            right = (zstop, xysize, xysize)
            self.tile = Tile(left=left, right=right)

    def get_image(self):
        return self.image[self.tile.slicer]

    def get_padded_image(self, pad=const.PAD, padval=const.PADVAL):
        return np.pad(self.get_image(), pad, mode='constant', constant_values=padval)

    def __getstate__(self):
        return {}

    def __setstate__(self):
        pass

    def __initargs__(self):
        return (self.filename, self.tile, None, None, None, self.invert)

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
            bar_decimals=2, display=True):
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

        display : boolean [default: True]
            a crutch so that we don't have a lot of `if`s later.  display
            or don't display the progress bar
        """
        # TODO -- add estimated time remaining
        self.num = num
        self.value = value
        self._percent = 0
        self.time_remaining = time_remaining
        self._deltas = []
        self.display = display

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
        if self.display:
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
        if self.display:
            print '\r{lett:>{screen}}'.format(**{'lett':'', 'screen': self.screen})


#=============================================================================
# useful decorators
#=============================================================================
import collections
import functools
import types

def newcache():
    out = {}
    out['hits'] = 0
    out['misses'] = 0
    out['size'] = 0
    return out

def memoize(cache_max_size=1e9):
    def memoize_inner(obj):
        cache_name = str(obj)

        @functools.wraps(obj)
        def wrapper(self, *args, **kwargs):
            # add the memoize cache to the object first, if not present
            # provide a method to the object to clear the cache too
            if not hasattr(self, '_memoize_caches'):
                def clear_cache(self):
                    for k,v in self._memoize_caches.iteritems():
                        self._memoize_caches[k] = newcache()
                self._memoize_caches = {}
                self._memoize_clear = types.MethodType(clear_cache, self)

            # next, add the particular cache for this method if it does
            # not already exist in the parent 'self'
            cache = self._memoize_caches.get(cache_name)
            if not cache:
                cache = newcache()
                self._memoize_caches[cache_name] = cache

            size = 0
            hashed = []

            # let's hash the arguments (both args, kwargs) and be mindful of
            # numpy arrays -- that is, only take care of its data, not the obj
            # itself
            for arg in args:
                if isinstance(arg, np.ndarray):
                    hashed.append(arg.tostring())
                else:
                    hashed.append(arg)
            for k,v in kwargs.iteritems():
                if isinstance(v, np.ndarray):
                    hashed.append(v.tostring())
                else:
                    hashed.append(v)

            hashed = tuple(hashed)
            if hashed not in cache:
                ans = obj(self, *args, **kwargs)

                # if it is not too much to ask, place the answer in the cache
                if isinstance(ans, np.ndarray):
                    size = ans.nbytes

                newsize = size + cache['size']
                if newsize < cache_max_size:
                    cache[hashed] = ans
                    cache['misses'] += 1
                    cache['size'] = newsize
                return ans

            cache['hits'] += 1
            return cache[hashed]

        return wrapper

    return memoize_inner

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
