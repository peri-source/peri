import os
import sys
import time
import datetime
import numpy as np
import code, traceback, signal
from contextlib import contextmanager

from cbamf import const, initializers

#=============================================================================
# Tiling utilities
#=============================================================================
def oddify(a):
    return a + (a%2==0)

def listify(a):
    if a is None:
        return []
    elif not isinstance(a, (tuple, list)):
        return [a]
    return list(a)

def amin(a, b):
    return np.vstack([a3(a), a3(b)]).min(axis=0)

def amax(a, b):
    return np.vstack([a3(a), a3(b)]).max(axis=0)

def a3(a):
    """ Convert an integer or iterable list to numpy 3 array """
    if not hasattr(a, '__iter__'):
        return np.array([a]*3, dtype='int')
    return np.array(a).astype('int')

class Tile(object):
    def __init__(self, left, right=None, mins=None, maxs=None,
            size=None, centered=False):
        """
        Creates a tile element using many different combinations (where []
        indicates an array created from either a single number or any
        iterable):

            left : [0,0,0] -> [left]
            left, right : [left] -> [right]
            left, size : [left] -> [left] + [size]              (if not centered)
            left, size : [left] - [size]/2 -> [left] + ([size]+1)/2 (if centered)

        The addition +1 on the last variety is to ensure that odd sized arrays
        are treated correctly i.e. left=0, size=3 -> [-1,0,1]. Each of these
        can be limited by using (mins, maxs) which are applied after
        calculating left, right for each element:

            left = max(left, [mins])
            right = min(right, [maxs])

        Since tiles are used for array slicing, they only allow integer values,
        which can truncated without warning from float.
        """
        if right is None:
            if size is None:
                right = left
                left = 0
            else:
                if not centered:
                    right = a3(left) + a3(size)
                else:
                    l, s = a3(left), a3(size)
                    left, right = l - s/2, l + (s+1)/2

        left = a3(left)
        right = a3(right)

        if mins is not None:
            left = amax(left, a3(mins))

        if maxs is not None:
            right = amin(right, a3(maxs))

        self.l = np.array(left)
        self.r = np.array(right)

    @property
    def slicer(self):
        l, r = self.bounds
        return np.s_[l[0]:r[0], l[1]:r[1], l[2]:r[2]]

    @property
    def shape(self):
        return self.r - self.l

    @property
    def bounds(self):
        return (self.l, self.r)

    def center(self, norm=1.0):
        """ Return the center of the tile """
        return (self.r + self.l)/2.0 / norm

    # FIXME -- switch meshed to False, or make a type list [meshed, vector, flat]
    def coords(self, norm=False, meshed=True, vector=False, flat=False):
        """
        Returns the coordinate vectors associated with the tile. 

        Norm can rescale the coordinates for you. False is no rescaling, True
        is rescaling so that all coordinates are from 0 -> 1.  If a scalar, the
        same norm is applied uniformally while if an iterable, each scale is
        applied to each dimension.

        If meshed, the arrays are broadcasted so that they form 3, 3D arrays.

        If vector, the arrays are broadcasted and formed into one [Nz, Ny, Nx, 3]
        dimensional array
        """
        if not norm:
            norm = np.ones(3)
        if norm is True:
            norm = self.shape
        if not hasattr(norm, '__iter__'):
            norm = [norm]*3

        z = np.arange(self.l[0], self.r[0]) / norm[0]
        y = np.arange(self.l[1], self.r[1]) / norm[1]
        x = np.arange(self.l[2], self.r[2]) / norm[2]

        if meshed:
            return np.meshgrid(z, y, x, indexing='ij')
        if vector:
            z,y,x = np.meshgrid(z, y, x, indexing='ij')
            return np.rollaxis(np.array(np.broadcast_arrays(z,y,x)),0,4)
        if flat:
            return z,y,x
        return z[:,None,None], y[None,:,None], x[None,None,:]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__class__.__name__)+" {} -> {} ({})".format(
                list(self.l), list(self.r), list(self.shape))

    def contains(self, items, pad=0):
        """
        Test whether coordinates are contained within this tile.

        Parameters:
        -----------
        items : ndarray [3] or [N, 3]
            N coordinates to check are within the bounds of the tile

        pad : integer or ndarray [3]
            anisotropic padding to apply in the contain test
        """
        o = ((items > self.l-pad) & (items < self.r+pad))
        if len(o.shape) == 2:
            o = o.all(axis=-1)
        return o

    # TODO -- implement as operators?
    @staticmethod
    def intersection(tiles, *args):
        """ Intersection of tiles, returned as a tile """
        tiles = listify(tiles) + listify(args)

        if len(tiles) < 2:
            return tiles[0]

        tile = tiles[0]
        l, r = tile.l.copy(), tile.r.copy()
        for tile in tiles[1:]:
            l = amax(l, tile.l)
            r = amin(r, tile.r)
        return Tile(l, r)

    @staticmethod
    def boundingtile(tiles, *args):
        """ Convex bounding box of a group of tiles """
        tiles = listify(tiles) + listify(args)

        if len(tiles) < 2:
            return tiles[0]

        tile = tiles[0]
        l, r = tile.l.copy(), tile.r.copy()
        for tile in tiles[1:]:
            l = amin(l, tile.l)
            r = amax(r, tile.r)
        return Tile(l, r)

    def __and__(self, other):
        return Tile.intersection(self, other)

    def __or__(self, other):
        return Tile.boundingtile(self, other)

    def copy(self):
        return Tile(self.l.copy(), self.r.copy())

    def translate(self, dr):
        """ Translate a tile by an amount dr """
        tile = self.copy()
        tile.l += dr
        tile.r += dr
        return tile

    def pad(self, pad):
        """ Pad this tile by an equal amount on each side as specified by pad """
        tile = self.copy()
        tile.l -= pad
        tile.r += pad
        return tile

    def overhang(self, tile):
        """
        Get the left and right absolute overflow -- the amount of box
        overhanging `tile`, can be viewed as self \\ tile (set theory relative
        complement, but in a bounding sense)
        """
        ll = np.abs(amin(self.l - tile.l, 0))
        rr = np.abs(amax(self.r - tile.r, 0))
        return ll, rr

    def reflect_overhang(self, clip):
        """
        Compute the overhang and reflect it internally so respect periodic
        padding rules (see states._tile_from_particle_change). Returns both
        the inner tile and the inner tile with necessary pad.
        """
        orig = self.copy()
        tile = self.copy()

        hangl, hangr = tile.overhang(clip)
        tile = tile.pad(hangl)
        tile = tile.pad(hangr)

        inner = Tile.intersection([clip, orig])
        outer = Tile.intersection([clip, tile])
        return inner, outer


class RawImage():
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
        self.filters = None

        self.load_image()

        if tile is not None:
            self.tile = tile
        else:
            zstart = zstart or 0
            zstop = zstop or self.image.shape[0]

            left = (zstart, 0, 0)
            right = (zstop, xysize, xysize)
            self.tile = Tile(left=left, right=right)

    def load_image(self):
        try:
            self.image = initializers.load_tiff(self.filename)
            self.image = initializers.normalize(self.image, invert=self.invert)
        except IOError as e:
            print "Could not find image '%s'" % self.filename
            raise e

    def get_image(self):
        im = self.image[self.tile.slicer]

        if not self.filters:
            return im
        return self.filtered_image(im)

    def get_padded_image(self, pad=const.PAD, padval=const.PADVAL):
        return np.pad(self.get_image(), pad, mode='constant', constant_values=padval)

    def filtered_image(self, im):
        q = np.fft.fftn(im)
        for k,v in self.filters:
            q[k] -= v
        return np.real(np.fft.ifftn(q))

    def set_filter(self, slices, values):
        self.filters = [[sl,values[sl]] for sl in slices]

    def __getstate__(self):
        d = self.__dict__.copy()
        cdd(d, ['image'])
        return d

    def __setstate__(self, idct):
        self.__dict__.update(idct)
        self.load_image()

    #def __getinitargs__(self):
    #    return (self.filename, self.tile, None, None, None, self.invert, self.filters)

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

#=============================================================================
# misc helper functions
#=============================================================================
@contextmanager
def indir(path):
    """
    Context manager for switching the current path of the process. Can be used:

        with indir('/tmp'):
            <do something in tmp>
    """
    cwd = os.getcwd()

    try:
        os.chdir(path)
        yield
    except Exception as e:
        raise
    finally:
        os.chdir(cwd)
