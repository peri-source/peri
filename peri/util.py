from __future__ import print_function

import os
import sys
import time
import inspect
import itertools
import numpy as np
from contextlib import contextmanager

from peri import initializers
from peri.logger import log
log = log.getChild('util')

#=============================================================================
# Tiling utilities
#=============================================================================
def oddify(a):
    return a + (a % 2 == 0)

def listify(a):
    if a is None:
        return []
    elif not isinstance(a, (tuple, list, np.ndarray)):
        return [a]
    return list(a)

def delistify(a):
    if isinstance(a, (tuple, list, np.ndarray)) and len(a) == 1:
        return a[0]
    return a

def imin(a, b, dim=3):
    return np.vstack([iN(a, dim), iN(b, dim)]).min(axis=0)

def imax(a, b, dim=3):
    return np.vstack([iN(a, dim), iN(b, dim)]).max(axis=0)

def iN(a, dim=3):
    """ Convert an integer or iterable list to numpy 3 array """
    if not hasattr(a, '__iter__'):
        return np.array([a]*dim, dtype='float')
    return np.array(a).astype('float')

def amin(a, b, dim=3):
    return np.vstack([aN(a, dim), aN(b, dim)]).min(axis=0)

def amax(a, b, dim=3):
    return np.vstack([aN(a, dim), aN(b, dim)]).max(axis=0)

def aN(a, dim=3):
    """ Convert an integer or iterable list to numpy 3 array """
    if not hasattr(a, '__iter__'):
        return np.array([a]*dim, dtype='int')
    return np.array(a).astype('int')

def getdim(a):
    if not hasattr(a, '__iter__'):
        return None
    return len(a)

class CompatibilityPatch(object):
    def patch(self, var):
        names, default_values = var.keys(), var.values()
        for n, v in zip(names, default_values):
            self.__dict__.update({n: self.__dict__.get(n, v)})

class Tile(CompatibilityPatch):
    def __init__(self, left, right=None, mins=None, maxs=None,
            size=None, centered=False, dim=None):
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

        Notes on dimension. The dimensionality is determined first by the shape
        of left, right, size if provided not as an integer. If it not provided there
        then it is assumed to be 3D. This can be overridden by setting dim in the
        arguments. For example:
            Tile(3)         : [0,0,0] -> [3,3,3]
            Tile(3, dim=2)  : [0,0] -> [3,3]
            Tile([3])       : [0] -> [3]
        """
        # first determine the dimensionality of the tile
        dims = set([getdim(i) for i in [left, right, size]] + [dim])
        dims = dims.difference(set([None]))

        if len(dims) == 0:
            dim = 3
        elif len(dims) == 1:
            dim = dims.pop()
        elif len(dims) > 1:
            raise AttributeError("Dimension mismatch between left, right, size, dim")

        if right is None:
            if size is None:
                right = left
                left = 0
            else:
                if not centered:
                    right = aN(left, dim) + aN(size, dim)
                else:
                    l, s = aN(left, dim), aN(size, dim)
                    left, right = l - s/2, l + (s+1)/2

        left = aN(left, dim)
        right = aN(right, dim)

        if dim is not None:
            self.dim = dim
            assert(left.shape[0] == dim)
            assert(right.shape[0] == dim)
        else:
            self.dim = left.shape[0]

        if mins is not None:
            left = amax(left, aN(mins, dim), dim=dim)

        if maxs is not None:
            right = amin(right, aN(maxs, dim), dim=dim)

        self.l = np.array(left)
        self.r = np.array(right)
        self._build_caches()

    def _build_caches(self):
        self._coord_slicers = []
        for i in xrange(self.dim):
            self._coord_slicers.append(
                tuple(None if j != i else np.s_[:] for j in xrange(self.dim))
            )

    @property
    def slicer(self):
        return tuple(np.s_[l:r] for l,r in zip(*self.bounds))

    def oslicer(self, tile):
        """ Opposite slicer, the outer part wrt to a field """
        mask = None
        vecs = tile.coords(form='meshed')
        for v in vecs:
            v[self.slicer] = -1
            mask = mask & (v > 0) if mask is not None else (v>0)
        return tuple(np.array(i).astype('int') for i in zip(*[v[mask] for v in vecs]))

    @property
    def shape(self):
        return self.r - self.l

    @property
    def bounds(self):
        return (self.l, self.r)

    @property
    def center(self):
        """ Return the center of the tile """
        return (self.r + self.l)/2.0

    @property
    def kcenter(self):
        """ Return the frequency center of the tile (says fftshift) """
        return np.array([
            np.abs(np.fft.fftshift(np.fft.fftfreq(q))).argmin()
            for q in self.shape
        ]).astype('float')

    @property
    def corners(self):
        corners = []
        for ind in itertools.product(*((0,1),)*self.dim):
            ind = np.array(ind)
            corners.append(self.l + ind*self.r)
        return np.array(corners)

    def _format_vector(self, vecs, form='broadcast'):
        """
        Format a 3d vector field in certain ways, see `coords` for a description
        of each formatting method.
        """
        if form == 'meshed':
            return np.meshgrid(*vecs, indexing='ij')
        elif form == 'vector':
            vecs = np.meshgrid(*vecs, indexing='ij')
            return np.rollaxis(np.array(np.broadcast_arrays(*vecs)),0,self.dim+1)
        elif form == 'flat':
            return vecs
        else:
            return [v[self._coord_slicers[i]] for i,v in enumerate(vecs)]

    def coords(self, norm=False, form='broadcast'):
        """
        Returns the coordinate vectors associated with the tile.

        Parameters:
        -----------
        norm : boolean
            can rescale the coordinates for you. False is no rescaling, True is
            rescaling so that all coordinates are from 0 -> 1.  If a scalar,
            the same norm is applied uniformally while if an iterable, each
            scale is applied to each dimension.

        form : string
            In what form to return the vector array. Can be one of:
                'broadcast' -- return 1D arrays that are broadcasted to be 3D

                'flat' -- return array without broadcasting so each component
                    is 1D and the appropriate length as the tile

                'meshed' -- arrays are explicitly broadcasted and so all have
                    a 3D shape, each the size of the tile.

                'vector' -- array is meshed and combined into one array with
                    the vector components along last dimension [Nz, Ny, Nx, 3]
        """
        if norm is False:
            norm = 1
        if norm is True:
            norm = np.array(self.shape)
        norm = iN(norm, self.dim)

        v = list(np.arange(self.l[i], self.r[i]) / norm[i] for i in xrange(self.dim))
        return self._format_vector(v, form=form)

    def kvectors(self, norm=False, form='broadcast', real=False, shift=False):
        """
        Return the kvectors associated with this tile, given the standard form
        of -0.5 to 0.5. `norm` and `form` arguments arethe same as that passed to
        `Tile.coords`.

        Parameters:
        -----------
        real : boolean
            whether to return kvectors associated with the real fft instead
        """
        if norm is False:
            norm = 1
        if norm is True:
            norm = np.array(self.shape)
        norm = iN(norm, self.dim)

        v = list(np.fft.fftfreq(self.shape[i])/norm[i] for i in xrange(self.dim))

        if shift:
            v = list(np.fft.fftshift(t) for t in v)

        if real:
            v[-1] = v[-1][:(self.shape[-1]+1)/2]

        return self._format_vector(v, form=form)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__class__.__name__)+" {} -> {} ({})".format(
            list(self.l), list(self.r), list(self.shape)
        )

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
        o = ((items >= self.l-pad) & (items < self.r+pad))
        if len(o.shape) == 2:
            o = o.all(axis=-1)
        elif len(o.shape) == 1:
            o = o.all()
        return o

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

    def __eq__(self, other):
        if other is None:
            return False
        return (self.l == other.l).all() and (self.r == other.r).all()

    def __ne__(self, other):
        if other is None:
            return True
        return ~self.__eq__(other)

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

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, idct):
        self.__dict__.update(idct)
        self.patch({'dim': 3})
        self._build_caches()


#=============================================================================
# Image classes
#=============================================================================
class Image(object):
    def __init__(self, image, tile=None, filters=None):
        """
        Create an image object from a raw np.ndarray.

        Parameters:
        -----------
        image : ndarray
            The image in float format with dimensions arranged as [z,y,x]

        tile : `peri.util.Tile`
            the region of the image to crop out to use for the actual featuring, etc

         """
        self.filters = filters or []
        self.image = image
        self.tile = tile or Tile(image.shape)

    def get_image(self):
        im = self.image[self.tile.slicer]

        if not self.filters:
            return im
        return self.filtered_image(im)

    def get_padded_image(self, pad, padval=0):
        if hasattr(pad, '__iter__'):
            pad = [[p, p] for p in pad]
        return np.pad(self.get_image(), pad, mode='constant', constant_values=padval)

    def filtered_image(self, im):
        """Returns a filtered image after applying the Fourier-space filters"""
        q = np.fft.fftn(im)
        for k,v in self.filters:
            q[k] -= v
        return np.real(np.fft.ifftn(q))

    def set_filter(self, slices, values):
        """
        Sets Fourier-space filters for the image. The image is filtered by
        subtracting values from the image at slices.

        Inputs
        ------
            slices : List of indices or slice objects.
                The q-values in Fourier space to filter.
            values : np.ndarray
                The complete array of Fourier space peaks to subtract off.
                values should be the same size as the FFT of the image;
                only the portions of values at slices will be removed.

        Example:
        -------
        To remove a two Fourier peaks in the data at q=(10, 10, 10) &
        (245, 245, 245), where im is an Image object:
            slices = [(10,10,10), (245, 245, 245)]
            values = np.fft.fftn(im.get_image())
            im.set_filter(slices, values)
        """
        self.filters = [[sl,values[sl]] for sl in slices]

    def __repr__(self):
        return "{} : {}".format(
            self.__class__.__name__, str(self.tile)
        )

    def __str__(self):
        return self.__repr__()

class NullImage(Image):
    def __init__(self, image=None, shape=None):
        """
        An image object that doesn't actual store any information so that small
        save states can be created for pure model states

        Parameters:
        -----------
        shape : tuple
            Size of the image which will be mocked
        """
        if image is not None:
            self.shape = image.shape
            super(NullImage, self).__init__(image)
        elif shape is not None:
            self.shape = shape
            super(NullImage, self).__init__(np.zeros(self.shape))
        else:
            raise AttributeError("Must provide either image or shape")

    def __getstate__(self):
        d = self.__dict__.copy()
        cdd(d, ['image'])
        return d

    def __setstate__(self, idct):
        self.__dict__.update(idct)
        super(NullImage, self).__init__(np.zeros(self.shape))

    def __repr__(self):
        return "{} : {}".format(self.__class__.__name__, self.shape)

    def __str__(self):
        return self.__repr__()

class RawImage(Image, CompatibilityPatch):
    def __init__(self, filename, tile=None, invert=False, exposure=None,
            float_precision=np.float64):
        """
        An image object which stores information about desired region and padding,
        etc.  There are number of ways to create an image object:

        filename : str
            path of the image file.  recommended that you supply a relative path
            so that transfer between computers is possible

        tile : `peri.util.Tile`
            the region of the image to crop out to use for the actual featuring, etc

        invert : boolean
            Whether to invert the image.

        exposure : tuple of floats (min, max) | None
            If set, it is the values used to normalize the image instead of setting
            min to 0 and max to 1 (which depends on noise, exposure). Setting this
            values allows a series of images to be initialized with the same
            ILM, PSF etc. Should be the bit value of the camera.

        float_precision : numpy float datatype
            One of numpy.float16, numpy.float32, numpy.float64; precision
            for precomputed arrays. Default is np.float64; make it 16 or 32
            to save memory.
        """
        self.filename = filename
        self.invert = invert
        self.filters = None
        self.exposure = exposure
        if float_precision not in (np.float64, np.float32, np.float16):
            raise ValueError('float_precision must be one of np.float64, ' +
                    'np.float32, np.float16')
        self.float_precision = float_precision

        image = self.load_image()
        super(RawImage, self).__init__(image, tile=tile)

    def load_image(self):
        try:
            image = initializers.load_tiff(self.filename)
            image = initializers.normalize(
                    image, invert=self.invert, scale=self.exposure,
                    dtype=self.float_precision)
        except IOError as e:
            log.error("Could not find image '%s'" % self.filename)
            raise e

        return image

    def get_scale(self):
        if self.exposure is not None:
            return self.exposure

        raw = initializers.load_tiff(self.filename)
        scaled = initializers.normalize(
            self.image, invert=self.invert, scale=self.exposure
        )
        t0, t1 = scaled.min(), scaled.max()
        r0, r1 = float(raw.min()), float(raw.max())

        rmin = (t1*r0 - t0*r1) / (t1 - t0)
        rmax = (r1 - r0) / (t1 - t0) + rmin
        return (rmin, rmax)

    def __getstate__(self):
        d = self.__dict__.copy()
        cdd(d, ['image'])
        return d

    def __setstate__(self, idct):
        self.__dict__.update(idct)
        self.patch({'float_precision': np.float64})
        self.image = self.load_image()

    def __repr__(self):
        return "{} <{}: {}>".format(
            self.__class__.__name__, self.filename, str(self.tile)
        )

    def __str__(self):
        return self.__repr__()

def cdd(d, k):
    """ Conditionally delete key (or list of keys) 'k' from dict 'd' """
    if not isinstance(k, list):
        k = [k]
    for i in k:
        if i in d:
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
            print(self._formatstr.format(**self.__dict__), end='')
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
            print('\r{lett:>{screen}}'.format(**{'lett':'', 'screen': self.screen}))


#=============================================================================
# useful decorators
#=============================================================================
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
# patching docstrings of sub-classes
#=============================================================================
def patch_docs(subclass, superclass):
    funcs0 = inspect.getmembers(subclass, predicate=inspect.ismethod)
    funcs1 = inspect.getmembers(superclass, predicate=inspect.ismethod)

    funcs1 = [f[0] for f in funcs1]

    for name, func in funcs0:
        if name.startswith('_'):
            continue

        if name not in funcs1:
            continue

        if func.__doc__ is None:
            func = getattr(subclass, name)
            func.im_func.__doc__ = getattr(superclass, name).im_func.__doc__

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
