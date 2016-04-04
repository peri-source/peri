import scipy.ndimage as nd
import numpy as np
import scipy as sp
import pylab as pl
from scipy.misc import imread
from scipy import signal
import matplotlib as mpl
import time
import glob
import itertools
from PIL import Image

from cbamf import const

#=======================================================================
# Image loading functions
#=======================================================================
def _sliceiter(img):
    i = 0
    while True:
        try:
            img.seek(i)
            yield np.array(img)
            i += 1
        except EOFError as e:
            break

def load_tiff(filename):
    img = Image.open(filename)
    return np.array(list(_sliceiter(img)))

def load_tiffs(fileglob):
    files = glob.glob(fileglob)
    files.sort()

    for f in files:
        yield f, load_tiff(f)

def load_tiff_layer(filename, layer):
    img = Image.open(filename)
    return np.array(list(itertools.islice(_sliceiter(img), layer, layer+1)))

def load_tiff_iter(filename, iter_slice_size):
    img = Image.open(filename)
    slices = _sliceiter(img)
    while True:
        ims = np.array(list(itertools.islice(slices, iter_slice_size)))
        if len(ims.shape) > 1:
            yield ims
        else:
            break

def load_tiff_iter_libtiff(filename, iter_slice_size):
    import libtiff
    tifs = libtiff.TIFF.open(filename).iter_images()

    while True:
        ims = [a for a in itertools.islice(tifs, iter_slice_size)]
        if len(ims) > 0:
            yield np.array(ims)
        else:
            break

#=======================================================================
# Featuring functions
#=======================================================================
def normalize(im, invert=False):
    out = im.astype('float').copy()

    if out.ptp() != 0 and out.max() != 0:
        out -= 1.0*out.min()
        out /= 1.0*out.max()

    if invert:
        out = 1 - out
    return out

def generate_sphere(radius):
    x,y,z = np.mgrid[0:2*radius,0:2*radius,0:2*radius]
    r = np.sqrt((x-radius-0.5)**2 + (y-radius-0.5)**2 + (z-radius-0.5)**2)
    sphere = r < radius - 1
    return sphere

def local_max_featuring(im, radius=10, smooth=4, masscut=None):
    g = nd.gaussian_filter(im, smooth, mode='mirror')
    footprint = generate_sphere(radius)
    e = nd.maximum_filter(g, footprint=footprint)
    lbl = nd.label(e == g)[0]
    ind = np.sort(np.unique(lbl))[1:]
    pos = np.array(nd.measurements.center_of_mass(e==g, lbl, ind))
    if masscut is not None:
        m = nd.convolve(im, footprint, mode='reflect')
        mass = np.array(map(lambda x: m[x[0],x[1],x[2]], pos))
        good = mass > masscut
        return pos[good].copy(), e, mass[good].copy()
    else:
        return pos, e

def trackpy_featuring(im, size=10):
    from trackpy.feature import locate
    size = size + (size+1) % 2
    a = locate(im, size, invert=True)
    pos = np.vstack([a.z, a.y, a.x]).T
    return pos

def remove_overlaps(pos, rad, zscale=1, doprint=False):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])
    for i in xrange(N):
        o = np.arange(i+1, N)
        d = np.sqrt( ((z*(pos[i] - pos[o]))**2).sum(axis=-1) )
        r = rad[i] + rad[o]

        diff = d-r
        mask = diff < 0
        imask = o[mask]
        dmask = diff[mask]

        for j, d in zip(imask, dmask):
            rad[i] -= np.abs(d)*rad[i]/(rad[i]+rad[j]) + 1e-10
            rad[j] -= np.abs(d)*rad[j]/(rad[i]+rad[j]) + 1e-10
            if doprint:
                print diff, rad[i], rad[j]

def remove_overlaps_naive(pos, rad, zscale=1, doprint=False):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                continue;
            d = np.sqrt(( (z*(pos[i] - pos[j]))**2 ).sum())
            r = rad[i] + rad[j]
            diff = d - r
            if diff < 0:
                rad[i] -= np.abs(diff)*rad[i]/(rad[i]+rad[j]) + 1e-10
                rad[j] -= np.abs(diff)*rad[j]/(rad[i]+rad[j]) + 1e-10
                if doprint:
                    print diff, rad[i], rad[j]

def remove_background(im, order=(5,5,4), mask=None):
    from cbamf.comp import ilms
    ilm = ilms.Polynomial3D(order=order, shape=im.shape) 
    ilm.from_data(im, mask=mask)

    return im - ilm.get_field()
