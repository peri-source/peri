import scipy.ndimage as nd
import numpy as np
import scipy as sp
import pylab as pl
from scipy.misc import imread
from scipy import signal
import matplotlib as mpl
import time
import itertools
from PIL import Image

def remove_overlaps(pos, rad):
    N = rad.shape[0]
    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                continue;
            d = np.sqrt(((pos[i] - pos[j])**2).sum())
            r = rad[i] + rad[j]
            diff = d - r
            if diff < 0:
                rad[i] -= np.abs(diff)/2 + 1e-10
                rad[j] -= np.abs(diff)/2 + 1e-10

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

def remove_z_mean(im):
    for i in xrange(im.shape[0]):
        im[i] -= im[i].mean()
    return im

def normalize(im, invert=False):
    out = im.astype('float').copy()
    out -= 1.0*out.min()
    out /= 1.0*out.max()
    if invert:
        out = 1 - out
    return out

def highpass(im, frac):
    fx = np.fft.fftfreq(im.shape[2])[None,None,:]
    fy = np.fft.fftfreq(im.shape[1])[None,:,None]
    fz = np.fft.fftfreq(im.shape[0])[:,None,None]
    fr = np.sqrt(fx**2 + fy**2 + fz**2)
    ff = np.fft.fftn(im)
    return np.real( np.fft.ifftn( ff * (fr > frac / np.sqrt(2))))

def smooth(im, sigma):
    return nd.gaussian_filter(im, sigma)

def log_featuring(im, size_range=[0,20]):
    from skimage.feature import blob_log

    potpos = []
    potim = []
    for layer in im:
        ll = smooth(highpass(layer, 5./512), 1)

        bl = blob_log(ll, min_sigma=5, max_sigma=20, threshold=0.09, overlap=0.2)

        ll *= 0
        if len(bl) > 0:
            x = np.clip(bl[:,1], 0, ll.shape[1])
            y = np.clip(bl[:,0], 0, ll.shape[0])
            ll[x, y] = 1

    a = potpos
    b = np.array(potim)

    q = nd.gaussian_filter(np.array(potim), 2)
    s = q*(q > 0.02)
    r = nd.label(s)[0]
    p = np.array(nd.measurements.center_of_mass(q, labels=r, index=np.unique(r)))

    return p, q, s

def generate_sphere(radius):
    x,y,z = np.mgrid[0:2*radius,0:2*radius,0:2*radius]
    r = np.sqrt((x-radius-0.5)**2 + (y-radius-0.5)**2 + (z-radius-0.5)**2)
    sphere = r < radius - 1
    return sphere

def local_max_featuring(im, radius=10, smooth=4):
    g = nd.gaussian_filter(im, smooth, mode='mirror')
    e = nd.maximum_filter(g, footprint=generate_sphere(radius))
    lbl = nd.label(e == g)[0]
    pos = np.array(nd.measurements.center_of_mass(e==g, lbl, np.unique(lbl)))
    return pos[1:], e

def trackpy_featuring(im, size=10):
    from trackpy.feature import locate
    size = size + (size+1) % 2
    a = locate(im, size, invert=True)
    pos = np.vstack([a.z, a.y, a.x]).T
    return pos


