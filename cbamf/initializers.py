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
    out -= 1.0*out.min()
    out /= 1.0*out.max()
    if invert:
        out = 1 - out
    return out

def fsmooth(im, sigma):
    kz, ky, kx = np.meshgrid(*[np.fft.fftfreq(i) for i in feat.shape], indexing='ij')
    ksq = kx**2 + ky**2 + kz**2
    kim = np.fft.fftn(im)
    kim *= np.exp(-ksq * sigma**2)
    return np.real(np.fft.ifftn(kim))

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

def remove_overlaps(pos, rad, zscale=1, doprint=False):
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


#=======================================================================
# More involved featuring functions using MC
#=======================================================================
def sample_particle_add(s, rad, tries=5):
    diff = (s.get_model_image() - s.get_true_image()).copy()

    smoothdiff = nd.gaussian_filter(diff, rad/2.0)
    maxfilter = nd.maximum_filter(smoothdiff, size=rad)
    eq = smoothdiff == maxfilter
    lbl = nd.label(eq)[0]
    pos = np.array(nd.center_of_mass(eq, lbl, np.unique(lbl)))[1:].astype('int')
    ind = np.arange(len(pos))

    val = [maxfilter[tuple(pos[i])] for i in ind]
    vals = sorted(zip(val, ind))

    accepts = 0
    for _, i in vals[-tries:][::-1]:
        diff = (s.get_model_image() - s.get_true_image())/(2*s.sigma**2)

        p = pos[i].reshape(-1,3)
        n = s.obj.typ.argmin()

        bp = s.block_particle_pos(n)
        br = s.block_particle_rad(n)
        bt = s.block_particle_typ(n)

        s.update(bp, p)
        s.update(br, np.array([rad]))
        s.update(bt, np.array([1]))

        bl = s.blocks_particle(n)[:-1]
        runner.sample_state(s, bl, stepout=1, N=1)

        diff2 = (s.get_model_image() - s.get_true_image())/(2*s.sigma**2)

        print p, (diff**2).sum(), (diff2**2).sum()
        if not (np.log(np.random.rand()) > (diff2**2).sum() - (diff**2).sum()):
            s.update(bt, np.array([0]))
        else:
            accepts += 1
    return accepts

def sample_particle_remove(s, rad, tries=5):
    diff = (s.get_model_image() - s.get_true_image()).copy()

    smoothdiff = nd.gaussian_filter(diff, rad/2.0)
    maxfilter = nd.maximum_filter(smoothdiff, size=rad)
    eq = smoothdiff == maxfilter
    lbl = nd.label(eq)[0]
    pos = np.array(nd.center_of_mass(eq, lbl, np.unique(lbl)))[1:].astype('int')
    ind = np.arange(len(pos))

    val = [maxfilter[tuple(pos[i])] for i in ind]
    vals = sorted(zip(val, ind))

    accepts = 0
    for _, i in vals[-tries:]:
        diff = (s.get_model_image() - s.get_true_image())/(2*s.sigma**2)

        n = ((s.obj.pos - pos[i])**2).sum(axis=0).argmin()

        bt = s.block_particle_typ(n)
        s.update(bt, np.array([0]))

        diff2 = (s.get_model_image() - s.get_true_image())/(2*s.sigma**2)

        print s.obj.pos[n], (diff**2).sum(), (diff2**2).sum()
        if not (np.log(np.random.rand()) > (diff2**2).sum() - (diff**2).sum()):
            s.update(bt, np.array([1]))
        else:
            accepts += 1
    return accepts

def full_feature(s, rad, globaloptimizes=2, add_remove_tries=20):
    for i in xrange(globaloptimizes):
        accepts = 1
        while accepts > 0:
            accepts = 0
            accepts += sample_particle_add(s, rad=rad, tries=add_remove_tries)
            accepts += sample_particle_remove(s, rad=rad, tries=add_remove_tries/5)
            runner.sample_particle_pos(s, stepout=1)
            runner.sample_block(s, 'ilm', stepout=0.1)
            runner.sample_block(s, 'off', stepout=0.1)

        for i in xrange(2):
            runner.sample_particle_pos(s, stepout=1)
            runner.sample_particle_rad(s, stepout=1)

        runner.do_samples(s, 5, 5)

    return runner.do_samples(s, 20, 10)

#=======================================================================
# Generating fake data
#=======================================================================
def fake_image(pos, rad, psf=(1, 2), shape=(128,128,128)):
    pass
