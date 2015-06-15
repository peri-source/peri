import scipy.ndimage as nd
import numpy as np
import scipy as sp
import pylab as pl
from scipy.misc import imread
from scipy import signal
import matplotlib as mpl
import time

from .cu import nbl

def load_stack(filename, do3d=False):
    if do3d:
        z = load_tiff(filename, do3d=True)
    else:
        try:
            z = imread(filename)
        except Exception as e:
            z = load_tiff(filename, do3d)
    return z

def load_tiff(filename, do3d=False):
    if do3d:
        import libtiff
        tif = libtiff.TIFF.open(filename)
        z = np.array([a for a in tif.iter_images()])
    else:
        import libtiff
        tif = libtiff.TIFF.open(filename)
        it = tif.iter_images()
        for i in xrange(4):
            it.next()
        z = it.next()
        #z = imread("./colloids.tif")
    return z

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

def fake_image_2d(N=32, radius=5, phi=0.53, noise=0.05, seed=10, psf=(0.5,5)):
    from colloids.sim import Sim
    from colloids.cu import fields
    s = Sim(N, radius=radius, seed=seed)
    s.init_random_2d(phi=phi)
    s.do_relaxation(4000)
    s.set_param_hs()
    s.do_steps(100)

    L = s.L.astype('int32')+1
    x = s.get_pos().flatten()
    r = s.get_rad().flatten()
    p = np.array(psf, dtype='float32')

    im = np.zeros(L, dtype='float32').flatten()
    field = fields.createField(L)
    fields.fieldSet(field, im)
    fields.process_image(field, x, r, p, fields.PSF_ISOTROPIC_DISC, 16)
    fields.fieldGet(field, im)
    fields.freeField(field)

    im = im.reshape(L[::-1])
    imnoise = np.random.normal(0, noise, L[::-1])
    im += imnoise
    print 'Noise disturbance added:', (imnoise**2).sum()/noise**2
    return im[:], x.reshape(-1,3), r, p

def fake_image_3d(N=32, radius=5, phi=0.53, noise=0.05, seed=10, psf=(0.5,5)):
    from colloids.sim import Sim
    from colloids.cu import fields
    s = Sim(N, radius=radius, seed=seed)
    s.init_random_3d(phi=phi)
    s.do_relaxation(4000)
    s.set_param_hs()
    s.do_steps(100)

    L = s.L.astype('int32')+1
    x = s.get_pos().flatten().astype('float64')
    r = s.get_rad().flatten().astype('float64')
    p = np.array(psf, dtype='float32')

    nbl.naive_renormalize_radii(x, r, 1)

    im = np.zeros(L, dtype='float32').flatten()
    field = fields.createField(L)
    fields.fieldSet(field, im)
    fields.process_image(field, x, r, p, fields.PSF_ISOTROPIC_DISC, int(radius*3))
    fields.fieldGet(field, im)
    fields.freeField(field)

    im = im.reshape(L[::-1])
    nonoise = im.copy()
    imnoise = np.random.normal(0, noise, L[::-1])
    im += imnoise
    print 'Noise disturbance added:', (imnoise**2).sum()/noise**2
    return nonoise, im[:], x.reshape(-1,3), r, p

def scan(im, cycles=1):
    pl.figure(1)
    pl.show()
    time.sleep(3)
    for c in xrange(cycles):
        for i, sl in enumerate(im):
            print i
            pl.clf()
            pl.imshow(sl, cmap=pl.cm.bone, interpolation='nearest',
                    origin='lower', vmin=0, vmax=1)
            pl.draw()
            time.sleep(0.3)

def scan_together(im, p, delay=2):
    pl.figure(1)
    pl.show()
    time.sleep(3)
    z,y,x = p.T
    for i in xrange(len(im)):
        print i
        sl = im[i]
        pl.clf()
        pl.imshow(sl, cmap=pl.cm.bone, interpolation='nearest', origin='lower')
        m = z.astype('int') == i
        #pl.plot(y[m], x[m], 'o')
        pl.plot(x[m], y[m], 'o')
        pl.xlim(0, sl.shape[0])
        pl.ylim(0, sl.shape[1])
        pl.draw()
        time.sleep(delay)

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

def local_max_featuring(im, size=10):
    x,y,z = np.mgrid[0:2*size,0:2*size,0:2*size]
    r = np.sqrt((x-size-0.5)**2 + (y-size-0.5)**2 + (z-size-0.5)**2)
    footprint = r < size - 1

    tim = im.copy()
    tim = remove_z_mean(tim)
    tim = smooth(tim, 2)
    tim = highpass(tim, size/2.0/im.shape[0])
    maxes = nd.maximum_filter(tim, footprint=footprint)
    equal = maxes == tim

    label = nd.label(equal)[0]
    pos = np.array(nd.measurements.center_of_mass(equal, labels=label, index=np.unique(label)))
    x,y,z = pos.T
    pos = np.vstack([z,y,x]).T
    return pos[1:], tim
