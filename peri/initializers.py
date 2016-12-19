import glob
import itertools
from PIL import Image

import numpy as np
import scipy.ndimage as nd

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
def normalize(im, invert=False, scale=None, dtype=np.float64):
    """
    Normalize a field to a (min, max) exposure range, default is (0, 255).
    (min, max) exposure values. Invert the image if requested.
    """
    if dtype not in {np.float16, np.float32, np.float64}:
        raise ValueError('dtype must be numpy.float16, float32, or float64.')
    out = im.astype('float').copy()

    scale = scale or (0.0, 255.0)
    l, u = (float(i) for i in scale)
    out = (out - l) / (u - l)
    if invert:
        out = -out + (out.max() + out.min())
    return out.astype(dtype)

def generate_sphere(radius):
    """Generates a centered boolean mask of a 3D sphere"""
    rint = np.ceil(radius).astype('int')
    t = np.arange(-rint, rint+1, 1)
    x,y,z = np.meshgrid(t, t, t, indexing='ij')
    r = np.sqrt(x*x + y*y + z*z)
    sphere = r < radius
    return sphere

def local_max_featuring(im, radius=2.5, noise_size=1., bkg_size=None,
        masscut=1., trim_edge=False):
    """
    Local max featuring to identify spherical particles in an image.

    Parameters
    ----------
        im : numpy.ndarray
            The image to identify particles in.
        radius : Float, optional
            Featuring radius of the particles. Default is 2.5
        noise_size : Float, optional
            Size of Gaussian kernel for smoothing out noise. Default is 1.
        masscut : Float, optional
            Return only particles with a ``mass > masscut``. Default is 1.
        trim_edge : Bool, optional
            Set to True to omit particles identified exactly at the edge of
            the image. False features frequently occur here because of the
            reflected bandpass featuring. Default is False.

    Returns
    -------
        positions : numpy.ndarray
        e : numpy.ndarray
            The maximum-filtered array...
        [, mass] : numpy.ndarray
            Particle masses; if ``masscut`` is not None.
    """
    #1. Remove noise
    g = nd.gaussian_filter(im, noise_size, mode='mirror')
    #2. Remove long-wavelength background:
    if bkg_size is None:
        bkg_size = 2*radius
    g -= nd.gaussian_filter(g, bkg_size, mode='mirror')
    #3. Local max feature
    footprint = generate_sphere(radius)
    e = nd.maximum_filter(g, footprint=footprint)
    mass_im = nd.convolve(g, footprint, mode='mirror')
    good_im = (e==g) * (mass_im > masscut)
    pos = np.transpose(np.nonzero(good_im))
    # pos = np.array(nd.measurements.center_of_mass(e==g, lbl, ind))
    if trim_edge:
        good = np.all(pos > 0, axis=1) & np.all(pos+1 < im.shape)
        pos = pos[good, :].copy()
    masses = mass_im[pos[:,0], pos[:,1], pos[:,2]].copy()
    return pos, e, masses

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
                continue
            d = np.sqrt(( (z*(pos[i] - pos[j]))**2 ).sum())
            r = rad[i] + rad[j]
            diff = d - r
            if diff < 0:
                rad[i] -= np.abs(diff)*rad[i]/(rad[i]+rad[j]) + 1e-10
                rad[j] -= np.abs(diff)*rad[j]/(rad[i]+rad[j]) + 1e-10
                if doprint:
                    print diff, rad[i], rad[j]

def otsu_threshold(data, bins=255):
    """
    Otsu threshold on data.

    Otsu thresholding [1]_is a method for selecting an intensity value
    for thresholding an image into foreground and background. The sel-
    ected intensity threshold maximizes the inter-class variance.

    Parameters
    ----------
        data : numpy.ndarray
            The data to threshold
        bins : Int or numpy.ndarray, optional
            Bin edges, as passed to numpy.histogram

    Returns
    -------
        numpy.float
            The value of the threshold which maximizes the inter-class
            variance.

    Notes
    -----
        This could be generalized to more than 2 classes.
    References
    ----------
        ..[1] N. Otsu, "A Threshold Selection Method from Gray-level
            Histograms," IEEE Trans. Syst., Man, Cybern., Syst., 9, 1,
            62-66 (1979)
    """
    h0, x0 = np.histogram(data.ravel(), bins=bins)
    h = h0.astype('float') / h0.sum()  #normalize
    x = 0.5*(x0[1:] + x0[:-1])  #bin center
    wk = np.array([h[:i+1].sum() for i in xrange(h.size)])  #omega_k
    mk = np.array([sum(x[:i+1]*h[:i+1]) for i in xrange(h.size)])  #mu_k
    mt = mk[-1]  #mu_T
    sb = (mt*wk - mk)**2 / (wk*(1-wk) + 1e-15)  #sigma_b
    ind = sb.argmax()
    return 0.5*(x0[ind] + x0[ind+1])

def harris_feature(im, region_size=5, to_return='matrix', scale=0.05):
    """
    Harris-motivated feature detection on a d-dimensional image.


    """
    ndim = im.ndim
    #1. Gradient of image
    grads = (nd.sobel(fim, axis=i) for i in xrange(ndim))
    #2. Corner response matrix
    matrix = np.zeros((ndim, ndim) + im.shape)
    for a in xrange(ndim):
        for b in xrange(ndim):
            matrix[a,b] = nd.filters.gaussian_filter(grads[a]*grads[b],
                    region_size)
    if to_return == 'matrix':
        return matrix
    #3. Trace, determinant
    trc = np.trace(matrix, axis1=0, axis2=1)
    det = np.linalg.det(matrix.T).T
    #4. Harris detector:
    harris = det - scale*trc*trc
    return harris

def identify_slab(im, sigma=5., mode='grad', region_size=10):
    """
    Identifies slabs in an image... doesn't work yet.

    Functions by running an edge detection on the image, thresholding
    the edge, then clustering.
    mode = 'grad' -- gradient filters image
    sigma : Gaussian sigma for gaussian gradient
    region_size : size of region for Harris corner...
    """
    #1. edge detect:
    mode = mode.lower()
    if mode == 'grad':
        gim = nd.filters.gaussian_gradient_magnitude(im, sigma)
    elif mode == 'harris':
        fim = nd.filters.gaussian_filter(im, sigma)

        #All part of a harris corner feature....
        ndim = im.ndim
        grads = [nd.sobel(im, axis=i) for i in xrange(ndim)]
        matrix = np.zeros((ndim, ndim) + im.shape)
        for a in xrange(ndim):
            for b in xrange(ndim):
                matrix[a,b] = nd.filters.gaussian_filter(grads[a]*grads[b],
                        region_size)
        trc = np.trace(matrix, axis1=0, axis2=1)
        det = np.linalg.det(matrix.T).T
        #..done harris
        #we want an edge == not a corner, so one eigenvalue is high and
        #one is low.
        #So -- trc high, det low:
        trc_cut = otsu_threshold(trc)
        det_cut = otsu_threshold(det)
        slabs = (trc > trc_cut) & (det < det_cut)
        pass
    else:
        raise ValueError('mode must be one of `grad`, `harris`')
    pass
