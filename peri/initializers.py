from builtins import range

import glob
import itertools
from PIL import Image

import numpy as np
import scipy.ndimage as nd

from peri.logger import log
log = log.getChild("initializers")

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
        minmass=1., trim_edge=False):
    """Local max featuring to identify bright spherical particles on a
    dark background.

    Parameters
    ----------
        im : numpy.ndarray
            The image to identify particles in.
        radius : Float > 0, optional
            Featuring radius of the particles. Default is 2.5
        noise_size : Float, optional
            Size of Gaussian kernel for smoothing out noise. Default is 1.
        bkg_size : Float or None, optional
            Size of the Gaussian kernel for removing long-wavelength
            background. Default is None, which gives `2 * radius`
        minmass : Float, optional
            Return only particles with a ``mass > minmass``. Default is 1.
        trim_edge : Bool, optional
            Set to True to omit particles identified exactly at the edge
            of the image. False-positive features frequently occur here
            because of the reflected bandpass featuring. Default is
            False, i.e. find particles at the edge of the image.

    Returns
    -------
        pos, mass : numpy.ndarray
            Particle positions and masses
    """
    if radius <= 0:
        raise ValueError('`radius` must be > 0')
    #1. Remove noise
    filtered = nd.gaussian_filter(im, noise_size, mode='mirror')
    #2. Remove long-wavelength background:
    if bkg_size is None:
        bkg_size = 2*radius
    filtered -= nd.gaussian_filter(filtered, bkg_size, mode='mirror')
    #3. Local max feature
    footprint = generate_sphere(radius)
    e = nd.maximum_filter(filtered, footprint=footprint)
    mass_im = nd.convolve(filtered, footprint, mode='mirror')
    good_im = (e==filtered) * (mass_im > minmass)
    pos = np.transpose(np.nonzero(good_im))
    if trim_edge:
        good = np.all(pos > 0, axis=1) & np.all(pos+1 < im.shape, axis=1)
        pos = pos[good, :].copy()
    masses = mass_im[pos[:,0], pos[:,1], pos[:,2]].copy()
    return pos, masses

def trackpy_featuring(im, size=10):
    from trackpy.feature import locate
    size = size + (size+1) % 2
    a = locate(im, size, invert=True)
    pos = np.vstack([a.z, a.y, a.x]).T
    return pos

def remove_overlaps(pos, rad, zscale=1, doprint=False):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])
    for i in range(N):
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
                log.info('{} {} {}'.format(diff, rad[i], rad[j]))

def remove_overlaps_naive(pos, rad, zscale=1, doprint=False):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.sqrt(( (z*(pos[i] - pos[j]))**2 ).sum())
            r = rad[i] + rad[j]
            diff = d - r
            if diff < 0:
                rad[i] -= np.abs(diff)*rad[i]/(rad[i]+rad[j]) + 1e-10
                rad[j] -= np.abs(diff)*rad[j]/(rad[i]+rad[j]) + 1e-10
                if doprint:
                    log.info('{} {} {}'.format(diff, rad[i], rad[j]))

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
    wk = np.array([h[:i+1].sum() for i in range(h.size)])  #omega_k
    mk = np.array([sum(x[:i+1]*h[:i+1]) for i in range(h.size)])  #mu_k
    mt = mk[-1]  #mu_T
    sb = (mt*wk - mk)**2 / (wk*(1-wk) + 1e-15)  #sigma_b
    ind = sb.argmax()
    return 0.5*(x0[ind] + x0[ind+1])

def harris_feature(im, region_size=5, to_return='harris', scale=0.05):
    """
    Harris-motivated feature detection on a d-dimensional image.

    Parameters
    ---------
        im
        region_size
        to_return : {'harris','matrix','trace-determinant'}

    """
    ndim = im.ndim
    #1. Gradient of image
    grads = [nd.sobel(im, axis=i) for i in range(ndim)]
    #2. Corner response matrix
    matrix = np.zeros((ndim, ndim) + im.shape)
    for a in range(ndim):
        for b in range(ndim):
            matrix[a,b] = nd.filters.gaussian_filter(grads[a]*grads[b],
                    region_size)
    if to_return == 'matrix':
        return matrix
    #3. Trace, determinant
    trc = np.trace(matrix, axis1=0, axis2=1)
    det = np.linalg.det(matrix.T).T
    if to_return == 'trace-determinant':
        return trc, det
    else:
        #4. Harris detector:
        harris = det - scale*trc*trc
        return harris

def identify_slab(im, sigma=5., region_size=10, masscut=1e4, asdict=False):
    """
    Identifies slabs in an image.

    Functions by running a Harris-inspired edge detection on the image,
    thresholding the edge, then clustering.

    Parameters
    ----------
        im : numpy.ndarray
            3D array of the image to analyze.
        sigma : Float, optional
            Gaussian blurring kernel to remove non-slab features such as
            noise and particles. Default is 5.
        region_size : Int, optional
            The size of region for Harris corner featuring. Default is 10
        masscut : Float, optional
            The minimum number of pixels for a feature to be identified as
            a slab. Default is 1e4; should be smaller for smaller images.
        asdict : Bool, optional
            Set to True to return a list of dicts, with keys of ``'theta'``
            and ``'phi'`` as rotation angles about the x- and z- axes, and
            of ``'zpos'`` for the z-position, i.e. a list of dicts which
            can be unpacked into a :class:``peri.comp.objs.Slab``

    Returns
    -------
        [poses, normals] : numpy.ndarray
            The positions and normals of each slab in the image; ``poses[i]``
            and ``normals[i]`` are the ``i``th slab. Returned if ``asdict``
            is False
        [list]
            A list of dictionaries. Returned if ``asdict`` is True
    """
    #1. edge detect:
    fim = nd.filters.gaussian_filter(im, sigma)
    trc, det = harris_feature(fim, region_size, to_return='trace-determinant')
    #we want an edge == not a corner, so one eigenvalue is high and
    #one is low compared to the other.
    #So -- trc high, normalized det low:
    dnrm = det / (trc*trc)
    trc_cut = otsu_threshold(trc)
    det_cut = otsu_threshold(dnrm)
    slabs = (trc > trc_cut) & (dnrm < det_cut)
    labeled, nslabs = nd.label(slabs)
    #masscuts:
    masses = [(labeled == i).sum() for i in range(1, nslabs+1)]
    good = np.array([m > masscut for m in masses])
    inds = np.nonzero(good)[0] + 1  #+1 b/c the lowest label is the bkg
    #Slabs are identifiied, now getting the coords:
    poses = np.array(nd.measurements.center_of_mass(trc, labeled, inds))
    #normals from eigenvectors of the covariance matrix
    normals = []
    z = np.arange(im.shape[0]).reshape(-1,1,1).astype('float')
    y = np.arange(im.shape[1]).reshape(1,-1,1).astype('float')
    x = np.arange(im.shape[2]).reshape(1,1,-1).astype('float')
    #We also need to identify the direction of the normal:
    gim = [nd.sobel(fim, axis=i) for i in range(fim.ndim)]
    for i, p in zip(range(1, nslabs+1), poses):
        wts = trc * (labeled == i)
        wts /= wts.sum()
        zc, yc, xc = [xi-pi for xi, pi in zip([z,y,x],p.squeeze())]
        cov = [[np.sum(xi*xj*wts) for xi in [zc,yc,xc]] for xj in [zc,yc,xc]]
        vl, vc = np.linalg.eigh(cov)
        #lowest eigenvalue is the normal:
        normal = vc[:,0]
        #Removing the sign ambiguity:
        gn = np.sum([n*g[tuple(p.astype('int'))] for g,n in zip(gim, normal)])
        normal *= np.sign(gn)
        normals.append(normal)
    if asdict:
        get_theta = lambda n: -np.arctan2(n[1], -n[0])
        get_phi = lambda n: np.arcsin(n[2])
        return [{'zpos':p[0], 'angles':(get_theta(n), get_phi(n))}
                for p, n in zip(poses, normals)]
    else:
        return poses, np.array(normals)
