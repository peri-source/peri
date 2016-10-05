import re
import glob
import numpy as np
import scipy as sp

from peri.priors import overlap
from peri.util import Tile
from peri.comp.objs import PlatonicSpheresCollection

def sorted_files(globber, num_sort=True, num_indices=None, return_num=False):
    """
    Give a globbing expression of files to find. They will be sorted upon return.
    This function is most useful when sorting does not provide numerical order,
    e.g.:
        9 -> 12 returned as 10 11 12 9 by string sort

    In this case use num_sort=True, and it will be sorted by numbers whose index
    is given by num_indices (possibly None for all numbers) then by string.
    """
    files = glob.glob(globber)
    files.sort()

    if not num_sort:
        return files

    # sort by numbers if desired
    num_indices = num_indices or np.s_[:]
    allfiles = []

    for fn in files:
        nums = re.findall(r'\d+', fn)
        data = [int(n) for n in nums[num_indices]] + [fn]
        allfiles.append(data)

    allfiles = sorted(allfiles)
    if return_num:
        return allfiles
    return [f[-1] for f in allfiles]

def dict_to_pos_rad(d):
    """Given a dictionary of a states params:values, returns the pos & rad."""
    p, r = [], []
    for i in itertools.count():
        try:
            p.append([d['sph-{}-{}'.format(i, c)] for c in 'zyx'])
            r.append(d['sph-{}-a'.format(i)])
        except KeyError:
            break
    return np.array(p), np.array(r)

def good_particles(state, inbox=True, inboxrad=False, fullinbox=False,
        pos=None, rad=None, ishape=None):
    """
    Returns a mask of `good' particles as defined by
        * radius > 0
        * position inside box

    Input Parameters
    ---------------
        state : peri.States instance
            The state to identify the good particles. If pos, rad, and ishape
            are provided, then this does not need to be passed.
        inbox : Bool
            Whether to only count particle centers within the image. Default
            is True.
        inboxrad : Bool
            Whether to only count particles that overlap the image at all.
            Default is False.
        fullinbox : Bool
            Whether to only include particles which are entirely in the
            image. Default is False
        pos : [3,N] np.ndarray or None
            If not None, the particles' positions.
        rad : [N] element numpy.ndarray or None
            If not None, the particles' radii.
        ishape : 3-element list-like or None
            If not None, the inner region of the state.

    Outputs
    -------
        mask : np.ndarray of bools
            A boolean mask of which particles are good (True) or bad.
    """
    if pos is None:
        pos = state.obj_get_positions()
    if rad is None:
        rad = state.obj_get_radii()

    mask = rad > 0

    if (inbox | inboxrad | fullinbox):
        if fullinbox:
            mask &= trim_box(state, pos, rad=-rad, ishape=ishape)
        elif inboxrad:
            mask &= trim_box(state, pos, rad=rad, ishape=ishape)
        else:
            mask &= trim_box(state, pos, rad=None, ishape=ishape)

    return mask

def trim_box(state, p, rad=None, ishape=None):
    """
    Returns particles within the image.  If rad is provided, then
    particles that intersect the image at all (p-r) > edge
    """
    if ishape is None:
        ishape = state.ishape.shape
    if rad is None:
        return ((p > 0) & (p < np.array(ishape))).all(axis=-1)
    return ((p+rad[:,None] > 0) & (p-rad[:,None] < np.array(ishape))).all(axis=-1)

def nearest(p0, p1, cutoff=None):
    """
    Correlate closest particles with each other (within cutoff).
    Returns ind0, ind1 so that p0[ind0] is close to p1[ind1].
    """
    ind0, ind1 = [], []
    for i in xrange(len(p0)):
        dist = np.sqrt(((p0[i] - p1)**2).sum(axis=-1))

        if cutoff is None:
            ind1.append(dist.argmin())

        elif dist.min() < cutoff:
            ind0.append(i)
            ind1.append(dist.argmin())

    if cutoff is None:
        return ind1
    return ind0, ind1

def gofr_normal(pos, rad, zscale):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])

    seps = []
    for i in xrange(N-1):
        o = np.arange(0, N)
        d = np.sqrt( ((z*(pos[i] - pos[o]))**2).sum(axis=-1) )
        seps.extend(d[d!=0])
    return np.array(seps)

def gofr_surfaces(pos, rad, zscale):
    N = rad.shape[0]
    z = np.array([zscale, 1, 1])

    seps = []
    for i in xrange(N-1):
        o = np.arange(0, N)
        d = np.sqrt( ((z*(pos[i] - pos[o]))**2).sum(axis=-1) )
        r = rad[i] + rad[o]

        diff = (d-r)
        seps.extend(diff[diff != 0])
    return np.array(seps)

def gofr(pos, rad, zscale, diameter=None, resolution=3e-2, rmax=10, method='normal',
        normalize=None, mask_start=None, phi_method='const', phi=None, state=None):
    """
    Pair correlation function calculation from 0 to rmax particle diameters

    method : str ['normal', 'surface']
        represents the gofr calculation method

    normalize : boolean
        if None, determined by method, otherwise 1/r^2 norm

    phi_method : str ['obj', 'state']
        which data to use to calculate the packing_fraction.
            -- 'pos' : (not stable) calculate based on fractional spheres in
                a cube, do not use

            -- 'const' : the volume fraction is provided by the user via
                the variable phi

            -- 'state' : the volume is calculated by using the platonic sphere
                image of a given state. must provide argument state
    """

    diameter = diameter or 2*rad.mean()
    vol_particle = 4./3*np.pi*(diameter)**3

    if phi_method == 'const':
        phi = phi or 1
    if phi_method == 'state':
        phi = packing_fraction_state(state)

    num_density = phi / vol_particle

    if method == 'normal':
        normalize = normalize or True
        o = gofr_normal(pos, rad, zscale)
        rmin = 0
    if method == 'surface':
        normalize = normalize or False
        o = diameter*gofr_surfaces(pos, rad, zscale)
        rmin = -1

    bins = np.linspace(rmin, diameter*rmax, diameter*rmax/resolution, endpoint=False)
    y,x = np.histogram(o, bins=bins)
    x = (x[1:] + x[:-1])/2

    if mask_start is not None:
        mask = x > mask_start
        x = x[mask]
        y = y[mask]

    if normalize:
        y = y/(4*np.pi*x**2)
    return x/diameter, y/(resolution * num_density * float(len(rad)))

def packing_fraction_obj(pos, rad, shape, inner, zscale=1):
    obj = PlatonicSpheresCollection(pos, rad, shape=shape, zscale=zscale)
    return obj.get_field()[inner].mean()

def packing_fraction_state(state):
    return state.get('obj').get_field()[state.inner].mean()

def average_packing_fraction(state, samples):
    phi = []

    for p,r in iter_pos_rad(state, samples):
        phi.append(packing_fraction(p,r,state=state))

    phi = np.array(phi)

    return phi.mean(axis=0)[0], phi.std(axis=0)[0]/np.sqrt(len(phi))
