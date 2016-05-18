import re
import glob
import numpy as np
import scipy as sp

from peri.priors import overlap
from peri.util import Tile
from peri.comp.objs import SphereCollectionRealSpace

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

def pr(state, samples, depth=-10):
    """
    You should use get_good_pos_rad instead
    """
    d = state.todict(samples)
    p = d['pos'][depth:].mean(axis=0)
    r = d['rad'][depth:].mean(axis=0)
    return p,r

def pos_rad(state, mask):
    """
    Gets all positions and radii of particles by mask
    """
    return state.obj.pos[mask], state.obj.rad[mask]

def good_particles(state, inbox=True, inboxrad=False):
    """
    Returns a mask of `good' particles as defined by
        * radius > 0
        * position inside box
        * active

    Parameters:
        inbox : whether to only count particle centers within the image
        inboxrad : whether to only count particles that overlap the image at all
    """
    pos = state.obj.pos
    rad = state.obj.rad

    mask = rad > 0
    mask &= (state.obj.typ == 1.)

    if inbox:
        if inboxrad:
            mask &= trim_box(state, pos, rad=rad)
        else:
            mask &= trim_box(state, pos, rad=None)

    return mask
    
def get_good_pos_rad(state, samples, depth=-10, return_err=False):
    """
    Returns the sampled positions, radii of particles that are:
        * radius > 0 
        * position inside box
        * active
    """
    
    d = state.todict(samples)
    p = d['pos'][depth:].mean(axis=0)
    r = d['rad'][depth:].mean(axis=0)
    if return_err:
        er_p = d['pos'][depth:].std(axis=0)
        er_r = d['rad'][depth:].std(axis=0)
        
    # m = good_particles(state, inbox=True, inboxrad=False)
    m = (r > 0) & trim_box(state, p)
    
    p -= state.pad
    if return_err:
        return p[m].copy(), r[m].copy(), er_p[m].copy(), er_r[m].copy()
    else:
        return p[m].copy(), r[m].copy()

def states_to_DataFrame(state_list):
    #FIXME
    raise NotImplementedError

def trim_box(state, p, rad=None):
    """
    Returns particles within the image.  If rad is provided, then
    particles that intersect the image at all (p-r) > edge
    """
    if rad is None:
        return ((p > state.pad) & (p < np.array(state.image.shape) - state.pad)).all(axis=-1)
    return ((p+rad[:,None] > state.pad) & (p-rad[:,None] < np.array(state.image.shape) - state.pad)).all(axis=-1)

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

def iter_pos_rad(state, samples):
    for sample in samples:
        pos = sample[state.b_pos].reshape(-1,3)
        rad = sample[state.b_rad]
        yield pos, rad

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

        diff = (d-r) / r
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

    phi_method : str ['pos', 'obj', 'state']
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

    if phi_method == 'pos':
        phi = packing_fraction(pos, rad)
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
    obj = SphereCollectionRealSpace(pos, rad, shape=shape)
    obj.initialize(zscale)
    return obj.particles[inner].mean()

def packing_fraction_state(state):
    return state.obj.particles[state.inner].mean()

def packing_fraction(pos, rad, bounds=None, state=None, full_output=False):
    if state is not None:
        bounds = Tile(left=state.pad,
                right=np.array(state.image.shape)-state.pad)
    else:
        if bounds is None:
            bounds = Tile(pos.min(axis=0), pos.max(axis=0))

    box_volume = np.prod(bounds.r - bounds.l)
    particle_volume = 0.0
    nparticles = 0

    for p,r in zip(pos, rad):
        if (p > bounds.l-r).all() and (p < bounds.r+r).all():
            vol = 4*np.pi/3 * r**3

            d0 = p + r - bounds.l
            d1 = bounds.r - p - r

            d0 = d0[d0<0]
            d1 = d1[d1<0]

            vol += (np.pi*d0/6 * (3*r**2 + d0**2)).sum()
            vol += (np.pi*d1/6 * (3*r**2 + d1**2)).sum()

            particle_volume += vol
            nparticles += 1

    if full_output:
        return particle_volume / box_volume, nparticles
    return particle_volume / box_volume

def average_packing_fraction(state, samples):
    phi = []

    for p,r in iter_pos_rad(state, samples):
        phi.append(packing_fraction(p,r,state=state))

    phi = np.array(phi)

    return phi.mean(axis=0)[0], phi.std(axis=0)[0]/np.sqrt(len(phi))
