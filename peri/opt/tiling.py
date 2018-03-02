from builtins import zip

import numpy as np
from scipy.cluster import vq
from scipy.optimize import fmin_tnc, fmin_cobyla

from peri import util
from peri.logger import log

# two ideas:
# 1) hierarchical -- add a tile for each large object and find if the smaller
#    ones fit. and move downward adding more tiles if there are things without
#    tiles
# 2) decide on uniform tiling and just find the best shape with neighborlists

def mem2pix(s, max_mem=1e9, max_decimation=1.0):
    nbytes = s.residuals.nbytes // s.residuals.size
    return max_mem // nbytes // max_decimation // 4

def parameter_tiles(s, params):
    return [s.get_update_io_tiles(p, s.get_values(p)+1e-5)[1] for p in params]

def tile_overlap(inner, outer, norm=False):
    """ How much of inner is in outer by volume """
    div = 1.0/inner.volume if norm else 1.0
    return div*(inner.volume - util.Tile.intersection(inner, outer).volume)

#=============================================================================
# Functions for determine uniform tiling for a set of tiles
#=============================================================================
def create_tiling(s, size=40, shift=0):
    imtile = s.oshape.translate(-s.pad)

    region = util.Tile(size, dim=s.dim)
    trange = np.ceil(imtile.shape.astype('float') / region.shape)

    translations = util.Tile(trange).coords(form='vector')
    translations = translations.reshape(-1, translations.shape[-1])

    return  [
        region.copy().translate(region.shape * v - s.pad + shift)
        for v in translations
    ]

def closest_uniform_tile(s, shift, size, other):
    """
    Given a tiling of space (by state, shift, and size), find the closest
    tile to another external tile
    """
    region = util.Tile(size, dim=s.dim, dtype='float').translate(shift - s.pad)
    vec = np.round((other.center - region.center) / region.shape)
    return region.translate(region.shape * vec)

def cost_uniform_tiling(params, s, ptiles):
    D = len(params) // 2

    #x0, region = params[:D], params[D:] # FIXME

    x0, region, vol = params[:D], params[D:-1], params[-1]
    region = np.hstack([region, [vol / np.prod(region)]])

    cost = 0
    for p in ptiles:
        itile = closest_uniform_tile(s, x0, region, p)
        cost += tile_overlap(p, itile)
    log.debug('{} {} {}'.format(x0, region, np.prod(region), cost))
    return cost

def cost_volume_constraint(x, imsize, max_pix):
    D = len(x) // 2
    x0, region = x[:D], x[D:]
    log.debug('{}'.format(np.hstack([[max_pix - np.prod(region)], imsize - region, region])))
    return -np.prod(np.hstack([[max_pix - np.prod(region)], imsize - region]))

def best_tiling_uniform(s, params, max_mem=1e9, decimation=1.0):
    max_pix = mem2pix(s, max_mem, decimation)
    tiles = parameter_tiles(s, params)

    # get some initial conditions (square of the maximum volume)
    side = np.power(max_pix/2.0, 1.0/len(s.residuals.shape))

    shift0 = np.zeros_like(tiles[0].shape)
    size0 = util.amin([side]*len(s.residuals.shape), s.residuals.shape)
    x0 = np.hstack([shift0, size0])

    #print x0 # FIXME
    #return fmin_cobyla(
    #    cost_uniform_tiling, x0, cost_volume_constraint,
    #    args=(s, tiles), consargs=(s.residuals.shape, max_pix,)
    #)

    # set up the boundaries. the only one we have is on the volume
    x0[-1] = max_pix
    bounds_x = [[-100, 100] for x in shift0]
    bounds_s = [[10, s.residuals.shape[i]] for i,_ in enumerate(size0)]
    bounds_s[-1] = [10, max_pix]

    bounds = bounds_x + bounds_s
    log.debug('{}'.format(bounds))

    return fmin_tnc(
        cost_uniform_tiling, x0, args=(s, tiles), bounds=bounds,
        approx_grad=True, disp=5, epsilon=1e-5
    )

#=============================================================================
# Functions for determine hierarchical tiling for a set of tiles
#=============================================================================
def cluster_tiles_by_volume(tiles, volumes, nclusters, nattempts, max_pix=None):
    max_pix = max_pix or np.inf

    # cluster the tiles by volume
    logvol = np.log10(volumes)
    centers = vq.kmeans(logvol, nclusters)[0]
    labels = vq.vq(logvol, centers)[0]
    ids = np.arange(labels.max())

    # get the centers in order so we can walk down the list
    centers, ids = (
        list(t) for t in zip(*sorted(zip(centers, ids), reverse=True))
    )

    # get the groups that are viable based on memory constraints
    grouped_labels = [
        labels[labels==i] for i in ids if volumes[labels==i].max() < max_pix
    ]
    return grouped_labels
    """

    # do hierarchical clustering starting with the largest sizes
    for 
    grp = groups[centers.argmax()]
    return tiles, volumes
    """

def best_tiling_hierarchical(s, params, max_mem=1e9, decimation=1.0):
    tiles = parameter_tiles(s)
    volms = np.array([t.volume for t in tiles]).astype('float')


#=============================================================================
# Interface functions to find tiling of space
#=============================================================================
def spatial_parameter_groups(s, params, max_mem=1e9, max_decimation=1.0, **kwargs):
    """
    Given a state and a list of parameters, spatially group the parameters into
    small regions which have overlapping influence so that they may be optimized
    concurrently.

    Parameters:
    -----------
    s : state
        A PERI state which contains parameters

    parameters : list of strings
        List of parameters which to group together

    Returns:
    --------
    groups : list of list of strings
        List of grouping of parameters. Each parameter will appear exactly once
        in the resulting groups.
    """
    # get the regions of the image that update based on each parameter update
    # by asking the state what that region is (includes psf, etc, for example)

    # FIXME -- magic factor from calc_particle_group_region_size?


def separate_particles_into_groups(s, region_size=40, bounds=None):
    """
    Given a state, returns a list of groups of particles. Each group of
    particles are located near each other in the image. Every particle
    located in the desired region is contained in exactly 1 group.

    Parameters:
    -----------
    s : state
        The PERI state to find particles in.

    region_size: int or list of ints
        The size of the box. Groups particles into boxes of shape region_size.
        If region_size is a scalar, the box is a cube of length region_size.
        Default is 40.

    bounds: 2-element list-like of 3-element lists.
        The sub-region of the image over which to look for particles.
            bounds[0]: The lower-left  corner of the image region.
            bounds[1]: The upper-right corner of the image region.
        Default (None -> ([0,0,0], s.oshape.shape)) is a box of the entire
        image size, i.e. the default places every particle in the image
        somewhere in the groups.

    Returns:
    -----------
    particle_groups: list
        Each element of particle_groups is an int numpy.ndarray of the
        group of nearby particles. Only contains groups with a nonzero
        number of particles, so the elements don't necessarily correspond
        to a given image region.
    """
    imtile = (
        s.oshape.translate(-s.pad) if bounds is None else
        util.Tile(bounds[0], bounds[1])
    )

    # does all particle including out of image, is that correct?
    region = util.Tile(region_size, dim=s.dim)
    trange = np.ceil(imtile.shape.astype('float') / region.shape)

    translations = util.Tile(trange).coords(form='vector')
    translations = translations.reshape(-1, translations.shape[-1])

    groups = []
    positions = s.obj_get_positions()
    for v in translations:
        tmptile = region.copy().translate(region.shape * v - s.pad)
        groups.append(find_particles_in_tile(positions, tmptile))

    return [g for g in groups if len(g) > 0]


