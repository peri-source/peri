import sys
sys.path.append("does_matter")
import numpy as np

import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredText, AnchoredOffsetbox
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid

import common
from cbamf.viz import util
from cbamf.util import Tile
from cbamf.comp.objs import sphere_analytical_gaussian 

def exact_volume(rvec, radius, function=sphere_analytical_gaussian):
    im = function(rvec, radius)

    vol_goal = 4./3*np.pi*radius**3
    rprime = radius

    for i in xrange(10):
        vol_curr = np.abs(im.sum())
        rprime = rprime + 1.0*(vol_goal - vol_curr) / (4*np.pi*rprime**2)
        im = function(rvec, rprime)

        if np.abs(vol_goal - vol_curr)/vol_goal < 1e-10:
            break

    return im

def calc(N=28, scale=41):
    pos = np.array([(N-1)/2.0]*3)
    radius = 5.0
    corner = list((pos-np.array([0] + [radius / np.sqrt(2)]*2) + 0.5).astype("int"))
    
    im,pos,pix = common.perfect_platonic_per_pixel(N, 5, scale=scale, returnpix=corner)

    tile = Tile(im.shape)
    rvec = np.sqrt(((tile.coord_vector() - pos)**2).sum(axis=-1))
    approx = exact_volume(rvec, radius)

    return im, pix, approx, corner

def go():
    doplot3(*calc())

def go2():
    data = calc(scale=81)
    doplot1(data[0], data[1], data[3])

def showpic(im, ax, title='', cmap='bone', c=0.5, d=0.5):
    size = [0, im.shape[0], 0, im.shape[1]]
    ax.imshow(im, extent=size, cmap=cmap, vmin=c-d, vmax=c+d)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, im.shape[0])
    ax.set_ylim(0, im.shape[1])
    ax.set_title(title, fontsize=28)

def doplot1(im, pix, corner):
    fig = pl.figure(figsize=(8,8))
    ax = fig.add_axes([0,0,1,1])
    im_s = im[corner[0]]
    im_p = pix[pix.shape[0]/2]
    showpic(im_s, ax)

    extent2 = [corner[1], corner[1]+1, corner[2], corner[2]+1]
    zoom = zoomed_inset_axes(ax, 8, loc=3)
    zoom.imshow(im_p, extent=extent2, cmap='bone')
    zoom.set_xticks([])
    zoom.set_yticks([])
    mark_inset(ax, zoom, loc1=2, loc2=4, fc="none", ec="1.0")

def doplot3(im, pix, approx, corner):
    fig = pl.figure(figsize=(12,5))
    gs = ImageGrid(fig, rect=[0.025, 0.025, 0.95, 0.95], nrows_ncols=(1,3),
            axes_pad=0.1)

    im_superres = im[corner[0]]
    im_superpix = pix[pix.shape[0]/2]
    im_approx = approx[corner[0]]

    extent = [0,im_superres.shape[0], 0, im_superres.shape[1]]
    showpic(im_superres, gs[0], title='Super-resolution')
    showpic(im_approx, gs[1], title='Approximation')
    showpic(im_superres-im_approx, gs[2], cmap='RdBu', c=0, d=0.5, title='Difference')

    extent2 = [corner[1], corner[1]+1, corner[2], corner[2]+1]
    zoom = zoomed_inset_axes(gs[0], 8, loc=3)
    zoom.imshow(im_superpix, extent=extent2, cmap='bone')
    zoom.set_xticks([])
    zoom.set_yticks([])
    mark_inset(gs[0], zoom, loc1=2, loc2=4, fc="none", ec="1.0")

