import numpy as np
import scipy as sp
from cbamf import runner
import pickle

import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Circle, Rectangle

def sample_center_particle(state):
    cind = state.closet_particle(np.array(state.image.shape)/2)

    blocks = state.blocks_particle(cind)
    hxy = runner.sample_state(state, blocks[1:3], N=5000, doprint=True)
    hr = runner.sample_state(state, [blocks[-1]], N=5000, doprint=True)

    z = state.state[blocks[0]]
    y,x = hh.get_histogram().T
    return x,y,z,r

def load():
    s,h,l = pickle.load(open('./crystal_fcc.tif_t001.tif-fit-gaussian-4d.pkl'))
    x,y,z,r = np.load('./crystal_fcc.tif_t001.tif-fit-gaussian-4d-sample-xyzr.npy').T
    x -= s.pad
    y -= s.pad
    return s,x,y,z,r

def f(s,x,y,z,r):
    pl.close('all')

    slicez = int(z.mean())
    slicex = s.image.shape[2]/2
    slicer1 = np.s_[slicez,s.pad:-s.pad,s.pad:-s.pad]
    slicer2 = np.s_[s.pad:-s.pad,s.pad:-s.pad,slicex]

    fig = pl.figure(figsize=(13,10))

    #=========================================================================
    #=========================================================================
    gs1 = ImageGrid(fig, rect=[0.0, 0.6, 1.0, 0.35], nrows_ncols=(1,3),
            axes_pad=0.1)
    ax_real = gs1[0]
    ax_fake = gs1[1]
    ax_diff = gs1[2]

    diff = s.get_model_image() - s.image
    ax_real.imshow(s.image[slicer1], cmap=pl.cm.bone_r)
    ax_real.set_xticks([])
    ax_real.set_yticks([])
    ax_real.set_title("Confocal image", fontsize=24)
    ax_fake.imshow(s.get_model_image()[slicer1], cmap=pl.cm.bone_r)
    ax_fake.set_xticks([])
    ax_fake.set_yticks([])
    ax_fake.set_title("Model image", fontsize=24)
    ax_diff.imshow(diff[slicer1], cmap=pl.cm.RdBu, vmin=-1.0, vmax=1.0)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    ax_diff.set_title("Difference", fontsize=24)

    #=========================================================================
    #=========================================================================
    gs2 = ImageGrid(fig, rect=[0.1, 0.0, 0.4, 0.55], nrows_ncols=(3,2),
            axes_pad=0.1)
    ax_plt1 = fig.add_subplot(gs2[0])
    ax_plt2 = fig.add_subplot(gs2[1])
    ax_ilm1 = fig.add_subplot(gs2[2])
    ax_ilm2 = fig.add_subplot(gs2[3])
    ax_psf1 = fig.add_subplot(gs2[4])
    ax_psf2 = fig.add_subplot(gs2[5])

    ax_plt1.imshow(1-s.obj.get_field()[slicer1], cmap=pl.cm.bone_r, vmin=0, vmax=1)
    ax_plt1.set_xticks([])
    ax_plt1.set_yticks([])
    ax_plt1.set_ylabel("Platonic", fontsize=22)
    ax_plt1.set_title("x-y", fontsize=24)
    ax_plt2.imshow(1-s.obj.get_field()[slicer2], cmap=pl.cm.bone_r, vmin=0, vmax=1)
    ax_plt2.set_xticks([])
    ax_plt2.set_yticks([])
    ax_plt2.set_title("y-z", fontsize=24)

    ax_ilm1.imshow(s.ilm.get_field()[slicer1], cmap=pl.cm.bone_r)
    ax_ilm1.set_xticks([])
    ax_ilm1.set_yticks([])
    ax_ilm1.set_ylabel("ILM", fontsize=22)
    ax_ilm2.imshow(s.ilm.get_field()[slicer2], cmap=pl.cm.bone_r)
    ax_ilm2.set_xticks([])
    ax_ilm2.set_yticks([])

    ax_psf1.imshow(s.ilm.get_field()[slicer1], cmap=pl.cm.bone_r)
    ax_psf1.set_xticks([])
    ax_psf1.set_yticks([])
    ax_psf1.set_ylabel("PSF", fontsize=22)
    ax_psf2.imshow(s.ilm.get_field()[slicer2], cmap=pl.cm.bone_r)
    ax_psf2.set_xticks([])
    ax_psf2.set_yticks([])

    #=========================================================================
    #=========================================================================
    #gs3 = ImageGrid(fig, rect=[0.48, 0.020, 0.45, 0.55], nrows_ncols=(1,1),
    #        axes_pad=0.1)
    #ax_zoom = gs3[0]
    ax_zoom = fig.add_axes([0.48, 0.018, 0.45, 0.52])

    im = s.image[slicer1]
    sh = np.array(im.shape)
    cx = x.mean()
    cy = y.mean()

    extent = [0,sh[0],0,sh[1]]
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.imshow(im, extent=extent, cmap=pl.cm.bone_r)
    ax_zoom.set_xlim(cx-12, cx+12)
    ax_zoom.set_ylim(cy-12, cy+12)
    ax_zoom.set_title("Sampling", fontsize=24)
    ax_zoom.hexbin(x,y, gridsize=32, mincnt=1, cmap=pl.cm.hot)

    zoom1 = zoomed_inset_axes(ax_zoom, 30, loc=3)
    zoom1.imshow(im, extent=extent, cmap=pl.cm.bone_r)
    zoom1.set_xlim(cx-1.0/6, cx+1.0/6)
    zoom1.set_ylim(cy-1.0/6, cy+1.0/6)
    zoom1.hexbin(x,y,gridsize=32, mincnt=1, cmap=pl.cm.hot)
    zoom1.set_xticks([])
    zoom1.set_yticks([])
    mark_inset(ax_zoom, zoom1, loc1=2, loc2=4, fc="none", ec="0.5")

    #zoom2 = zoomed_inset_axes(ax_zoom, 10, loc=4)
    #zoom2.imshow(im, extent=extent, cmap=pl.cm.bone_r)
    #zoom2.set_xlim(cx-1.0/2, cx+1.0/2)
    #zoom2.set_ylim(cy-1.0/2, cy+1.0/2)
    #zoom2.hexbin(x,y,gridsize=32, mincnt=1, cmap=pl.cm.hot)
    #zoom2.set_xticks([])
    #zoom2.set_yticks([])
    #mark_inset(zoom1, zoom2, loc1=1, loc2=3, fc="none", ec="0.5")

