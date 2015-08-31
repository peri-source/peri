import matplotlib as mpl
import matplotlib.pylab as pl

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Circle, Rectangle

from cbamf.test import analyze
from cbamf import util

import numpy as np
import time
import pickle

def trim_box(state, p):
    return ((p > state.pad) & (p < np.array(state.image.shape) - state.pad)).all(axis=-1)

def summary_plot(state, samples, zlayer=None, xlayer=None, truestate=None):
    def MAD(d):
        return np.median(np.abs(d - np.median(d)))

    s = state
    t = s.get_model_image()

    if zlayer is None:
        zlayer = t.shape[0]/2

    if xlayer is None:
        xlayer = t.shape[2]/2

    mu = samples.mean(axis=0)
    std = samples.std(axis=0)

    fig, axs = pl.subplots(3,3, figsize=(20,12))
    axs[0][0].imshow(s.image[zlayer], vmin=0, vmax=1)
    axs[0][1].imshow(t[zlayer], vmin=0, vmax=1)
    axs[0][2].imshow((s.image-t)[zlayer], vmin=-1, vmax=1)
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])
    axs[0][2].set_xticks([])
    axs[0][2].set_yticks([])

    axs[1][0].imshow(s.image[:,:,xlayer], vmin=0, vmax=1)
    axs[1][1].imshow(t[:,:,xlayer], vmin=0, vmax=1)
    axs[1][2].imshow((s.image-t)[:,:,xlayer], vmin=-1, vmax=1)
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])
    axs[1][2].set_xticks([])
    axs[1][2].set_yticks([])

    try:
        alpha = 0.5 if truestate is not None else 0.8
        axs[2][0].hist(std[s.b_rad], bins=np.logspace(-3,0,50), label='Radii',
                histtype='stepfilled', alpha=alpha, color='red')
        if truestate is not None:
            d = np.abs(mu - truestate)
            axs[2][0].hist(d[s.b_pos], bins=np.logspace(-3,0,50), color='red',
                    histtype='step', alpha=1)
        axs[2][0].semilogx()

        axs[2][0].hist(std[s.b_pos], bins=np.logspace(-3,0,50), label='Positions',
                histtype='stepfilled', alpha=alpha, color='blue')
        if truestate is not None:
            d = np.abs(mu - truestate)
            axs[2][0].hist(d[s.b_rad], bins=np.logspace(-3,0,50), color='blue',
                    histtype='step', alpha=1)
        axs[2][0].semilogx()
        axs[2][0].legend(loc='upper right')
        axs[2][0].set_xlabel("Estimated standard deviation")
        axs[2][0].set_ylim(bottom=0)
    except Exception as e:
        pass

    d = s.state[s.b_rad]
    m = 2*1.4826 * MAD(d)
    mb = d.mean()

    d = d[(d > mb - m) & (d < mb +m)]
    d = s.state[s.b_rad]
    axs[2][1].hist(d, bins=50, histtype='stepfilled', alpha=0.8)
    axs[2][1].set_xlabel("Radii")
    axs[2][1].set_ylim(bottom=0)

    if truestate is not None:
        axs[2][1].hist(truestate[s.b_rad], bins=50, histtype='step', alpha=0.8)

    axs[2][2].hist((s.image-t)[s.image_mask==1].ravel(), bins=150,
            histtype='stepfilled', alpha=0.8)
    axs[2][2].set_xlim(-0.35, 0.35)
    axs[2][2].semilogy()
    axs[2][2].set_ylim(bottom=0)
    axs[2][2].set_xlabel("Pixel value differences")

    pl.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.05)
    pl.tight_layout()

def pretty_summary(state, samples, zlayer=None, xlayer=None, vertical=False):
    s = state
    h = np.array(samples)

    slicez = zlayer or s.image.shape[0]/2
    slicex = xlayer or s.image.shape[2]/2
    slicer1 = np.s_[slicez,s.pad:-s.pad,s.pad:-s.pad]
    slicer2 = np.s_[s.pad:-s.pad,s.pad:-s.pad,slicex]
    center = (slicez, s.image.shape[1]/2, slicex)

    if vertical:
        fig = pl.figure(figsize=(12,24))
    else:
        fig = pl.figure(figsize=(24,8))

    #=========================================================================
    #=========================================================================
    if vertical:
        gs1 = ImageGrid(fig, rect=[0.02, 0.55, 0.99, 0.40], nrows_ncols=(2,3), axes_pad=0.1)
    else:
        gs1 = ImageGrid(fig, rect=[0.02, 0.0, 0.4, 1.00], nrows_ncols=(2,3), axes_pad=0.1)

    for i,slicer in enumerate([slicer1, slicer2]):
        ax_real = gs1[3*i+0]
        ax_fake = gs1[3*i+1]
        ax_diff = gs1[3*i+2]

        diff = s.get_model_image() - s.image
        ax_real.imshow(s.image[slicer], cmap=pl.cm.bone_r)
        ax_real.set_xticks([])
        ax_real.set_yticks([])
        ax_fake.imshow(s.get_model_image()[slicer], cmap=pl.cm.bone_r)
        ax_fake.set_xticks([])
        ax_fake.set_yticks([])
        ax_diff.imshow(diff[slicer], cmap=pl.cm.RdBu, vmin=-1.0, vmax=1.0)
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])

        if i == 0:
            ax_real.set_title("Confocal image", fontsize=24)
            ax_fake.set_title("Model image", fontsize=24)
            ax_diff.set_title("Difference", fontsize=24)
            ax_real.set_ylabel('x-y')
        else:
            ax_real.set_ylabel('x-z')

    #=========================================================================
    #=========================================================================
    mu = h.mean(axis=0)
    std = h.std(axis=0)

    if vertical:
        gs2 = GridSpec(2,2, left=0.10, bottom=0.10, right=0.99, top=0.52,
                wspace=0.45, hspace=0.45)
    else:
        gs2 = GridSpec(2,2, left=0.50, bottom=0.12, right=0.95, top=0.95,
                wspace=0.35, hspace=0.35)
    
    ax_hist = pl.subplot(gs2[0,0])
    ax_hist.hist(std[s.b_pos], bins=np.logspace(-2.5, 0, 50), alpha=0.7, label='POS', histtype='stepfilled')
    ax_hist.hist(std[s.b_rad], bins=np.logspace(-2.5, 0, 50), alpha=0.7, label='RAD', histtype='stepfilled')
    ax_hist.set_xlim((10**-2.4, 1))
    ax_hist.semilogx()
    ax_hist.set_xlabel(r"$\bar{\sigma}$")
    ax_hist.set_ylabel(r"$P(\bar{\sigma})$")
    ax_hist.legend(loc='upper right')

    ax_diff = pl.subplot(gs2[0,1])
    ax_diff.hist((s.get_model_image() - s.image)[s.image_mask==1.].ravel(), bins=1000, histtype='stepfilled', alpha=0.7)
    ax_diff.semilogy()
    ax_diff.set_ylabel(r"$P(\delta)$")
    ax_diff.set_xlabel(r"$\delta = M_i - d_i$")

    pos = mu[s.b_pos].reshape(-1,3)
    rad = mu[s.b_rad]
    mask = trim_box(s, pos)
    pos = pos[mask]
    rad = rad[mask]

    gx, gy = analyze.gofr_full(pos, rad, mu[s.b_zscale][0], resolution=5e-2,mask_start=0.5)
    mask = gx < 5
    gx = gx[mask]
    gy = gy[mask]
    gy /= gy[-1]
    ax_gofr = pl.subplot(gs2[1,0])
    ax_gofr.plot(gx, gy, '-', lw=1)
    ax_gofr.set_xlabel(r"$r/a$")
    ax_gofr.set_ylabel(r"$g(r/a)$")
    #ax_gofr.semilogy()

    gx, gy = analyze.gofr_full(pos, rad, mu[s.b_zscale][0], method=analyze.gofr_surfaces)
    mask = gx < 5
    gx = gx[mask]
    gy = gy[mask]
    gy /= gy[-1]
    ax_gofrs = pl.subplot(gs2[1,1])
    ax_gofrs.plot(gx, gy, '-', lw=1)
    ax_gofrs.set_xlabel(r"$r/a$")
    ax_gofrs.set_ylabel(r"$g_{\rm{surface}}(r/a)$")
    ax_gofrs.semilogy()

def scan(im, cycles=1, sleep=0.3, vmin=0, vmax=1, cmap='bone'):
    pl.figure(1)
    pl.show()
    time.sleep(3)
    for c in xrange(cycles):
        for i, sl in enumerate(im):
            print i
            pl.clf()
            pl.imshow(sl, cmap=cmap, interpolation='nearest',
                    origin='lower', vmin=vmin, vmax=vmax)
            pl.draw()
            time.sleep(sleep)

def scan_together(im, p, delay=2, vmin=0, vmax=1, cmap='bone'):
    pl.figure(1)
    pl.show()
    time.sleep(3)
    z,y,x = p.T
    for i in xrange(len(im)):
        print i
        sl = im[i]
        pl.clf()
        pl.imshow(sl, cmap=cmap, interpolation='nearest', origin='lower',
                vmin=vmin, vmax=vmax)
        m = z.astype('int') == i
        pl.plot(x[m], y[m], 'o')
        pl.xlim(0, sl.shape[0])
        pl.ylim(0, sl.shape[1])
        pl.draw()
        time.sleep(delay)

def sample_compare(N, samples, truestate, burn=0):
    h = samples[burn:]
    strue = truestate

    mu = h.mean(axis=0)
    std = h.std(axis=0)
    pl.figure(figsize=(20,4))
    pl.errorbar(xrange(len(mu)), (mu-strue), yerr=5*std/np.sqrt(h.shape[0]),
            fmt='.', lw=0.15, alpha=0.5)
    pl.vlines([0,3*N-0.5, 4*N-0.5], -1, 1, linestyle='dashed', lw=4, alpha=0.5)
    pl.hlines(0, 0, len(mu), linestyle='dashed', lw=5, alpha=0.5)
    pl.xlim(0, len(mu))
    pl.ylim(-0.02, 0.02)
    pl.show()


def generative_model(s,x,y,z,r):
    pl.close('all')

    slicez = int(z.mean())
    slicex = s.image.shape[2]/2
    slicer1 = np.s_[slicez,s.pad:-s.pad,s.pad:-s.pad]
    slicer2 = np.s_[s.pad:-s.pad,s.pad:-s.pad,slicex]
    center = (slicez, s.image.shape[1]/2, slicex)

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

    t = s.ilm.get_field().copy()
    t *= 0
    t[center] = 1
    s.psf.set_tile(util.Tile(t.shape))
    psf = s.psf.execute(t)
    print slicer1, slicer2, center

    ax_psf1.imshow(psf[slicer1], cmap=pl.cm.bone)
    ax_psf1.set_xticks([])
    ax_psf1.set_yticks([])
    ax_psf1.set_ylabel("PSF", fontsize=22)
    ax_psf2.imshow(psf[slicer2], cmap=pl.cm.bone)
    ax_psf2.set_xticks([])
    ax_psf2.set_yticks([])

    #=========================================================================
    #=========================================================================
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
    ax_zoom.hexbin(x,y, gridsize=32, mincnt=5, cmap=pl.cm.hot)

    zoom1 = zoomed_inset_axes(ax_zoom, 30, loc=3)
    zoom1.imshow(im, extent=extent, cmap=pl.cm.bone_r)
    zoom1.set_xlim(cx-1.0/6, cx+1.0/6)
    zoom1.set_ylim(cy-1.0/6, cy+1.0/6)
    zoom1.hexbin(x,y,gridsize=32, mincnt=5, cmap=pl.cm.hot)
    zoom1.set_xticks([])
    zoom1.set_yticks([])
    zoom1.hlines(cy-1.0/6 + 1.0/32, cx-1.0/6+5e-2, cx-1.0/6+5e-2+1e-1, lw=3)
    zoom1.text(cx-1.0/6 + 1.0/24, cy-1.0/6+5e-2, '0.1px')
    mark_inset(ax_zoom, zoom1, loc1=2, loc2=4, fc="none", ec="0.5")

    #zoom2 = zoomed_inset_axes(ax_zoom, 10, loc=4)
    #zoom2.imshow(im, extent=extent, cmap=pl.cm.bone_r)
    #zoom2.set_xlim(cx-1.0/2, cx+1.0/2)
    #zoom2.set_ylim(cy-1.0/2, cy+1.0/2)
    #zoom2.hexbin(x,y,gridsize=32, mincnt=1, cmap=pl.cm.hot)
    #zoom2.set_xticks([])
    #zoom2.set_yticks([])
    #mark_inset(zoom1, zoom2, loc1=1, loc2=3, fc="none", ec="0.5")
