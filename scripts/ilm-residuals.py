import pickle
import numpy as np
import scipy as sp

from cbamf.viz.plots import lbl
from cbamf.comp import ilms
from cbamf.initializers import load_tiff, normalize

import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid

ilm_file = "/media/scratch/bamf/illumination_field.tif"

def load_background_image():
    return normalize(load_tiff(ilm_file))

def do_fit(im, order=(15,5,5), maxcalls=50):
    ilm = ilms.LegendrePoly2P1D(im.shape, order=order)
    ilm.params[-order[-1]] = 1
    ilm.from_data(im, maxcalls=maxcalls)
    ilm.initialize()
    return ilm

def do_fits():
    diffs = []
    orders = ((3,3,2), (7,5,2), (11,7,2), (17,11,2), (35,19,2))

    d = load_background_image()[10][None,:,:]
    for order in orders:
        ilm = do_fit(d, order=order)
        pickle.dump(ilm, open(ilm_file+"-leg2p1d-"+"_".join([str(a) for a in order])+".pkl", 'w'))
        
        diffs.append(ilm.get_field() - d)

    return np.array(diffs), orders

def doplot(diffs, orders):
    fig = pl.figure(figsize=(16,6))
    gs = ImageGrid(fig, rect=[0.05, 0.05, 0.90, 0.90], nrows_ncols=(1,len(diffs)), axes_pad=0.05)

    scale = 0.5*np.abs(diffs).max()
    zslice = diffs[0].shape[0]/2

    for i, (diff, order) in enumerate(zip(diffs, orders)):
        ax = gs[i]
        ax.imshow(diff[zslice], vmin=-scale, vmax=scale, cmap=pl.cm.RdBu)
        ax.set_title(str(order))
        ax.set_xticks([])
        ax.set_yticks([])

def plot_noise(diff):
    dat = diff.flatten()
    sig = dat.std()

    fig = pl.figure(figsize=(13,4))

    #=====================================================
    gs2 = GridSpec(1,2, left=0.05, bottom=0.22, right=0.45, top=0.85,
                wspace=0.25, hspace=0.25)

    ax = pl.subplot(gs2[0,0])
    ax.imshow(diff[0], cmap=pl.cm.bone)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False, which='both', axis='both')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    lbl(ax, 'A')

    q = np.fft.fftn(diff)
    q = np.fft.fftshift(np.abs(q[0])**0.1)

    ax = pl.subplot(gs2[0,1])
    ax.imshow(q, cmap=pl.cm.bone)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False, which='both', axis='both')
    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    lbl(ax, 'B')

    #=====================================================
    gs = GridSpec(1,1, left=0.59, bottom=0.22, right=0.89, top=0.85,
                wspace=0.35, hspace=0.35)

    y,x = np.histogram(dat, bins=np.linspace(-5*sig, 5*sig, 300), normed=True)
    x = (x[1:] + x[:-1])/2

    ax = pl.subplot(gs[0,0])
    ax.plot(x, y, '-', lw=2, alpha=0.6, label='Confocal image')
    ax.plot(x, 1/np.sqrt(2*np.pi*sig**2) * np.exp(-x**2/(2*sig**2)), 'k--', lw=1.5, label='Gaussian fit')
    ax.semilogy()

    ymin = y[y>0].min()
    ymax = y[y>0].max()

    #ax.legend(loc='best', bbox_to_anchor=(0.75, 0.25), prop={'size':18})
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlim(-5*sig, 5*sig)
    ax.set_ylim(ymin, 1.35*ymax)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Probability")
    ax.grid(False, which='minor', axis='y')
    lbl(ax, 'C')

    return gs
