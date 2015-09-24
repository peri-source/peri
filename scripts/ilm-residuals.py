import pickle
import numpy as np
import scipy as sp

from cbamf.viz.plots import lbl
from cbamf.comp import ilms
from cbamf.initializers import load_tiff, normalize

import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid

ilm_file = "/media/scratch/bamf/illumination_field/illumination_field.tif"

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

def pole_removal(noise, poles=None, sig=3):
    """
    Remove the noise poles from a 2d noise distribution to show that affects
    the real-space noise picture.  
    
    noise -- fftshifted 2d array of q values

    poles -- N,2 list of pole locations. the last index is in the order y,x as
    determined by mpl interactive plots

    for example: poles = np.array([[190,277], [227,253], [233, 256]]
    """
    center = np.array(noise.shape)/2

    v = np.rollaxis(
            np.array(
                np.meshgrid(*(np.arange(s) for s in noise.shape), indexing='ij')
            ), 0, 3
        ).astype("float")
    filter = np.zeros_like(noise, dtype='float')

    for p in poles:
        for pp in [p, center - (p-center)]:
            dist = ((v-pp)**2).sum(axis=-1)
            filter += np.exp(-dist / (2*sig**2))

    filter[filter > 1] = 1
    return noise*(1-filter)

def plot_noise_pole(diff):
    center = np.array(diff.shape)/2
    poles = np.array([
        [190,277],
        [227,253],
        [233, 256],
        [255, 511],
        [255, 271],
        #[255, 240],
        #[198, 249],
        #[247, 253]
    ])

    q = np.fft.fftshift(np.fft.fftn(diff))
    t = pole_removal(q, poles, sig=4)
    r = np.real(np.fft.ifftn(np.fft.fftshift(t)))

    fig = pl.figure(figsize=(20,10))
    gs = ImageGrid(fig, rect=[0.05, 0.05, 0.45, 0.90], nrows_ncols=(2,2), axes_pad=0.05)
    
    images = [[diff, q], [r, t]]
    maxs, mins = [], []
    for i, im in enumerate(images):
        a,b = im[0], np.abs(im[1])**0.2
        maxs.append([a.max(), b.max()])
        mins.append([a.min(), b.min()])

    maxs, mins = np.array(maxs), np.array(mins)
    maxs = maxs.max(axis=0)
    mins = mins.min(axis=0)

    labels = [['A', 'B'], ['C', 'D']]
    for i, (im, lb) in enumerate(zip(images, labels)):
        ax = [gs[2*i+0], gs[2*i+1]]

        ax[0].imshow(im[0], vmin=mins[0], vmax=maxs[0], cmap=pl.cm.bone)
        ax[1].imshow(np.abs(im[1])**0.2, vmin=mins[1], vmax=maxs[1], cmap=pl.cm.bone)
        lbl(ax[0], lb[0])
        lbl(ax[1], lb[1])

        for a in ax:
            a.grid(False, which='both', axis='both')
            a.set_xticks([])
            a.set_yticks([])

        if i == 0:
            ax[0].set_title("Real-space")
            ax[1].set_title("k-space")
            ax[0].set_ylabel("Raw difference")

        if i == 1:
            ax[0].set_ylabel("Poles removed")
            for p in poles:
                for pp in [p, 2*center - p]:
                    a.plot(pp[1], pp[0], 'wo')

    ax = fig.add_axes([0.57, 0.15, 0.40, 0.7])
    sig = diff.std()

    y,x = np.histogram(diff, bins=np.linspace(-5*sig, 5*sig, 300), normed=True)
    x = (x[1:] + x[:-1])/2
    ax.plot(x, y, '-', lw=2, alpha=0.6, label='Raw noise')

    y,x = np.histogram(r, bins=np.linspace(-5*sig, 5*sig, 300), normed=True)
    x = (x[1:] + x[:-1])/2
    ax.plot(x, y, '-', lw=2, alpha=0.6, label='Poles removed')
    ax.plot(x, 1/np.sqrt(2*np.pi*sig**2) * np.exp(-x**2/(2*sig**2)), 'k--', lw=1.5, label='Gaussian fit')

    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Probability")
    ax.legend(loc='best', prop={'size':18})
    ax.grid(False, which='minor', axis='y')
    lbl(ax, 'E')

    ax.semilogy()

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
