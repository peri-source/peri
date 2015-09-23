import pickle
import numpy as np
import scipy as sp

from cbamf.comp import ilms
from cbamf.initializers import load_tiff, normalize

import matplotlib.pyplot as pl
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

    y,x = np.histogram(dat, bins=np.linspace(-5*sig, 5*sig, 200), normed=True)
    x = (x[1:] + x[:-1])/2

    pl.plot(x, y, '-', lw=2, alpha=0.6, label='Confocal image')
    pl.plot(x, 1/np.sqrt(2*np.pi*sig**2) * np.exp(-x**2/(2*sig**2)), 'k--', lw=1.5, label='Gaussian fit')
    pl.semilogy()

    ymin = y[y>0].min()
    ymax = y[y>0].max()

    pl.legend(loc='best', bbox_to_anchor=(0.75, 0.25))
    pl.xlim(-5*sig, 5*sig)
    pl.ylim(ymin, 1.35*ymax)
    pl.xlabel("Pixel value")
    pl.ylabel("Probability")
    pl.grid(False, which='minor', axis='y')
