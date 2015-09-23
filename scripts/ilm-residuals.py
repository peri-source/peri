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

def do_fit(im, order=(15,5,5)):
    ilm = ilms.LegendrePoly2P1D(im.shape, order=order)
    ilm.params[-order[-1]] = 1
    ilm.from_data(im, maxcalls=30)
    ilm.initialize()
    return ilm

def do_fits():
    diffs = []
    orders = ((3,3,2), (7,5,3), (11,7,5), (17,11,7))

    d = load_background_image()
    for order in orders:
        ilm = do_fit(d, order=order)
        pickle.dump(ilm, open(ilm_file+"-leg2p1d-"+str(order)+".pkl", 'w'))
        
        diffs.append(ilm.get_field() - d)

    return np.array(diffs), orders

def doplot(diffs, orders):
    fig = pl.figure(figsize=(16,6))
    gs = ImageGrid(fig, rect=[0.05, 0.05, 0.90, 0.90], nrows_ncols=(1,4), axes_pad=0.05)

    scale = 0.25*np.abs(diffs).max()
    zslice = diffs[0].shape[0]/2

    for i, (diff, order) in enumerate(zip(diffs, orders)):
        ax = gs[i]
        ax.imshow(diff[zslice], vmin=-scale, vmax=scale, cmap=pl.cm.RdBu)
        ax.set_title(str(order))
        ax.set_xticks([])
        ax.set_yticks([])
