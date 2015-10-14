import pickle
import numpy as np
import scipy.ndimage as nd

import matplotlib as mpl
from matplotlib import pyplot as pl
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid

def smile_comparison_plot(state0, state1):
    fig = pl.figure(figsize=(10,12))

    ig = ImageGrid(fig, rect=[0.05, 0.55, 0.90, 0.40], nrows_ncols=(1,2), axes_pad=0.2)
    ax0 = fig.add_axes([0.13, 0.05, 0.74, 0.40])
    ax1 = ax0.twinx()

    stringer = lambda o: ', '.join([str(i) for i in o])

    states = [state0, state1]
    orders = [stringer(s.ilm.order) for s in states]
    colors = ['#333333', '#AAAAAA']

    for i,(s,o,color) in enumerate(zip(states, orders, colors)):
        ax = ig[i]

        m = s.obj.typ == 1.
        p = s.obj.pos[m]
        r = s.obj.rad[m]
        z,y,x = p.T

        sl = np.s_[s.pad:-s.pad,s.pad:-s.pad,s.pad:-s.pad]

        mu = r.mean()
        std = 1.0*r.std()
        c = pl.cm.RdBu_r(Normalize(vmin=mu-std, vmax=mu+std)(r))[:,:3]
        diff = -(s.image - s.get_model_image())[sl]

        ax.set_title(o)
        ax.imshow(diff[-5])
        ax.scatter(x-s.pad,y-s.pad,s=60,c=c)
        ax.set_xlim(0,diff.shape[1])
        ax.set_ylim(0,diff.shape[2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

        sect = 255*nd.gaussian_filter(diff.mean(axis=(0,1)), 3, mode='reflect')
        ax0.plot(sect, lw=2, c=color, label='Residual %s' % o)
        ax1.plot(x, r, 'o', mfc=color, mec='black', label='Radii %s' % o)

        ax0.set_xlim(50, diff.shape[1])
        ax1.set_xlim(50, diff.shape[1])
        ax0.set_ylim(sect.mean()-16*sect.std(), sect.mean()+26*sect.std())
        ax1.set_ylim(r.mean()-5*r.std(), r.mean()+2*r.std())
        ax0.grid('off')
        ax1.grid('off')
        ax0.set_xticks([])
        ax1.set_xticks([])
        ax0.set_xlabel("Pixel position")
        ax0.set_ylabel("Residual value")
        ax1.set_ylabel("Particle Radius")
        ax0.legend(bbox_to_anchor=(1.08,1.3), ncol=4)
        ax1.legend(bbox_to_anchor=(1,1.17), ncol=4, numpoints=1)
