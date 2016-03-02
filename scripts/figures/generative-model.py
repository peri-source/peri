import numpy as np
import scipy as sp
from cbamf import runner, util
from cbamf.viz.plots import generative_model
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
    s,h,l = pickle.load(open('/media/scratch/bamf/crystal-fcc/crystal_fcc.tif_t001.tif-fit-gaussian-4d.pkl'))
    x,y,z,r = np.load('/media/scratch/bamf/crystal-fcc/crystal_fcc.tif_t001.tif-fit-gaussian-4d-sample-xyzr.npy').T
    x -= s.pad
    y -= s.pad
    return s,x,y,z,r

def dorun():
    generative_model(*load())
