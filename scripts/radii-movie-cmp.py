import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import run, initializers
import pickle

movie = initializers.load_tiff_iter("/media/scratch/bamf/averaged_dz015_N200_1.tif", 70)

for i, frame in enumerate(movie):
    state, ll = run.feature(rawimage=frame, sweeps=20, samples=10,
            prad=7.3, psize=9, pad=16, imsize=128, imzstart=12, sigma=0.05, invert=True)
    pickle.dump([state, ll], open("/media/scratch/bamf/averaged_dz015_N200_1.tif-%i.pkl" % i, 'w'))
