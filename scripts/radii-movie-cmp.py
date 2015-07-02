import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import runner, initializers
import pickle

movie = initializers.load_tiffs("/media/scratch/bamf/tmp/hyperfine*.tif")

for i, (f, frame) in enumerate(movie):
    state, ll = runner.feature(rawimage=frame, sweeps=20, samples=10,
            prad=5.3, psize=5, pad=16, imsize=128, imzstart=4, sigma=0.05, zscale=1.06, invert=True,
            threads=1)
    pickle.dump([state, ll], open("%f-featured.pkl" % f, 'w'))
