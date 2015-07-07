import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import runner, initializers
import pickle

imsize = 128
movie = initializers.load_tiffs("/media/scratch/bamf/tmp/hyperfine*.tif")

for i, (f, frame) in enumerate(movie):
    print f
    state, ll = runner.feature(rawimage=frame, sweeps=20, samples=10,
            prad=5.3, psize=5, pad=16, imsize=imsize, imzstart=4, sigma=0.05, zscale=1.06,
            invert=True, threads=4, addsubtract=True)
    pickle.dump([state, ll], open("%s-%i-featured.pkl" % (imsize, f), 'w'))
