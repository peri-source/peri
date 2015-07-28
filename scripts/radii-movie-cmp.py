import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import runner, initializers
import pickle

imsize = 64
movie = initializers.load_tiffs("/media/scratch/bamf/tmp/hyperfine*.tif")

for i, (f, frame) in enumerate(movie):
    print f
    s = runner.raw_to_state(rawimage=frame, rad=5.3, frad=5, imsize=imsize,
            imzstart=4, sigma=0.05, zscale=1.06, invert=True, threads=4,
            pad_for_extra=True)
    s = runner.feature_addsubtract(s, rad=5.3)
    h, l = runner.do_samples(s, 30, 10, stepout=0.1)

    pickle.dump([s, h, l], open("%s-%i-featured.pkl" % (f, imsize), 'w'))
