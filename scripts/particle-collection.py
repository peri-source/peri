import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl

from cbamf import runner
from cbamf.test import init
from cbamf.viz import plots
import pickle
import time

s = init.create_state_random_packing(imsize=92, radius=8.0, sigma=0.05, seed=10)
strue = s.state.copy()

#raise IOError
h, ll = runner.do_samples(s, sweeps=30, burn=10)
plots.sample_compare(s.N, h, strue)
plots.summary_plot(s, h, truestate=strue)
