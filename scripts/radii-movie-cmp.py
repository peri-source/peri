import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import run

state, ll = run.feature("/media/scratch/bamf/neil-large-clean.tif", imsize=128)

