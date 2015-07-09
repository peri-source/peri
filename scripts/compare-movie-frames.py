import numpy as np
import scipy as sp
import pickle
import pylab as pl

def load(i):
    return pickle.load(open("/media/scratch/bamf/tmp/hyperfine_dz015_N50_3.tif_t%03d.tif-64-featured.pkl" % i))[0]
    #return pickle.load(open("./averaged_dz015_N200_1.tif-%i.pkl" % i))[0]

def to_vals(s, full=False):
    n = (s.shape[-1]-23)/5
    t = s[:,4*n:5*n].mean(axis=0) == 1.

    p = s[:,:3*n].reshape(-1, n, 3).mean(axis=0)
    r = s[:,3*n:4*n].reshape(-1, n).mean(axis=0)
    stdr = s[:,3*n:4*n].reshape(-1, n).std(axis=0)

    if full:
        return n, p, r, sdr
    return t.sum(), p[t], r[t], stdr[t]

N = 6
rdiff = []
pdiff = []
rstd = []

for i in xrange(1, N-1):
    s = load(i)
    n, p, r, stdr = to_vals(s)
    rstd.append(stdr)
    #pl.figure()
    #pl.hist(stdr, bins=np.logspace(-3, 1, 30), histtype='stepfilled')
    #pl.semilogx()
    for j in xrange(i+1, N):
        s2 = load(j)
        n2, p2, r2, stdr2 = to_vals(s2)

        print n, n2

        diffs = []
        diffs2 = []
        for k in xrange(n):
            a = ((p[k] - p2)**2).sum(axis=-1).argmin()
            diffs.append(np.abs(r[k] - r2[a]))
            diffs2.append(np.sqrt(((p[k,:2] - p2[a,:2])**2).sum()))

        rdiff.append(diffs)
        pdiff.append(diffs2)

rdiff = np.array(rdiff)
pdiff = np.array(pdiff)

pl.figure()
for i in xrange(len(pdiff)):
    pl.hist(pdiff[i], bins=np.logspace(-3, 1, 30), histtype='stepfilled', alpha=0.4)
pl.semilogx()

pl.figure()
for i in xrange(len(rdiff)):
    pl.hist(rdiff[i], bins=np.logspace(-3, 1, 30), histtype='stepfilled', alpha=0.4)
pl.semilogx()

pl.figure()
for i in xrange(len(rstd)):
    pl.hist(rstd[i], bins=np.logspace(-3, 1, 30), histtype='stepfilled', alpha=0.4)
pl.semilogx()

for i in xrange(1,N):
    s = load(i)
    psf = s[:,-5:-3].mean(axis=0)
    #print psf, psf[0]/psf[1], s[:,-1].mean()
    print s[:,-23:-21].mean(axis=0)
