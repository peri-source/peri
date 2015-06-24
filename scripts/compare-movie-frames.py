import numpy as np
import scipy as sp
import pickle
import pylab as pl

def load(i):
    return pickle.load(open("./averaged_dz015_N200_1.tif-%i.pkl" % i))[0]

N = 9
rdiff = []
pdiff = []

for i in [0]:#xrange(N-1):
    s = load(i)
    n = (s.shape[-1]-5)/4
    print n
    p = s[:,:3*n].reshape(-1, n, 3).mean(axis=0)
    r = s[:,3*n:4*n].reshape(-1, n).mean(axis=0)
    stdr = s[:,3*n:4*n].reshape(-1, n).std(axis=0)
    #pl.figure()
    #pl.hist(stdr, bins=np.logspace(-3, 1, 30), histtype='stepfilled')
    #pl.semilogx()
    for j in xrange(i+1, N):
        s2 = load(j)
        n2 = (s2.shape[-1]-5)/4
        p2 = s2[:,:3*n2].reshape(-1, n2, 3).mean(axis=0)
        r2 = s2[:,3*n2:4*n2].reshape(-1, n2).mean(axis=0)

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

for i in xrange(N):
    s = load(i)
    psf =s[:,-5:-3].mean(axis=0)
    print psf, psf[0]/psf[1], s[:,-1].mean()

