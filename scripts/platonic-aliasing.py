import sys
import pylab as pl
import numpy as np

import scipy.ndimage as nd
from scipy.special import j1, erf
from scipy.optimize import leastsq

from peri.viz import util

discs = [0]*3
discs[0] = lambda k,R: 2*R*np.sin(k)/k
discs[1] = lambda k,R: 2*np.pi*R**2 * j1(k)/k
discs[2] = lambda k,R: 4*np.pi*R**3 * (np.sin(k)/k - np.cos(k))/k**2

methods = [0]*7
methods[0] = lambda r, r0, alpha: 1.0 / (1.0 + np.exp(alpha*(r - r0)))
methods[1] = lambda r, r0, alpha: (np.arctan(alpha*(r0-r))/(np.pi/2) + 1)/2
methods[2] = lambda r, r0, alpha: (alpha*(r0-r) / np.sqrt(1 + (alpha*(r-r0))**2) + 1)/2
methods[3] = lambda r, r0, alpha: (erf(alpha*(r0-r))+1)/2
methods[4] = lambda r, r0, alpha: 1-np.clip((r-(r0-alpha)) / (2*alpha), 0, 1)
methods[5] = lambda r, r0, alpha: 1-np.clip((r-r0+alpha)**2/(2*alpha**2)*(r0 > r)*(r>r0-alpha) + 1*(r>r0)-(r0+alpha-r)**2/(2*alpha**2)*(r0<r)*(r<r0+alpha), 0,1)
methods[6] = lambda r, r0, alpha: alpha*methods[4](r,r0,0.453) + (1-alpha)*methods[5](r,r0,0.6618)

mlabels = ['logistic', 'arctan', 'poly', 'erf', 'linear', 'triangle', 'combo']

def score(param, r0, fsphere, method, sigma, dx=None):
    alpha = param[0]
    N = fsphere.shape[0]

    sph = rspace_sphere(N, r0, alpha=alpha, method=method, sigma=sigma, dx=dx)
    return (sph - fsphere).ravel()

def fit(N=32, radius=5.0, fourier_space=False, method=0, sigma=2.0, samples=20):

    params, errors = [], []
    for i in xrange(samples):
        print i,
        sys.stdout.flush()

        dx = np.random.rand(3)
        if fourier_space:
            fsphere = kspace_sphere(N, radius, sigma=sigma, dx=dx)
        else:
            fsphere = rspace_cut(N, radius, sigma=sigma, dx=dx)

        out, call = leastsq(score, x0=np.array([2]), xtol=1e-14, ftol=1e-14,
                            args=(radius, fsphere, method, sigma, dx))
        
        sc = score(out, radius, fsphere, method, sigma, dx=dx)
        params.append(out[0])
        errors.append(((sc**2).mean()))

    print ''
    return params, errors

def psf(field, sigma=2.0):
    return nd.gaussian_filter(field, sigma)

def rspace_cut(N, r0, factor=8, sigma=0.0):
    f = factor
    N = factor * N

    l = np.linspace(-N/2,N/2,N,endpoint=False)
    x,y,z = np.meshgrid(*(l,)*3, indexing='ij')

    sh = -(f-1.0)/2
    x += sh
    y += sh
    z += sh
    r = np.sqrt(x**2 + y**2 + z**2)

    i = np.linspace(0, N/f, N, endpoint=False).astype('int')
    z,y,x = np.meshgrid(*(i,i,i), indexing='ij')
    ind = x + N*y + N*N*z

    # finally, c-g'ed image
    im = psf((r < f*r0).astype('float'), f*sigma)
    cg = nd.mean(im, labels=ind, index=np.unique(ind)).reshape(N/f, N/f, N/f)
    return cg

def rspace_sphere(N, r0, alpha=6.5, sigma=0.0, method=0, dx=None):
    l = np.linspace(-N/2,N/2,N,endpoint=False).astype('float')
    x,y,z = np.meshgrid(*(l,)*3, indexing='ij')

    if dx is not None:
        x += dx[0]
        y += dx[1]
        z += dx[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    sph = methods[method](r, r0, alpha)
    return psf(sph, sigma)

def kspace_sphere(N, r0, sigma=0.0, dx=None):
    M = N/2
    p0 = np.array([M,M,M]).astype('float')

    if dx is not None:
        p0 -= dx

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array(np.broadcast_arrays(kx,ky,kz)).T
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    q = discs[2](r0*k+1e-5, r0)*np.exp(-1.j*(kv*p0).sum(axis=-1)).T
    sph = np.real(np.fft.ifftn(q))
    return psf(sph, sigma)

def gogogo():
    fits, errs = [], []
    rads = np.linspace(2.0, 10.0, 20)

    for i,method in enumerate(mlabels):
        print method
        do, de = [], []
        for rad in rads:
            print 'rad', rad
            o,e = fit(N=32, radius=rad, fourier_space=True, method=i,
                      sigma=0.0, samples=30)
            do.append(o)
            de.append(e)
        fits.append(do)
        errs.append(de)

    fits, errs = [np.array(i) for i in [fits, errs]]

    pl.figure()
    pl.title("Fit value")
    for i,q in enumerate(fits):
        pl.plot(rads, q.mean(axis=-1), 'o-', label=mlabels[i])
    pl.semilogy()
    pl.legend(loc='best', numpoints=1)

    pl.figure()
    pl.title("MSE")
    for i,q in enumerate(errs):
        pl.plot(rads, q.mean(axis=-1), 'o-', label=mlabels[i])
    pl.semilogy()
    pl.legend(loc='best', numpoints=1)

    pl.figure()
    pl.title("MSE normed")
    for i,q in enumerate(errs):
        pl.plot(rads, q.mean(axis=-1)/errs[-1].mean(axis=-1), 'o-', label=mlabels[i])
    pl.semilogy()
    pl.legend(loc='best', numpoints=1)

    return fits, errs
