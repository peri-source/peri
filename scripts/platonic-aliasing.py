import pylab as pl
import numpy as np

import scipy.ndimage as nd
from scipy.special import j1, erf
from scipy.optimize import leastsq

def score(param, r0, fsphere):
    alpha = param[0]
    N = fsphere.shape[0]

    sph = rspace_sphere(N, r0, alpha=alpha)
    return (sph - fsphere).ravel()

def fit(N=32, radius=5.0, fourier_space=False):

    if fourier_space:
        fsphere = kspace_sphere(N, radius)
    else:
        fsphere = rspace_cut(N, radius)

    out, call = leastsq(score, x0=np.array([2]), args=(radius, fsphere), xtol=1e-6, ftol=1e-9)
    
    sc = score(out, radius, fsphere)
    print (sc**2).mean(), sc.ptp()
    return out[0]#, fsphere, rspace_sphere(N, radius, 6.8)

def psf(field, sigma=2.0):
    return nd.gaussian_filter(field, sigma)

def rspace_cut(N, r0, factor=8, sigma=2.0):
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

def rspace_sphere(N, r0, alpha=5., sigma=2.0, method=0):
    l = np.linspace(-N/2,N/2,N,endpoint=False)
    x,y,z = np.meshgrid(*(l,)*3, indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)

    methods = [0]*10
    methods[0] = lambda r, r0, alpha: 1.0 / (1.0 + np.exp(alpha*(r - r0)))
    methods[1] = lambda r, r0, alpha: (np.arctan(alpha*(r0-r))/(np.pi/2) + 1)/2
    methods[2] = lambda r, r0, alpha: (np.tanh(alpha*(r0-r))+1)/2
    methods[3] = lambda r, r0, alpha: (r0-r) / np.sqrt(1 + (r-r0)**2)
    methods[4] = lambda r, r0, alpha: (erf(alpha*(r0-r))+1)/2

    sph = methods[method](r, r0, alpha)
    return psf(sph, sigma)

def kspace_sphere(N, r0, sigma=2.0):
    M = N/2
    p0 = np.array([M,M,M])

    f = 2*np.pi*np.fft.fftfreq(N)
    kx,ky,kz = np.meshgrid(*(f,)*3, indexing='ij')
    kv = np.array(np.broadcast_arrays(kx,ky,kz)).T
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    disc1 = lambda k,R: 2*R*np.sin(k)/k
    disc2 = lambda k,R: 2*np.pi*R**2 * j1(k)/k
    disc3 = lambda k,R: 4*np.pi*R**3 * (np.sin(k)/k - np.cos(k))/k**2

    q = disc3(r0*k+1e-5, r0)*np.exp(-1.j*(kv*p0).sum(axis=-1)).T
    sph = np.real(np.fft.ifftn(q))
    return psf(sph, sigma)

