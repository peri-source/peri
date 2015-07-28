import numpy as np
import scipy as sp
import pylab as pl

from matplotlib.colors import Normalize
from matplotlib import cm

"""
Let's try convolving a 1d function with a varying 1d gaussian
who's sigma goes as \sigma(x) = \sigma_0 * (1 + \\alpha x)
"""

def sig(sigma, alpha, xp):
    return sigma*(1 + alpha*xp)

def gauss(x, xp, sigma, alpha):
    s = sig(sigma, alpha, xp)
    return 1.0/np.sqrt(2*np.pi*s**2) * np.exp(-x**2 / (2*s**2))

def conv(f, sigma=1.0, alpha=1.0):
    o = f*0
    fp = conv2d_varying(f, sigma, alpha)
    z = np.arange(f.shape[0]).astype('float')

    for i in xrange(len(fp)):
        s = sig(sigma, alpha/z[-1], z[i])
        m = (z > z[i]-3*s) & (z < z[i]+3*s)
        g = gauss(z[m] - z[i], z[i], sigma, alpha/z[-1])
        o[i] *= 0

        for gp, fpp in zip(g, fp[m]):
            o[i] += (gp * fpp)
    return o

def conv2d(f, sigma):
    z, y, x = rvecs(f)
    gauss = np.exp(-(x[0]**2 + y[0]**2) / (2*sigma**2))
    q1 = np.fft.fft2(gauss)
    q1 /= q1[0,0]
    q2 = np.fft.fft2(f)
    return np.real(np.fft.ifft2(q1*q2))

def conv2d_varying(f, sigma, alpha):
    z, y, x = rvecs(f)

    zp = np.arange(f.shape[0]).astype('float')
    gauss = 0*x
    for i in xrange(len(zp)):
        s = sig(sigma, alpha/zp[-1], zp[i])
        gauss[i] = np.exp(-(x[0]**2 + y[0]**2) / (2*s**2))
        gauss[i] /= gauss[i,0,0]

    q1 = np.fft.fft2(gauss)
    q2 = np.fft.fft2(f)
    return np.real(np.fft.ifft2(q1*q2))

def rvecs(f):
    kz = 2*f.shape[0]*np.fft.fftfreq(f.shape[0])
    ky = 2*f.shape[1]*np.fft.fftfreq(f.shape[1])
    kx = 2*f.shape[2]*np.fft.fftfreq(f.shape[2])
    return np.meshgrid(kz, ky, kx, indexing='ij')

def kvecs(f):
    kz = np.fft.fftfreq(f.shape[0])
    ky = np.fft.fftfreq(f.shape[1])
    kx = np.fft.fftfreq(f.shape[2])
    return np.meshgrid(kz, ky, kx, indexing='ij')

def gen1():
    N = (32,64,64)
    z = np.linspace(0, 1, N[0])
    y = np.linspace(0, 1, N[1])
    x = np.linspace(0, 1, N[2])
    return np.random.rand(*N), np.array([z,y,x])

def gen2():
    f,x = gen1()
    return f*(f>0.95), x

def disp(f, x, func=conv):
    pl.plot(f, lw=2, c='k')

    vals = np.linspace(0.1, 7, 20)
    colors = cm.copper(Normalize()(vals))

    for v,c in zip(vals, colors):
        o = func(f, x, 1.0, v)
        pl.plot(o, lw=1, c=c)

def test1():
    f,x = gen1()
    disp(f,x)

def test2():
    f,x = gen2()
    disp(f,x)

