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

def conv(f, x, sigma, alpha):
    o = f*0
    for i in xrange(len(f)):
        o[i] = (gauss(x - x[i], x[i], sigma, alpha) * f).sum()
    return o

def conv2(f, x, sigma, alpha):
    o = f*0
    for i in xrange(len(f)):
        s = sig(sigma, alpha, x[i])
        m = (x > x[i]-3*s) & (x < x[i]+3*s)
        o[i] = (gauss(x[m] - x[i], x[i], sigma, alpha) * f[m]).sum()
    return o

def gen1():
    N = 200
    return np.random.rand(N), np.linspace(0, N,N)

def gen2():
    N = 200
    f,x = np.random.rand(N), np.linspace(0, N,N)
    return f*(f>0.95), x

def disp(f, x, func=conv):
    pl.plot(f, lw=2, c='k')

    vals = np.linspace(0.1, 7, 20)
    colors = cm.copper(Normalize()(vals))

    for v,c in zip(vals, colors):
        o = func(f, x, 1.0, v/len(x))
        pl.plot(o, lw=1, c=c)

def test1():
    f,x = gen1()
    disp(f,x)

def test2():
    f,x = gen2()
    disp(f,x)

