import pylab as pl
import numpy as np
from scipy.special import j1, erf
import scipy.ndimage as nd

def disc1(k, R):
    return 2*R*np.sin(k)/k

def disc2(k, R):
    return 2*np.pi*R**2 * j1(k) / k

def disc3(k, R):
    return 4*np.pi*R**3 * (np.sin(k)/k - np.cos(k))/k**2

def psf_disc(k, params):
    return (1.0 + np.exp(-params[0]*params[1])) / (1.0 + np.exp(params[0]*(k - params[1])))

def show_all(qs, rs, titles=[]):
    import matplotlib.gridspec as gridspec
    import matplotlib as mpl
    fig = pl.figure(figsize=(len(qs)*5, 2*5), dpi=100)
    gs = gridspec.GridSpec(2, len(qs))
    gs.update(left=0.1, right=0.90, hspace=0.00, wspace=0.0)

    for i, (title, q, r) in enumerate(zip(titles, qs, rs)):
        ax = pl.subplot(gs[0, i])
        ax.set_title(title, fontsize=16)
        ax.imshow(np.real(q*q.conj())**0.1, cmap=pl.cm.bone,origin='lower', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(r"$\mathcal{F}(\Pi)$", fontsize=18)

        ax = pl.subplot(gs[1, i])
        ax.imshow(r, vmin=0, vmax=1, cmap=pl.cm.bone,origin='lower', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(r"$\Pi$", fontsize=18)
    pl.show()

def show_imag(q):
    pl.figure()
    o = np.real(q*q.conj())**0.1
    o[0,0]=0
    pl.imshow(o, cmap=pl.cm.bone)

def show_real(q):
    pl.figure()
    pl.imshow(q, cmap=pl.cm.bone, vmin=0, vmax=1)

def score(param, N, fsphere, r0):
    alpha = param[0]

    x,y = np.mgrid[-M:M:N*1.j, -M:M:N*1.j]
    x -= x[N/2, N/2]
    y -= y[N/2, N/2]
    r = np.sqrt(x**2 + y**2)

    #sph = (np.arctan(3*(r0-r))/(np.pi/2) + 1)/2
    #sph = (np.tanh(2.5*(r0-r))+1)/2
    #sph = ((r0-r) / np.sqrt(1 + (r-r0)**2)+1)/2
    #sph = (erf(alpha*(r0-r))+1)/2
    sph = 1.0 / (1.0 + np.exp(alpha*(r - r0)))
    #return sph
    return (fsphere - sph).ravel()

def fit_platonic(N, fsphere, r0):
    from scipy.optimize import leastsq
    out, call = leastsq(score, x0=np.array([2]), args=(N, fsphere, r0))
    print (score(out, N, fsphere, r0)**2).mean()
    return out[0]

def coarse_grain(field, factor=2):
    return nd.filters.convolve(field, np.ones((factor,factor)))[factor/2::factor, factor/2::factor]

N = 64
M = N/2
r0 = 5
p0 = np.array([M,M])

kx = 2*np.pi*np.fft.fftfreq(N)[:,None]
ky = 2*np.pi*np.fft.fftfreq(N)[None,:]
kv = np.array(np.broadcast_arrays(kx,ky)).T
k = np.sqrt(kx**2 + ky**2)

## =========================================================
x,y = np.mgrid[-M:M:N*1.j, -M:M:N*1.j]
x -= x[N/2, N/2]
y -= y[N/2, N/2]
r = np.sqrt(x**2 + y**2)

sph1 = (r < r0).astype('float')
q1 = np.fft.fftn(sph1)

## =========================================================
f = 10
x,y = np.mgrid[-f*M:f*M:f*N*1.j, -f*M:f*M:f*N*1.j]
x -= x[f*N/2, f*N/2]
y -= y[f*N/2, f*N/2]
r = np.sqrt(x**2 + y**2)

sph2 = coarse_grain((r < f*r0).astype('float'), f) / f**2
q2 = np.fft.fftn(sph2)

## =========================================================
x,y = np.mgrid[-M:M:N*1.j, -M:M:N*1.j]
x -= x[N/2, N/2]
y -= y[N/2, N/2]
r = np.sqrt(x**2 + y**2)

sph3 = 1.0 / (1.0 + np.exp(5.258*(r - r0)))
#sph3 = (np.arctan(3*(r0-r))/(np.pi/2) + 1)/2
#sph3 = (np.tanh(2.5*(r0-r))+1)/2
#sph3 = (r0-r) / np.sqrt(1 + (r-r0)**2)
#sph3 = (erf(2*(r0-r))+1)/2
q3 = np.fft.fftn(sph3)

## =========================================================
q4 = disc2(r0*k+1e-9, r0)*np.exp(-1.j*(kv*p0).sum(axis=-1)).T
sph4 = np.real(np.fft.ifftn(q4))

print sph1.max(), sph2.max(), sph3.max(), sph4.max()
print sph1.min(), sph2.min(), sph3.min(), sph4.min()
qs = [q1, q2, q3, q4]
rs = [sph1, sph2, sph3, sph4]
titles = ["Boolean cut", "Coarse-grained boolean cut", "Real space sigmoid", "Fourier space circle"]

show_all(qs, rs, titles)

if False:
    show_imag(q1)
    show_imag(q2)
    show_imag(q3)
    show_real(sph1)
    show_real(sph2)
    show_real(sph3)
pl.show()
