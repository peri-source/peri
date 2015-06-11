import pylab as pl
import numpy as np
from scipy.special import j1

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
            ax.set_ylabel(r"$\mathcal{F}(\Pi)$", fontsize=16)

        ax = pl.subplot(gs[1, i])
        ax.imshow(r, vmin=0, vmax=1, cmap=pl.cm.bone,origin='lower', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(r"$\Pi$", fontsize=16)
    pl.show()
        
def show_imag(q):
    pl.figure()
    pl.imshow(np.real(q*q.conj())**0.1, cmap=pl.cm.bone)

def show_real(q):
    pl.figure()
    pl.imshow(q, cmap=pl.cm.bone, vmin=0, vmax=1)

N = 256
M = N/2
r0 = 24
p0 = np.array([M,M])

kx = 2*np.pi*np.fft.fftfreq(N)[:,None]
ky = 2*np.pi*np.fft.fftfreq(N)[None,:]
kv = np.array(np.broadcast_arrays(kx,ky)).T
k = np.sqrt(kx**2 + ky**2)

## =========================================================
x,y = np.mgrid[-M:M:N*1.j, -M:M:N*1.j] 
r = np.sqrt(x**2 + y**2)
sph1 = (r < r0).astype('float')
q1 = np.fft.fftn(sph1)

## =========================================================
x,y = np.mgrid[-M:M:N*1.j, -M:M:N*1.j] 
r = np.sqrt(x**2 + y**2)
sph2 = 1.0 / (1.0 + np.exp(2*(r - r0)))
q2 = np.fft.fftn(sph2)

## =========================================================
q3 = disc2(r0*k+1e-9, r0)*np.exp(-1.j*(kv*p0).sum(axis=-1)).T * np.exp(-k**2/(2*(np.pi/2)**2))
q3[:,q3.shape[0]/2] = 0
q3[q3.shape[0]/2,:] = 0
sph3 = np.real(np.fft.ifftn(q3))

print sph1.max(), sph2.max(), sph3.max()
print sph1.min(), sph2.min(), sph3.min()
qs = [q1, q2, q3]
rs = [sph1, sph2, sph3]
titles = ["Boolean cut", "Real space sigmoid", "Fourier space circle"]

show_all(qs, rs, titles)

if False:
    show_imag(q1)
    show_imag(q2)
    show_imag(q3)
    show_real(sph1)
    show_real(sph2)
    show_real(sph3)
pl.show()
