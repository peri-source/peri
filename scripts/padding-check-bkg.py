import numpy as np
import scipy as sp

N = 256
bkg, _ = np.mgrid[0:1:N*1.j, 0:1:N*1.j] * 2*np.pi
bkg = 0.2*np.sin(bkg) + 0.8

def add_sphere(x0, y0, r0, img):
    M = img.shape[0]
    x,y = np.mgrid[0:M:M*1.j, 0:M:M*1.j] 
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    return 1.0 / (1.0 + np.exp(2*(r - r0)))

spheres = np.zeros_like(bkg)
spheres += add_sphere(N/2, N/2, 10, spheres)
spheres += add_sphere(N/2-21, N/2, 10, spheres)
spheres += add_sphere(N/2+21, N/2, 10, spheres)
spheres += add_sphere(90, 12, 10, spheres)

img = bkg * (1 - spheres) 
pad_img = np.pad(img, img.shape[0]/2, mode='constant', constant_values=0)

sig = 10
k = np.fft.fftfreq(pad_img.shape[0])
kx, ky = k[:,None], k[None,:]
k = np.sqrt(kx**2 + ky**2)

kimg = np.fft.fftn(pad_img) * np.exp(-k**2 / 2 * sig**2)
q = np.real(np.fft.ifftn(kimg))
out = q[2*N/4:-2*N/4, 2*N/4:-2*N/4]
