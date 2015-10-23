import numpy as np

from cbamf.comp.psfs import FromArray
N = 17

# get the proper r-vector from fftshift
r = N*np.fft.fftfreq(N)
x,y,z = np.meshgrid(*(r,)*3, indexing='ij')
rr = np.sqrt(x*x + y*y + z*z)
f = (rr==0.).astype('float')

# create the centered PSF using FromArray
ff = np.fft.fftshift(f)
psf = np.array([ff for i in xrange(N)])
a = FromArray(psf, shape=(N,)*3)

# create a blank array to test shift on
# place the marker in the center, one pixel
o = np.zeros((N,)*3)
o[N/2, N/2, N/2] = 1.

# get the convolved image
t = a.execute(o)

# did it shift position?
print np.unravel_index(o.argmax(), o.shape), o.max()
print np.unravel_index(t.argmax(), t.shape), t.max()
