import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
from colloids.cu import fields
from colloids.cu import nbody

NN=32
PPL = 10
N = 1 << 3
radius = 0.5
phi = 0.63

nbody.initializeDevice(0)
nbody.setSeed(10)

def getpos(sim):
    pos = np.zeros(3*N, dtype='float32')
    nbody.simGetX(sim, pos)
    pos = pos.reshape(-1, 3)
    return pos[:,:2]

def getrad(sim):
    rad = np.zeros(N, dtype='float32')
    nbody.simGetRadii(sim, rad)
    return rad

def generate_configuration():
    l = np.sqrt(N*np.pi / phi) * radius
    L = np.array([l,l,1.01], dtype='float32')
    pbc = np.array([1,1,1], dtype='int32')

    rad = np.zeros(N, dtype='float32')
    sim = nbody.createSystem(N, 3)
    nn = nbody.createNBL(N, 2*(radius*(1.01)), 16, L, pbc)
    nbody.init_random_2d(sim, L, radius)

    nbody.simGetRadii(sim, rad)
    rad = rad - 0.05*np.random.rand(N)
    nbody.simSetRadii(sim, rad.astype('float32'))
    nbody.setParametersRelaxation(sim)
    nbody.doSteps(sim, nn, 4000)
    nbody.setParametersHardSphere(sim)

    return sim, nn, getpos(sim), getrad(sim), L[:2]

def genimage(particles, L, donoise=False):
    ccd = np.zeros((L*PPL+1).astype('int'))

    k = np.fft.fftfreq(ccd.shape[0])
    kx = k[:,None]
    ky = k[None,:]
    kv = np.array(np.broadcast_arrays(kx, ky))
    r = np.sqrt(kx**2 + ky**2)
    cut = 0.5*PPL/ccd.shape[0]

    shape_func = 1 - 1/(1+np.exp(-(r-cut)/(cut/10)))

    #r = np.sqrt(kx**2 + ky**2) < 0.5*PPL/ccd.shape[0]
    r2 = np.sqrt(kx**2 + ky**2) < 0.29

    ir = np.fft.fftn(shape_func)
    iccd = ir*np.exp(-2*np.pi*PPL*1.j*kv.T.dot(particles.T).T).sum(axis=0)
    ccd = np.real(np.fft.ifftn(iccd.conj()))

    iccd = np.fft.fftn(ccd)
    ccd = np.real(np.fft.ifftn(r2*iccd.conj()))

    if donoise:
        ccd += np.random.normal(0, sigma, ccd.shape)
    return ccd


sim, nn, pos, rad, LL = generate_configuration()

t = np.zeros((NN,NN), dtype='float32').flatten()
a = fields.createField(np.array([NN,NN], dtype='int32'))
fields.fieldSet(a, t)
fields.setupFFT(a)
fields.process_image(a, sim)

#fields.setx(a)
fields.fieldGet(a, t)
t = t.reshape(NN,NN)
#c = t[:]
#print t

#t = np.zeros((NN,NN), dtype='float32').flatten()
#fields.derivative_x(a)
#fields.fieldGet(a, t)
#t = t.reshape(NN,NN)
#print t

#kx = np.fft.fftfreq(NN)
#ft = np.fft.fftn(c)
#ans = np.fft.ifftn(ft * 1.j*kx)
pl.imshow(t, cmap=mpl.cm.bone, interpolation='nearest')
pl.xticks([])
pl.yticks([])
