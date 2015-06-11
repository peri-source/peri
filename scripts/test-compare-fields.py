# coding: utf-8
import scipy.ndimage as nd
import numpy as np
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl

from colloids.cu import nbody
from colloids.salsa import process

DOPLOT = 0
N = 1 << 14
radius = 1. #304
phi = 0.59

l = np.power(1/phi*4*np.pi/3*radius**3*N, 1./3) - radius
L = np.array([l,l,l], dtype='float32')
pbc = np.array([1,1,1], dtype='int32')

nbody.initializeDevice(0)
nbody.setSeed(10)

resample = 5
shape = [3,3] + [ resample*(2*i+1) for i in [15, 15, 3]]
size = int(np.prod(shape))

field1 = np.zeros(shape, dtype='float32').flatten()
field2 = np.zeros(shape, dtype='float32').flatten()
field_stress = nbody.createDeviceArray(size)
field_fabric = nbody.createDeviceArray(size)
gstress = nbody.createDeviceArray(N*9)

shell = 0.015
sim = nbody.createSystem(N, 3)
nn = nbody.createNBL(N, 2*(radius*(1+shell)), 16, L, pbc)
nbody.init_crystal(sim, L, radius, 1)
nbody.setParametersHardSphere(sim)

start = time.time()

if DOPLOT:
    import cplot
    plt = cplot.plot_init(500, L, radius)

steps = int(1e4)
for i in xrange(steps):
    if i % 100 == 0:
        print i
    nbody.doSteps(sim, nn, 100)
    nbody.calculateStress(sim, nn, gstress)
    nbody.binRotatedField(sim, field_stress, resample, 15, 15, 3, 8736, gstress)
    nbody.calculateFabric(sim, nn, gstress, nbody.FABRIC_STRICT_SHELL)
    nbody.binRotatedField(sim, field_fabric, resample, 15, 15, 3, 8736, gstress)

    if DOPLOT:
        cplot.plot_clear_screen(plt)
        cplot.plot_render_particles(plt, sim)

        if cplot.plot_update_from_keys(plt) == 0:
            break

end = time.time()
print end - start

nbody.copyFromDevice(field_stress, field1)
nbody.copyFromDevice(field_fabric, field2)
field1 = field1.reshape(shape)
field2 = field2.reshape(shape)

# plot the stresses
sl = np.s_[37:-42,37:-42,17] 
pref = 293 * (radius / (2*shell))
field2d1 = [[0,0,0],[0,0,0],[0,0,0]]
field2d2 = [[0,0,0],[0,0,0],[0,0,0]]
for i in xrange(0, 3):
    for j in xrange(0, 3):
        field2d1[i][j] = nd.gaussian_filter(field1[i,j] * 13.8e-6 / steps * 1e3, 7.6)[sl]
        field2d2[i][j] = nd.gaussian_filter(field2[i,j] * 13.8e-6 / steps * 1e3 * pref, 7.6)[sl]
field2d1 = np.array(field2d1)
field2d2 = np.array(field2d2)

nbody.freeDeviceArray(field_stress)
nbody.freeDeviceArray(field_fabric)
nbody.freeDeviceArray(gstress)
nbody.freeNBL(nn)
nbody.destroyDevice()
