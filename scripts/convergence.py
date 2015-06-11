# coding: utf-8
import os
HASDISPLAY = os.environ.haskey("DISPLAY")

import scipy.ndimage as nd
import numpy as np
import time
import matplotlib
if not HASDISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as pl

from colloids.cu import nbody
from colloids.salsa import process

DOPLOT = 0
N = 1 << 14
radius = 1.0
phi = 0.59

l = np.power(1/phi*4*np.pi/3*radius**3*N, 1./3) - radius
L = np.array([l,l,l], dtype='float32')
pbc = np.array([1,1,1], dtype='int32')

resample = 5
shape = [3,3] + [ resample*(2*i+1) for i in [15, 15, 3]]
size = int(np.prod(shape))

def calculate_long_time_fx(steps=int(1e3)):
    nbody.setSeed(10)
    nbody.initializeDevice(0)

    field = np.zeros(shape, dtype='float32').flatten()
    field_stress = nbody.createDeviceArray(size)
    gstress = nbody.createDeviceArray(N*9)
    
    shell = 0.001
    sim = nbody.createSystem(N, 3)
    nn = nbody.createNBL(N, 2*(radius*(1+shell)), 16, L, pbc)
    nbody.init_crystal(sim, L, radius, 1)
    nbody.setParametersHardSphere(sim)
    
    start = time.time()
    for i in xrange(steps):
        if i % 100 == 0:
            print i
        nbody.doSteps(sim, nn, 300)
        nbody.calculateStress(sim, nn, gstress)
        nbody.binRotatedField(sim, field_stress, resample, 15, 15, 3, 8736, gstress)
    end = time.time()
    print end - start
    
    nbody.copyFromDevice(field_stress, field)
    field = field.reshape(shape)
    
    # plot the stresses
    sl = np.s_[37:-42,37:-42,17] 
    field2d = [[0,0,0],[0,0,0],[0,0,0]]
    for i in xrange(0, 3):
        for j in xrange(0, 3):
            field2d[i][j] = nd.gaussian_filter(field[i,j]*13.8e-6/steps*1e3, 7.6)[sl]
    field2d = np.array(field2d)

    nbody.freeDeviceArray(field_stress)
    nbody.freeDeviceArray(gstress)
    nbody.freeNBL(nn)
    nbody.freeSystem(sim)

    return field2d

def calculate_relative_error(fx, steps=int(1e3), skip=1):
    nbody.setSeed(10)
    nbody.initializeDevice(0)

    field = np.zeros(shape, dtype='float32').flatten()
    field_stress = nbody.createDeviceArray(size)
    gstress = nbody.createDeviceArray(N*9)
    
    shell = 0.001
    sim = nbody.createSystem(N, 3)
    nn = nbody.createNBL(N, 2*(radius*(1+shell)), 16, L, pbc)
    nbody.init_crystal(sim, L, radius, 1)
    nbody.setParametersHardSphere(sim)
   
    ns, err = [], []
    start = time.time()
    for i in xrange(steps):
        if i % 100 == 0:
            print i
        nbody.doSteps(sim, nn, 300)
        nbody.calculateFabric(sim, nn, gstress, nbody.FABRIC_STRICT_SHELL)
        nbody.binRotatedField(sim, field_stress, resample, 15, 15, 3, 8736, gstress)

        if i % skip == 0:
            sl = np.s_[37:-42,37:-42,17] 
            pref = 13.8e-6/(i+1)*1e3 * 293 * (radius / (2*shell))
            field2d = [[0,0,0],[0,0,0],[0,0,0]]
            nbody.copyFromDevice(field_stress, field)
            tfield = field.reshape(shape)[:]
            for k in xrange(0, 3):
                for j in xrange(0, 3):
                    field2d[k][j] = nd.gaussian_filter(tfield[k,j]*pref, 7.6)[sl]
            field2d = np.array(field2d)

            def rms(a, b):
                return np.sqrt(((a-b)**2).mean()) / np.sqrt((b**2).mean())

            terr = []
            pressure_fx = 0*field2d[0,0]
            pressure_salsa = 0*field2d[0,0]
            for m in xrange(3):
                pressure_fx += fx[m,m]/3
                pressure_salsa += field2d[m,m]/3

            terr.append(rms(pressure_salsa, pressure_fx))
            for m in xrange(3):
                terr.append(rms(field2d[m,m]-pressure_salsa, fx[m,m]-pressure_fx))

                for n in xrange(m+1,3):
                    terr.append(rms(field2d[m,n], fx[m,n]))
            err.append(terr)
            ns.append(i+1)
    end = time.time()
    print end - start
    
    nbody.freeDeviceArray(field_stress)
    nbody.freeDeviceArray(gstress)
    nbody.freeNBL(nn)
    nbody.freeSystem(sim)

    return np.array(ns), np.array(err)

def plot_convergence(steps=int(1e4), skip=1):
    from rcparams import setPlotOptions
    setPlotOptions(pl)
    fx = calculate_long_time_fx(steps)
    n, err = calculate_relative_error(fx, steps, skip)
    #return n, err
    pl.figure()
    for i in xrange(7):
        pl.loglog(n, err[:,i])
    pl.xlabel(r"$N$")
    pl.ylabel(r"$\sqrt{\langle\left(\Psi^{\rm{Fx}}_{ij} - \Psi^{\rm{SALSA}}_{ij}\right)^2\rangle} / \sqrt{\langle\left(\Psi^{\rm{Fx}}_{ij}\right)^2\rangle}$")
    return n, err
