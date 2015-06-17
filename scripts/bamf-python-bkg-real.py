import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf.cu import nbl, fields
from cbamf import observers, samplers, models, engines, initializers, states, run

GS = 0.02
RADIUS = 12.0
PSF = (1.2, 4)
ORDER = (3,3,2)
PAD = 22

sweeps = 20
samples = 10
burn = sweeps - samples

#itrue = initializers.normalize(initializers.load_tiff("/media/scratch/bamf/brian-frozen.tif", do3d=True)[12:,:128,:128], True)
#itrue = initializers.normalize(initializers.load_tiff("/media/scratch/bamf/neil-large-clean.tif", do3d=True)[12:,:128,:128], False)
raw = initializers.load_tiff("/media/scratch/bamf/neil-large-clean.tif", do3d=True)
itrue = initializers.normalize(raw[12:,:128,:128], False)
xstart, proc = initializers.local_max_featuring(itrue, 9)
itrue = initializers.normalize(itrue, True)

rstart = 7*np.ones(xstart.shape[0])
pstart = np.array(PSF)
bstart = np.zeros(np.prod(ORDER))
astart = np.ones(1)*-0.2
zstart = np.ones(1)*1.064
GN = rstart.shape[0]
bstart[0] = 1.2

itrue = np.pad(itrue, PAD+2, mode='constant', constant_values=-10)
xstart += PAD+2

strue = np.hstack([xstart.flatten(), rstart, pstart, bstart, astart, zstart])
s = states.ConfocalImagePython(GN, itrue, pad=PAD, order=ORDER, state=strue,
        sigma=GS, psftype=states.PSF_ANISOTROPIC_GAUSSIAN, threads=4)

run.renorm(s)

from scipy.optimize import minimize

def build_bounds(state):
    bounds = []

    bound_dict = {
        'pos': (1,512),
        'rad': (0, 20),
        'typ': (0,1),
        'psf': (0, 10),
        'bkg': (-100, 100),
        'amp': (-3, 3),
        'zscale': (0.5, 1.5)
    }

    for i,p in enumerate(state.param_order):
        bounds.extend([bound_dict[p]]*state.param_lengths[i])
    return np.array(bounds)

def loglikelihood(vec, state):
    state.set_state(vec)
    state.set_current_particle()
    state.create_final_image()
    return state.loglikelihood()

def gradloglikelihood(vec, state):
    state.set_state(vec)
    state.set_current_particle()
    return state.gradloglikelihood()

def gradient_descent(state, method='L-BFGS-B'):
    bounds = build_bounds(state)
    minimize(loglikelihood, state.state, args=(state,),
            method='CG', jac=gradloglikelihood, bounds=bounds)

raise IOError
if True:
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        run.sample_particles(s)
        run.sample_block(s, 'psf', explode=False)
        run.sample_block(s, 'bkg', explode=False)
        run.sample_block(s, 'amp', explode=True)
        #run.sample_block(s, 'zscale', explode=True)

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
