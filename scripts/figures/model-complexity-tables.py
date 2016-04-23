import copy
import numpy as np
import scipy as sp
import pylab as pl

from cbamf import util, runner
from cbamf.test import nbody
from cbamf.comp import ilms, objs, psfs, exactpsf
from cbamf.opt import optimize as opt

def create_image(N=128, size=64, radius=6.0, pad=16):
    blank = np.zeros((size,)*3)

    pos, rad, tile = nbody.initialize_particles(
        N, radius=radius, tile=util.Tile(blank.shape), polydispersity=0.0
    )
    sim = nbody.BrownianHardSphereSimulation(pos, rad, tile)
    sim.relax(2000)
    sim.step(5000)
    sim.relax(2000)

    slab_zpos = -radius
    s = runner.create_state(
        blank, pos, rad, slab=slab_zpos, sigma=0.001,
        stateargs={'pad': pad, 'offset': 0.18},
        psftype='cheb-linescan', psfargs={'zslab': slab_zpos, 'cheb_degree': 6, 'cheb_evals': 8},
        ilmtype='barnesleg2p1dx', ilmargs={'order': (1,1,3), 'npts': (30,10,5)}
    )
    s.ilm.randomize_parameters(ptp=0.4, vmax=1.0, fourier=False)
    s.reset()
    s.model_to_true_image()
    return s

def optimize(s):
    args = dict(eig_update=True, update_J_frequency=2, partial_update_frequency=1, max_iter=3)
    blocks = s.b_ilm | s.b_psf | s.b_zscale

    lm0 = opt.LMGlobals(s, blocks, **args)
    lm1 = opt.LMParticles(s, particles=np.arange(s.N), **args)

    for i in xrange(3):
        lm0.do_run_2()
        lm0.reset(0.5)

        lm1.do_run_2()
        lm1.reset(0.5)

def table(s, datas, names, vary_func):
    p0 = s.obj.pos.copy()
    r0 = s.obj.rad.copy()

    slicer = np.s_[s.image[s.inner].shape[0]/2]
    model_image = s.image[s.inner][slicer].copy()

    results = {}
    results['reference'] = (0, model_image, p0, r0)

    for i, (name, data) in enumerate(zip(names, datas)):
        print i, name, data

        vary_func(s, data)
        s.obj.pos = p0.copy()
        s.obj.rad = r0.copy()
        s.reset()

        optimize(s)

        results[name] = (
            i, s.get_difference_image()[slicer].copy(),
            s.obj.pos.copy(),
            s.obj.rad.copy()
        )

    return results

def table_platonic():
    np.random.seed(10)
    s = create_image()

    platonics = [
        ('lerp', 0.05),
        ('lerp', 0.5),
        ('logistic',),
        ('exact-gaussian-fast',)
    ]
    names = [
        'Boolean cut',
        'Linear interpolation',
        'Logistic function',
        'Exact Gaussian convolution'
    ]

    def vary_func(s, data):
        s.obj.exact_volume = False
        s.obj.volume_error = 100.
        s.obj.set_draw_method(*data)

    return table(s, platonics, names, vary_func)

def table_ilms():
    np.random.seed(11)
    s = create_image()

    lilms = [
        ilms.LegendrePoly2P1D(shape=s.ilm.shape, order=(1,1,1)),
        ilms.LegendrePoly2P1D(shape=s.ilm.shape, order=(3,3,3)),
        ilms.BarnesStreakLegPoly2P1DX3(shape=s.ilm.shape, order=(1,1,1), npts=(10,5)),
        ilms.BarnesStreakLegPoly2P1DX3(shape=s.ilm.shape, order=(1,1,2), npts=(30,10)),
        ilms.BarnesStreakLegPoly2P1DX3(shape=s.ilm.shape, order=s.ilm.order, npts=(30,10,5)),
    ]
    names = [
        'Legendre 2+1D (0,0,0)',
        'Legendre 2+1D (2,2,2)',
        'Barnes streaks (10, 5), 0th order z',
        'Barnes streaks (30, 10), 1st order z',
        'Barnes streaks (30, 10, 5), 2nd order z',
    ]

    def vary_func(s, data):
        s.set_ilm(data)

    return table(s, lilms, names, vary_func)

def table_psfs():
    np.random.seed(12)
    s = create_image()

    lpsfs = [
        psfs.IdentityPSF(shape=s.psf.shape, params=np.array([0.0])),
        psfs.AnisotropicGaussian(shape=s.psf.shape, params=(2.0, 1.0, 3.0)),
        psfs.Gaussian4DLegPoly(shape=s.psf.shape, order=(3,3,3)),
        exactpsf.ChebyshevLineScanConfocalPSF(shape=s.psf.shape, zrange=(0, s.psf.shape[0]), cheb_degree=3, cheb_evals=6),
        exactpsf.ChebyshevLineScanConfocalPSF(shape=s.psf.shape, zrange=(0, s.psf.shape[0]), cheb_degree=6, cheb_evals=8),
    ]
    names = [
        'Identity',
        'Gaussian(x,y)',
        'Gaussian(x,y,z,z\')',
        'Chebyshev line scan (3,6)',
        'Chebyshev line scan (6,8)',
    ]

    def vary_func(s, data):
        s.set_psf(data)

    return table(s, lpsfs, names, vary_func)

def gogogo():
    r0 = table_platonic()
    r1 = table_ilms()
    r2 = table_psfs()
    return r0, r1, r2

def scores(results):
    tmp = copy.copy(results)

    scores = []
    for result in tmp:
        ref = tmp.pop('reference')

        errors = {}
        for k,v in tmp.iteritems():
            errors[k] = (
                np.sqrt(((ref[2] - v[2])**2).sum(axis=-1)).mean(),
                np.sqrt(((ref[2] - v[2])**2).sum(axis=-1)).std(),
                np.sqrt((ref[3] - v[3])**2).mean(),
                np.sqrt((ref[3] - v[3])**2).std(),
            )
        scores.append(errors)
    return scores

def make_plots(results):
    pass

