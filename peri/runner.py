import numpy as np
import scipy.ndimage as nd

from peri import const, util
from peri import states, initializers

# poly fit function, because I can
def poly_fit(x, y, order=2, sigma=0.1, N=100, burn=100):
    """
    generate data:
    x = np.linspace(0, 1, 10000)
    y = np.polyval(np.random.randn(4), x) + 0.05*np.random.randn(10000)
    """
    from peri.states import PolyFitState
    s = PolyFitState(x, y, order=order, sigma=sigma)
    h = sample_state(s, s.params, N=burn, doprint=True, procedure='uniform')
    h = sample_state(s, s.params, N=burn, doprint=True, procedure='uniform')

    import pylab as pl
    pl.plot(s.data, 'o')
    pl.plot(s.model, '-')
    return s, h.get_histogram()

#=============================================================================
# Optimization methods like gradient descent
#=============================================================================
def optimize_particle(state, index, method='gn', doradius=True):
    """
    Methods available are
        gn : Gauss-Newton with JTJ (recommended)
        nr : Newton-Rhaphson with hessian

    if doradius, also optimize the radius.
    """
    blocks = state.blocks_particle(index)

    if not doradius:
        blocks = blocks[:-1]

    g = state.gradloglikelihood(blocks=blocks)
    if method == 'gn':
        h = state.jtj(blocks=blocks)
    if method == 'nr':
        h = state.hessloglikelihood(blocks=blocks)
    step = np.linalg.solve(h, g)

    h = np.zeros_like(g)
    for i in xrange(len(g)):
        state.update(blocks[i], state.state[blocks[i]] - step[i])
    return g,h

def optimize_particles(state, *args, **kwargs):
    for i in state.active_particles():
        optimize_particle(state, i, *args, **kwargs)

def modify(state, blocks, vec):
    for bl, val in zip(blocks, vec):
        state.update(bl, np.array([val]))

import time

def residual(vec, state, blocks, relax_particles=True):
    print time.time(), 'res', state.loglikelihood()
    modify(state, blocks, vec)

    for i in xrange(3):
        #sample_particles(state, quiet=True)
        optimize_particles(state)

    return state.residuals().flatten()

def gradient_descent(state, blocks, method='L-BFGS-B'):
    from scipy.optimize import minimize

    t = np.array([state.state[b] for b in blocks])
    return minimize(residual_sq, t, args=(state, blocks),
            method=method)#, jac=gradloglikelihood, hess=hessloglikelihood)

def lm(state, blocks, method='lm'):
    from scipy.optimize import root

    t = np.array(blocks).any(axis=0)
    return root(residual, state.state[t], args=(state, blocks),
            method=method)

def leastsq(state, blocks, dojac=True):
    from scipy.optimize import leastsq

    if dojac:
        jacfunc = jac
    else:
        jacfunc = None

    t = np.array([state.state[b] for b in blocks])
    return leastsq(residual, t, args=(state, blocks), Dfun=jacfunc, col_deriv=True)

def gd(state, N=1, ratio=1e-1):
    state.set_current_particle()
    for i in xrange(N):
        print state.loglikelihood()
        grad = state.gradloglikelihood()
        n = state.state + 1.0/np.abs(grad).max() * ratio * grad
        state.set_state(n)
        print state.loglikelihood()

#=============================================================================
# Initialization methods to go full circle
#=============================================================================
def create_state(image, pos, rad, sigma=0.05, slab=None, pad_extra_particles=False,
        ignoreimage=False, psftype='gauss4d', ilmtype='poly3d',
        psfargs={}, ilmargs={}, objargs={}, stateargs={}):
    """
    Create a state from a blank image, set of pos and radii

    Parameters:
    -----------
    image : ndarray or `peri.util.RawImage`
        raw confocal image with which to compare.

    pos : initial conditions for positions (in raw image coordinates)
    rad : initial conditions for radii array (can be scalar)
    sigma : float, noise level

    slab : float
        z-position of the microscope slide in the image (pixel units)

    pad_for_extra : boolean
        whether to include extra blank particles for sampling over

    ignoreimage : boolean
        whether to refer to `image` just for shape. really use the model
        image as the true raw image

    psftype : ['identity', 'gauss2d', 'gauss3d', 'gauss4d', 'gaussian_pca',
               'linescan', 'cheb-linescan', 'cheb-linescan-fixedss']
        which type of psf to use in the state

    ilmtype : ['poly3d', 'leg3d', 'cheb2p1d', 'poly2p1d', 'leg2p1d',
               'barnesleg2p1d', 'barnesleg2p1dx']
        which type of illumination field

    psfargs : arguments to the psf object
    ilmargs: arguments to the ilm object
    objargs: arguments to the sphere collection object
    stateargs : dictionary of arguments to pass to state
    """
    tpsfs = ['gauss3d', 'gauss4d']
    tilms = ['poly3d', 'leg3d', 'cheb2p1d', 'poly2p1d', 'leg2p1d']

    # first, decide if we got a RawImage or just an image array
    rawimage = None
    if isinstance(image, RawImage):
        rawimage = image
        image = rawimage.get_image()

    # we accept radius as a scalar, so check if we need to expand it
    if not hasattr(rad, '__iter__'):
        rad = rad*np.ones(pos.shape[0])

    # let's create the padded image required of the state
    pad = stateargs.get('pad', const.PAD)
    image, pos, rad = states.prepare_for_state(image, pos, rad, pad=pad)

    # create some default arguments for the image components
    def_obj = {'pos': pos, 'rad': rad, 'shape': image.shape}
    def_psf = {'shape': image.shape}
    def_ilm = {'order': (1,1,1), 'shape': image.shape}

    # create the SphereCollectionRealSpace object
    def_obj.update(objargs)

    nfake = None
    if pad_extra_particles:
        nfake = pos.shape[0]
        pos, rad = pad_fake_particles(pos, rad, nfake)

    def_obj.update({'pad': nfake})
    obj = objs.SphereCollectionRealSpace(**def_obj)

    # setup the ilm based on the choice and arguments
    if ilmtype == 'poly3d':
        def_ilm.update(ilmargs)
        ilm = ilms.Polynomial3D(**def_ilm)
    elif ilmtype == 'leg3d':
        def_ilm.update(ilmargs)
        ilm = ilms.LegendrePoly3D(**def_ilm)
    elif ilmtype == 'cheb2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.ChebyshevPoly2P1D(**def_ilm)
    elif ilmtype == 'poly2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.Polynomial2P1D(**def_ilm)
    elif ilmtype == 'leg2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.LegendrePoly2P1D(**def_ilm)
    elif ilmtype == 'barnesleg2p1d':
        def_ilm.update(ilmargs)
        ilm = ilms.BarnesStreakLegPoly2P1D(**def_ilm)
    elif ilmtype == 'barnesleg2p1dx':
        def_ilm.update(ilmargs)
        ilm = ilms.BarnesStreakLegPoly2P1DX3(**def_ilm)
    else:
        raise AttributeError("ilmtype not one of supplied options, see help")

    # setup the psf based on the choice and arguments
    if psftype == 'identity':
        def_psf.update({'params': (1.0, 1.0)})
        def_psf.update(psfargs)
        psf = psfs.IdentityPSF(**def_psf)
    elif psftype == 'gauss2d':
        def_psf.update({'params': (2.0, 4.0)})
        def_psf.update(psfargs)
        psf = psfs.AnisotropicGaussian(**def_psf)
    elif psftype == 'gauss3d':
        def_psf.update({'params': (2.0, 1.0, 4.0)})
        def_psf.update(psfargs)
        psf = psfs.AnisotropicGaussianXYZ(**def_psf)
    elif psftype == 'gauss4d':
        def_psf.update({'params': (1.5, 0.7, 3.0)})
        def_psf.update(psfargs)
        psf = psfs.Gaussian4DPoly(**def_psf)
    elif psftype == 'linescan':
        def_psf.update({
            'zrange': (0, image.shape[0]),
            'cutoffval': 1./255,
            'measurement_iterations': 3,
        })
        def_psf.update(psfargs)
        psf = exactpsf.ExactLineScanConfocalPSF(**def_psf)
    elif psftype == 'cheb-linescan':
        def_psf.update({
            'zrange': (0, image.shape[0]),
            'cutoffval': 1./255,
            'measurement_iterations': 3,
        })
        def_psf.update(psfargs)
        psf = exactpsf.ChebyshevLineScanConfocalPSF(**def_psf)
    elif psftype == 'cheb-linescan-fixedss':
        def_psf.update({
            'support_size': [31, 17, 29],
            'zrange': (0, image.shape[0]),
            'cutoffval': 1./255,
            'measurement_iterations': 3,
        })
        def_psf.update(psfargs)
        psf = exactpsf.FixedSSChebLinePSF(**def_psf)
    else:
        raise AttributeError("psftype not one of supplied options, see help")

    if slab is not None:
        slab = objs.Slab(zpos=slab+pad, shape=image.shape)
    if rawimage is not None:
        image = rawimage

    stateargs.update({'sigma': sigma})
    stateargs.update({'slab': slab})
    s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm, **stateargs)

    if ignoreimage:
        s.model_to_true_image()
    return s

