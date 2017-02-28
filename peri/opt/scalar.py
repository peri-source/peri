from builtins import range, zip

import time
import numpy as np

from peri.logger import log

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
    blocks = state.param_particle(index)

    if not doradius:
        blocks = blocks[:-1]

    g = state.gradloglikelihood(blocks=blocks)
    if method == 'gn':
        h = state.jtj(blocks=blocks)
    if method == 'nr':
        h = state.hessloglikelihood(blocks=blocks)
    step = np.linalg.solve(h, g)

    h = np.zeros_like(g)
    for i in range(len(g)):
        state.update(blocks[i], state.state[blocks[i]] - step[i])
    return g,h

def optimize_particles(state, *args, **kwargs):
    for i in state.active_particles():
        optimize_particle(state, i, *args, **kwargs)

def modify(state, blocks, vec):
    for bl, val in zip(blocks, vec):
        state.update(bl, np.array([val]))


def residual(vec, state, blocks, relax_particles=True):
    log.info('res {}'.format(state.loglikelihood()))
    modify(state, blocks, vec)

    for i in range(3):
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
    for i in range(N):
        log.info('{}'.format(state.loglikelihood()))
        grad = state.gradloglikelihood()
        n = state.state + 1.0/np.abs(grad).max() * ratio * grad
        state.set_state(n)
        log.info('{}'.format(state.loglikelihood()))

