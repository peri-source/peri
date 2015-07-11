import sys
import numpy as np
import scipy.ndimage as nd

from cbamf import const
from cbamf.mc import samplers, engines, observers

#=============================================================================
# Sampling methods that run through blocks and sample
#=============================================================================
def sample_state(state, blocks, stepout=1, slicing=True, N=1, doprint=False):
    eng = engines.SequentialBlockEngine(state)
    opsay = observers.Printer()
    ohist = observers.HistogramObserver(block=blocks[0])
    eng.add_samplers([samplers.SliceSampler(stepout, block=b) for b in blocks])

    eng.add_likelihood_observers(opsay) if doprint else None
    eng.add_state_observers(ohist)

    eng.dosteps(N)
    return ohist

def scan_ll(state, element, size=0.1, N=1000):
    start = state.state[element]

    ll = []
    vals = np.linspace(start-size, start+size, N)
    for val in vals:
        state.update(element, val)
        l = state.loglikelihood()
        ll.append(l)

    state.update(element, start)
    return vals, np.array(ll)

def scan_noise(image, state, element, size=0.01, N=1000):
    start = state.state[element]

    xs, ys = [], []
    for i in xrange(N):
        print i
        test = image + np.random.normal(0, state.sigma, image.shape)
        x,y = sample_ll(test, state, element, size=size, N=300)
        state.update(element, start)
        xs.append(x)
        ys.append(y)

    return xs, ys

def sample_particles(state, stepout=1):
    print '{:-^39}'.format(' POS / RAD ')
    for particle in xrange(state.obj.N):
        if not state.isactive(particle):
            continue

        print particle
        sys.stdout.flush()

        blocks = state.blocks_particle(particle)
        sample_state(state, blocks, stepout=stepout)

    return state.state.copy()

def sample_particle_pos(state, stepout=1):
    print '{:-^39}'.format(' POS ')
    for particle in xrange(state.obj.N):
        if not state.isactive(particle):
            continue

        print particle
        sys.stdout.flush()

        blocks = state.blocks_particle(particle)[:-1]
        sample_state(state, blocks, stepout=stepout)

    return state.state.copy()

def sample_particle_rad(state, stepout=1):
    print '{:-^39}'.format(' RAD ')
    for particle in xrange(state.obj.N):
        if not state.isactive(particle):
            continue

        print particle
        sys.stdout.flush()

        blocks = [state.blocks_particle(particle)[-1]]
        sample_state(state, blocks, stepout=stepout)

    return state.state.copy()

def sample_block(state, blockname, explode=True, stepout=0.1):
    print '{:-^39}'.format(' '+blockname.upper()+' ')
    blocks = [state.create_block(blockname)]

    if explode:
        blocks = state.explode(blocks[0])

    return sample_state(state, blocks, stepout)

def sample_block_list(state, blocklist, stepout=0.1):
    for bl in blockslist:
        sample_block(state, bl, stepout=stepout)
    return state.state.copy(), state.loglikelihood()

def do_samples(s, sweeps, burn, stepout=0.1):
    h = []
    ll = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        sample_particles(s, stepout=stepout)
        sample_block(s, 'psf', stepout=stepout)
        sample_block(s, 'ilm', stepout=stepout)
        sample_block(s, 'off', stepout=stepout)
        sample_block(s, 'zscale', stepout=stepout)

        if i >= burn:
            h.append(s.state.copy())
            ll.append(s.loglikelihood())

    h = np.array(h)
    ll = np.array(ll)
    return h, ll

#=============================================================================
# Optimization methods like gradient descent
#=============================================================================
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
    state.create_final_image()
    return -state.loglikelihood()

def gradloglikelihood(vec, state):
    state.set_state(vec)
    return -state.gradloglikelihood()

def gradient_descent(state, method='L-BFGS-B'):
    from scipy.optimize import minimize

    bounds = build_bounds(state)
    minimize(loglikelihood, state.state, args=(state,),
            method=method, jac=gradloglikelihood, bounds=bounds)

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
def pad_fake_particles(pos, rad, nfake):
    opos = np.vstack([pos, np.zeros((nfake, 3))])
    orad = np.hstack([rad, rad[0]*np.ones(nfake)])
    return opos, orad

def zero_particles(n):
    return np.zeros((n,3)), np.ones(n), np.zeros(n)

def raw_to_state(rawimage, rad=7.3, frad=9, imsize=-1, imzstart=0, imzstop=-1, invert=False,
        pad_for_extra=True, threads=-1, phi=0.5, sigma=0.05, zscale=1.0,
        PSF=(2.0, 4.0), ORDER=(3,3,2)):
    from cbamf import states, initializers
    from cbamf.comp import objs, psfs, ilms

    itrue = initializers.normalize(rawimage[imzstart:imzstop,:imsize,:imsize], invert)
    feat = initializers.remove_background(itrue.copy(), order=ORDER)

    xstart, proc = initializers.local_max_featuring(feat, frad, frad/3.)
    image, pos, rad = states.prepare_for_state(itrue, xstart, rad, invert=True)

    if pad_for_extra:
        nfake = xstart.shape[0]
        pos, rad = pad_fake_particles(pos, rad, nfake)
    else:
        nfake = None

    imsize = image.shape
    obj = objs.SphereCollectionRealSpace(pos=pos, rad=rad, shape=imsize, pad=nfake)
    psf = psfs.AnisotropicGaussian(PSF, shape=imsize, threads=threads)
    ilm = ilms.LegendrePoly3D(order=ORDER, shape=imsize)
    ilm.from_data(image, mask=image > const.PADVAL)

    diff = (ilm.get_field() - image)
    ptp = diff[image > const.PADVAL].ptp()

    params = ilm.get_params()
    params[0] += ptp * (1-phi)
    ilm.update(params)

    s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm,
            zscale=zscale, sigma=sigma, offset=ptp, doprior=(not pad_for_extra),
            nlogs=(not pad_for_extra), varyn=pad_for_extra)

    return s

def feature_addsubtract(s, sweeps=3, rad=5):
    from cbamf import states, initializers
    addsubtract(s, rad=rad, sweeps=sweeps, particle_group_size=s.N/(sweeps+1))

    """
    initializers.remove_overlaps(s.obj.pos, s.obj.rad, zscale=s.zscale)
    s = states.ConfocalImagePython(s.image, obj=s.obj, psf=s.psf, ilm=s.ilm,
            zscale=s.zscale, sigma=s.sigma, offset=s.offset, doprior=True,
            nlogs=True, varyn=False)
    """
    return s

def feature(rawimage, sweeps=20, samples=15, rad=7.3, frad=9,
        imsize=-1, imzstart=0, imzstop=-1, zscale=1.06, sigma=0.02, invert=False,
        PSF=(2.0, 4.1), ORDER=(3,3,2), threads=-1, addsubtract=True, phi=0.5):

    burn = sweeps - samples

    print "Initial featuring"
    s = raw_to_state(rawimage, rad=rad, frad=frad, imsize=imsize,
            imzstart=imzstart, imzstop=imzstop, invert=invert, pad_for_extra=addsubtract,
            threads=threads, phi=phi, sigma=sigma, zscale=zscale, PSF=PSF, ORDER=ORDER)

    if addsubtract:
        print "Adding, removing particles"
        s = feature_addsubtract(s, rad=rad)

    return do_samples(s, sweeps, burn, stepout=0.10)

def trim_extra_particles(s):
    # TODO -- take out particles that are already
    # not able to be sampled (according to the sampler methods)
    raise AttributeError("STUB")

#=======================================================================
# More involved featuring functions using MC
#=======================================================================
def sample_n_add(s, rad, tries=5):
    diff = (s.get_model_image() - s.get_true_image()).copy()

    smoothdiff = nd.gaussian_filter(diff, rad/2.0)
    maxfilter = nd.maximum_filter(smoothdiff, size=rad)
    eq = smoothdiff == maxfilter
    lbl = nd.label(eq)[0]
    ind = np.sort(np.unique(lbl))[1:]
    pos = np.array(nd.center_of_mass(eq, lbl, ind)).astype('int')
    ind = np.arange(len(pos))

    val = [maxfilter[tuple(pos[i])] for i in ind]
    vals = sorted(zip(val, ind))

    accepts = 0
    for _, i in vals[-tries:][::-1]:
        ll0 = s.loglikelihood()

        p = pos[i].reshape(-1,3)
        n = s.obj.typ.argmin()

        bp = s.block_particle_pos(n)
        br = s.block_particle_rad(n)
        bt = s.block_particle_typ(n)

        s.update(bp, p)
        s.update(br, np.array([rad]))
        s.update(bt, np.array([1]))

        bl = s.blocks_particle(n)[:-1]
        sample_state(s, bl, stepout=1, N=1)

        ll1 = s.loglikelihood()

        print p, ll0, ll1
        if not (np.log(np.random.rand()) < (ll0**2).sum() - (ll1**2).sum()):
            s.update(bt, np.array([0]))
        else:
            accepts += 1
    return accepts

def sample_n_remove(s, rad, tries=5):
    diff = (s.get_model_image() - s.get_true_image()).copy()

    smoothdiff = nd.gaussian_filter(diff, rad/2.0)
    maxfilter = nd.maximum_filter(smoothdiff, size=rad)
    eq = smoothdiff == maxfilter
    lbl = nd.label(eq)[0]
    ind = np.sort(np.unique(lbl))[1:]
    pos = np.array(nd.center_of_mass(eq, lbl, ind)).astype('int')
    ind = np.arange(len(pos))

    val = [maxfilter[tuple(pos[i])] for i in ind]
    vals = sorted(zip(val, ind))

    accepts = 0
    for _, i in vals[-tries:]:
        ll0 = s.loglikelihood()

        n = ((s.obj.pos - pos[i])**2).sum(axis=0).argmin()

        bt = s.block_particle_typ(n)
        s.update(bt, np.array([0]))

        ll1 = s.loglikelihood()

        print s.obj.pos[n], ll0, ll1
        if not (np.log(np.random.rand()) < (ll0**2).sum() - (ll1**2).sum()):
            s.update(bt, np.array([1]))
        else:
            accepts += 1
    return accepts

def addsubtract(s, rad, sweeps=3, particle_group_size=100, add_remove_tries=8):
    total = 1

    print "Relaxing current configuration"
    for i in xrange(2):
        sample_block(s, 'ilm', stepout=0.1)
        sample_block(s, 'off', stepout=0.1)

    for i in xrange(sweeps):
        total = 0
        accepts = 1
        while accepts > 0 and total <= particle_group_size:
            accepts = 0
            accepts += sample_n_add(s, rad=rad, tries=add_remove_tries)
            accepts += sample_n_remove(s, rad=rad, tries=add_remove_tries/2)

            print "Added / removed %i particles" % accepts
            total += accepts

        print "Relaxing pos / radii"
        sample_particle_pos(s, stepout=2)
        sample_particle_rad(s, stepout=2)

        do_samples(s, 2, 2)
