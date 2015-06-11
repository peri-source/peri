import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import pylab as pl

import copy
import scipy as sp
import numpy as np
import scipy.ndimage as nd

from colloids.cu import nbody, fields, mc
from colloids.salsa import process
from colloids.sim import Sim

nbody.initializeDevice(0)
nbody.setSeed(10)

def generate_configuration(N=8, radius=0.5, phi=0.63):
    out = Sim(N, radius=radius)
    out.init_random_2d(phi=phi)
    np.random.seed(out.seed)
    mc.setSeed(out.seed)

    perturb_radii(out, sigma=0.10)
    out.do_relaxation(2000)
    return out

def perturb_radii(out, mu=0.5, sigma=0.1):
    rad = np.zeros(out.N, dtype='float32')
    nbody.simGetRadii(out.sim, rad)
    rad = mu - sigma*np.random.rand(out.N)
    nbody.simSetRadii(out.sim, rad.astype('float32'))

def sample_hamiltonian(sim, NN=100):
    sim.do_steps(100)

def sample_radii(sim, mu=0.5, sigma=0.1, n=2):
    mc.propose_particle_radius(sim.sim, sim.nn, mu, sigma, n)

sigma = 0.05
PPL = 10 # pix-per-length

def plot_single(x, L):
    pl.figure()
    pl.imshow(genimage(x,L), cmap=mpl.cm.bone, interpolation='nearest')

def plot_compare(*imgs):
    imgs = list(imgs)
    fig = pl.figure()
    gs = gridspec.GridSpec(1, len(imgs))
    gs.update(left=0.05, right=0.9, hspace=0.05, wspace=0.1)

    for i in xrange(len(imgs)):
        ax = pl.subplot(gs[0,i])
        ax.imshow(imgs[i][4,:,:], cmap=mpl.cm.bone, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])

NNG = 32
NNZ = 16
CUT = 0
gfield = fields.createField(np.array([NNG,NNG,NNZ], dtype='int32'))
fields.setupFFT(gfield)

def gen_image(ss, donoise=False):
    t = np.zeros((NNZ,NNG,NNG), dtype='float32').flatten()

    fields.fieldSet(gfield, t)
    fields.process_image(gfield, ss.sim)

    fields.fieldGet(gfield, t)
    t = t.reshape(NNZ,NNG,NNG)
    t -= t.min()
    t /= t.max()
    tslice = np.s_[:]
    #tslice = np.s_[CUT,:,:]
    if donoise:
        noise = np.random.normal(0, sigma/1, (NNZ,NNG,NNG))
        return (t+noise)[tslice], noise[tslice]
    return t[tslice]

def likelihood(iguess, itrue):
    return np.exp(loglikelihood(igues, itrue))

def loglikelihood(iguess, itrue):
    return -((iguess - itrue)**2).sum() / (2*sigma**2)

#def dosample_mcmc():
if True:
    simtrue = generate_configuration()
    itrue,ntrue = gen_image(simtrue, True)
    xtrue = simtrue.get_pos()
    rtrue = simtrue.get_radii()

    sim = copy.deepcopy(simtrue)
    perturb_radii(sim, mu=0.50, sigma=0.0)
    sim.do_relaxation(1000)

    sim.set_param_hs()
    sim.do_steps(250)
    sim.set_param_mc()

    istart = gen_image(sim)
    xstart = sim.get_pos()
    rstart = sim.get_radii()

    nwarm = int(1e4)
    nsteps = nwarm + int(2e3)

    rguess = 0*rtrue
    guess, std, total = 0*xtrue, 0*xtrue, 0
    positions, crosses = [], []
    lnew = lold = loglikelihood(istart, itrue)
    pnew = pold = 1#logimageprior(istart, ptrue)
    accepts = 0

    print "goal:", loglikelihood(ntrue, 0*ntrue)
    r = np.sqrt(((sim.get_pos()- xtrue)**2).sum(-1))
    print r.mean()

    for i in xrange(nsteps):
        vsig = 1.0000
        simcopy = copy.deepcopy(sim)
        nbody.init_set_random_velocities(sim.sim, 0, vsig)
        startv = sim.get_vel()

        if i % 2 == 0:
            sample_hamiltonian(sim, 20)
        else:
            sample_radii(sim, mu=0.5, sigma=0.10, n=4)

        t = sim.get_pos()
        r = sim.get_radii()
        im = gen_image(sim)
        lnew = loglikelihood(im, itrue)
        #pnew = logimageprior(PPL*t, ptrue)

        endv = sim.get_vel()
        vfact1 = np.exp(-(startv**2).sum(axis=-1).mean()/(2*(vsig**2)))
        vfact2 = np.exp(-(endv**2).sum(axis=-1).mean()/(2*(vsig**2)))

        acceptance = np.exp(lnew-lold)
        #acceptance *= pnew/pold
        #acceptance *= np.exp(pnew-pold)
        acceptance *= (vfact1/vfact2)

        if i % 100 == 1:
            print i, 'ratio', lnew, lold, pnew/pold, acceptance, total, float(accepts)/i

        if np.random.rand() < min(acceptance, 1):
            lold = lnew
            pold = pnew
            accepts += 1
        else:
            nbody.simsys_cpu2cpu(simcopy.sim, sim.sim)
        if i > nwarm and i % 10 == 0:
            total += 1
            guess += t
            std += t*t
            rguess += r

    rguess /= total
    guess = guess / total
    std = np.sqrt(std/total - guess**2)

    sim.set_pos(guess.flatten())
    sim.set_radii(rguess.flatten())
    iguess = gen_image(sim)

    r = PPL*np.sqrt(((guess - xtrue)**2)[:,:2].sum(-1))
    print r.mean(), r.std()#, std
    plot_compare(itrue, istart, iguess, itrue-iguess-ntrue)
