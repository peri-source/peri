import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
import pickle

from cbamf import states, runner, initializers
from cbamf.comp import objs, psfs, ilms
from cbamf.viz import plots

def pad_fake_particles(pos, rad, nfake):
    opos = np.vstack([pos, np.zeros((nfake, 3))])
    orad = np.hstack([rad, rad[0]*np.ones(nfake)])
    return opos, orad

#raise IOError
ORDER = (4,4,3)
sweeps = 30
samples = 20
burn = sweeps - samples

sigma = 0.05
PSF = (2.4, 4.6)
PAD, FSIZE, RAD, INVERT, IMSIZE, zstart, zscale = 16, 5, 5.0, True, 78, 14, 1.056
raw = initializers.load_tiff("/media/scratch/bamf/frozen-particles/zstack_dx0/0.tif")

feat = initializers.normalize(raw[zstart:,:IMSIZE,:IMSIZE], INVERT)
feat = initializers.remove_background(feat, order=ORDER)
#xstart, proc = initializers.local_max_featuring(feat, FSIZE, FSIZE/3.)
xstart, rstart = pickle.load(open("/media/scratch/fff.pkl"))

itrue = initializers.normalize(raw[zstart:,:IMSIZE,:IMSIZE], not INVERT)
itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
#xstart += PAD
#rstart = RAD*np.ones(xstart.shape[0])
#initializers.remove_overlaps(xstart, rstart, zscale=zscale)
nfake = 20
#xstart, rstart = pad_fake_particles(xstart, rstart, nfake)

imsize = itrue.shape
obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize, pad=nfake)
psf = psfs.AnisotropicGaussian(PSF, shape=imsize)
ilm = ilms.Polynomial3D(order=ORDER, shape=imsize)
ilm.from_data(itrue, mask=itrue > -10)

diff = (ilm.get_field() - itrue)[itrue > -10].max()
params = ilm.get_params()
params[0] += diff/2
#params = np.load("/media/scratch/bamf/frozen-particles/ilm.npy")
ilm.update(params)

s = states.ConfocalImagePython(itrue, obj=obj, psf=psf, ilm=ilm,
        zscale=zscale, pad=16, sigma=sigma, constoff=True, offset=0.45,
        doprior=False)

import scipy.ndimage as nd

def sample_add(s, rad=5):
    diff = s.get_model_image() - s.image

    smoothdiff = nd.gaussian_filter(diff, 1)
    eq = smoothdiff == smoothdiff.max()
    lbl = nd.label(eq)[0]
    pos = np.array(nd.center_of_mass(eq, lbl, np.unique(lbl)))[1:]

    n = s.obj.typ.argmin()
    bp = s.block_particle_pos(n)
    br = s.block_particle_rad(n)
    bt = s.block_particle_typ(n)
    s.update(bp, pos)
    s.update(br, np.array([rad]))
    s.update(bt, np.array([1]))

    bl = s.blocks_particle(n)
    runner.sample_state(s, bl, stepout=1, N=3)

    diff2 = s.get_model_image() - s.image

    if np.log(np.random.rand()) > (diff2**2).sum() - (diff**2).sum():
        neighs = s.nbl.neighs[n].keys()
        print "Added, modifying neighbors"
        for neigh in neighs:
            print neigh
            bl = s.blocks_particle(neigh)
            runner.sample_state(s, bl, stepout=0.3)
        return True
    else:
        s.update(bt, np.array([0]))
    return False

def sample_remove(s, rad=5):
    diff = s.get_model_image() - s.image

    smoothdiff = nd.gaussian_filter(diff, 1)
    eq = smoothdiff == smoothdiff.min()
    lbl = nd.label(eq)[0]
    pos = np.array(nd.center_of_mass(eq, lbl, np.unique(lbl)))[1:]

    n = ((s.obj.pos - pos)**2).sum(axis=0).argmin()
    bt = s.block_particle_typ(n)
    s.update(bt, np.array([0]))

    diff2 = s.get_model_image() - s.image

    if np.log(np.random.rand()) > (diff2**2).sum() - (diff**2).sum():
        neighs = s.nbl.neighs[n].keys()
        print "Removed, modifying neighbors"
        for neigh in neighs:
            print neigh
            bl = s.blocks_particle(neigh)
            runner.sample_state(s, bl, stepout=0.3)
        return True
    else:
        s.update(bt, np.array([1]))
    return False

def sample_good_particles(s, N=1):
    m = np.arange(s.N)[s.obj.typ == 1]
    for j in xrange(N):
        for i in m:
            print i
            b = s.blocks_particle(i)
            runner.sample_state(s, b, stepout=1)

def iterate(s, n=10):
    for i in xrange(n):
        if sample_add(s):
            #sample_good_particles(s)
            runner.sample_block(s, 'ilm', stepout=0.03)
            runner.sample_block(s, 'psf', stepout=0.03)
            runner.sample_block(s, 'off', stepout=0.03)

        if sample_remove(s):
            #sample_good_particles(s)
            runner.sample_block(s, 'ilm', stepout=0.03)
            runner.sample_block(s, 'psf', stepout=0.03)
            runner.sample_block(s, 'off', stepout=0.03)

def sample(s):
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        runner.sample_particles(s, stepout=0.1)
        #runner.sample_block(s, 'off', stepout=0.1)
        runner.sample_block(s, 'psf', stepout=0.1)
        #runner.sample_block(s, 'ilm', stepout=0.1)
        runner.sample_block(s, 'zscale')

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
    return h
