import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
import pickle

from cbamf import states, runner, initializers
from cbamf.comp import objs, psfs, ilms
from cbamf.viz import plots

def sample(s, sweeps, burn):
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        runner.sample_particles(s, stepout=0.1)
        runner.sample_block(s, 'ilm', stepout=0.1)
        runner.sample_block(s, 'off', stepout=0.1)
        runner.sample_block(s, 'psf', stepout=0.1)
        runner.sample_block(s, 'zscale')

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
    return h

def wrap_feature(filename):
    try:
        feature_snap(filename)
    except Exception as e:
        return

def feature_snap(filename):
    ORDER = (1,1,1)
    sweeps = 10
    samples = 7
    burn = sweeps - samples

    sigma = 0.0141
    PSF = (2.402, 5.151)
    PAD, FSIZE, RAD, zscale = 22, 6, 7.3, 1.06
    raw = initializers.load_tiff(filename)

    feat = initializers.normalize(raw, True)
    xstart, proc = initializers.local_max_featuring(feat, FSIZE, FSIZE/3.)
    itrue = initializers.normalize(feat, True)
    itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
    xstart += PAD
    rstart = RAD*np.ones(xstart.shape[0])
    initializers.remove_overlaps(xstart, rstart)

    imsize = itrue.shape
    obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
    psf = psfs.AnisotropicGaussian(PSF, shape=imsize, threads=1)
    ilm = ilms.Polynomial3D(order=ORDER, shape=imsize)
    s = states.ConfocalImagePython(itrue, obj=obj, psf=psf, ilm=ilm,
            zscale=zscale, pad=16, sigma=sigma)

    runner.sample_particles(s, stepout=1)
    runner.sample_particles(s, stepout=0.3)
    h = sample(s, sweeps=sweeps, burn=burn)
    pickle.dump(h, open(filename+"-samples.pkl", 'w'))

from multiprocessing import Pool
import glob

if __name__ == '__main__':
    fldr = '/b/bierbaum/bamf-salsa-polycrystal/'
    files = glob.glob(fldr+'*.tif')

    pool = Pool(12)
    pool.map(wrap_feature, files)
