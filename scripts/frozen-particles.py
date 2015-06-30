import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from cbamf import states, runner, initializers
from cbamf.comp import objs, psfs, ilms
from cbamf.viz import plots

def image(p, r, diff, imsize, psf, ilm, sigma):
    ilm = ilms.Polynomial3D(order=(1,1,1), shape=imsize)
    ilm.update(np.array([0]))

    o = objs.SphereCollectionRealSpace(pos=p.reshape(-1,3), rad=r, shape=imsize)
    s = states.ConfocalImagePython(diff, obj=o, psf=psf, ilm=ilm, pad=16, sigma=sigma, constoff=True, offset=0.45)
    return s.get_model_image()

def score(n, pos, imsize, psf, ilm, sigma, d):
    t = image(pos[n].reshape(-1, 3), np.array([6]), d, imsize, psf, ilm, sigma)
    return ((t-d)**2).sum()

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
xstart, proc = initializers.local_max_featuring(feat, FSIZE, FSIZE/3.)

itrue = initializers.normalize(raw[zstart:,:IMSIZE,:IMSIZE], not INVERT)
itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
xstart += PAD
rstart = RAD*np.ones(xstart.shape[0])
initializers.remove_overlaps(xstart, rstart, zscale=zscale)

imsize = itrue.shape
obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian(PSF, shape=imsize)
ilm = ilms.Polynomial3D(order=ORDER, shape=imsize)
ilm.from_data(itrue, mask=itrue > -10)

diff = (ilm.get_field() - itrue)[itrue > -10].max()
params = ilm.get_params()
params[0] += diff
#params = np.load("/media/scratch/bamf/frozen-particles/ilm.npy")
ilm.update(params)

s = states.ConfocalImagePython(itrue, obj=obj, psf=psf, ilm=ilm,
        zscale=zscale, pad=16, sigma=sigma, constoff=True, offset=0.45)

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
