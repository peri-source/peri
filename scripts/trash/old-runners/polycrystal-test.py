import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pylab as pl
from peri import states, runner, initializers
from peri.comp import objs, psfs, ilms
from peri.viz import plots

ORDER = (3,3,2)
sweeps = 30
samples = 20
burn = sweeps - samples

sigma = 0.03041
PSF = (2.402, 4.151)
PAD, FSIZE, RAD, INVERT, IMSIZE, zstart, zscale = 24, 5, 5.04, True, 64, 5, 1.0717
raw = initializers.load_tiff('/media/scratch/bamf/tmp/hyperfine_dz015_N50_3.tif_t001.tif')

feat = initializers.normalize(raw[zstart:,:IMSIZE,:IMSIZE], INVERT)
xstart, proc = initializers.local_max_featuring(feat, FSIZE, FSIZE/3.)
itrue = initializers.normalize(feat, True)
itrue = np.pad(itrue, PAD, mode='constant', constant_values=-10)
xstart += PAD
rstart = RAD*np.ones(xstart.shape[0])

nfake = xstart.shape[0]
initializers.remove_overlaps(xstart, rstart)
xstart, rstart = runner.pad_fake_particles(xstart, rstart, nfake)

imsize = itrue.shape
obj = objs.SphereCollectionRealSpace(pos=xstart, rad=rstart, shape=imsize)
psf = psfs.AnisotropicGaussian(PSF, shape=imsize, threads=1)
ilm = ilms.LegendrePoly3D(order=ORDER, shape=imsize)

ilm.from_data(itrue, mask=itrue > -10)

phi = 0.5
diff = (ilm.get_field() - itrue)
ptp = diff[itrue > -10].ptp()

params = ilm.get_params()
params[0] += ptp * (1-phi)
ilm.update(params)

s = states.ConfocalImagePython(itrue, obj=obj, psf=psf, ilm=ilm,
        zscale=zscale, pad=PAD, sigma=sigma, constoff=True, doprior=False, nlogs=False, offset=ptp)

def sample(s):
    h = []
    for i in xrange(sweeps):
        print '{:=^79}'.format(' Sweep '+str(i)+' ')

        runner.sample_particles(s, stepout=0.1)
        runner.sample_block(s, 'ilm', stepout=0.1)
        runner.sample_block(s, 'off', stepout=0.1)
        runner.sample_block(s, 'psf', stepout=0.1)
        runner.sample_block(s, 'zscale', stepout=0.1)

        if i > burn:
            h.append(s.state.copy())

    h = np.array(h)
    return h
