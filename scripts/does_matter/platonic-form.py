import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd

import common
from peri import const, runner
from peri.test import init

def create_comparison_state(image, position, radius=5.0, snr=20,
        method='constrained-cubic', extrapad=2, zscale=1.0):
    """
    Take a platonic image and position and create a state which we can
    use to sample the error for peri. Also return the blurred platonic
    image so we can vary the noise on it later
    """
    # first pad the image slightly since they are pretty small
    image = common.pad(image, extrapad, 0)

    # place that into a new image at the expected parameters
    s = init.create_single_particle_state(imsize=np.array(image.shape), sigma=1.0/snr,
            radius=radius, psfargs={'params': np.array([2.0, 1.0, 3.0]), 'error': 1e-6, 'threads': 2},
            objargs={'method': method}, stateargs={'sigmapad': False, 'pad': 4, 'zscale': zscale})
    s.obj.pos[0] = position + s.pad + extrapad
    s.reset()
    s.model_to_true_image()

    timage = 1-np.pad(image, s.pad, mode='constant', constant_values=0)
    timage = s.psf.execute(timage)
    return s, timage[s.inner]

def dorun(method, platonics=None, nsnrs=20, noise_samples=30, sweeps=30, burn=15):
    """
    platonics = create_many_platonics(N=50)
    dorun(platonics)
    """
    sigmas = np.logspace(np.log10(1.0/2048), 0, nsnrs)
    crbs, vals, errs, poss = [], [], [], []

    for sigma in sigmas:
        print "#### sigma:", sigma

        for i, (image, pos) in enumerate(platonics):
            print 'image', i, '|', 
            s,im = create_comparison_state(image, pos, method=method)

            # typical image
            set_image(s, im, sigma)
            crbs.append(crb(s))

            val, err = sample(s, im, sigma, N=noise_samples, sweeps=sweeps, burn=burn)
            poss.append(pos)
            vals.append(val)
            errs.append(err)


    shape0 = (nsnrs, len(platonics), -1)
    shape1 = (nsnrs, len(platonics), noise_samples, -1)

    crbs = np.array(crbs).reshape(shape0)
    vals = np.array(vals).reshape(shape1)
    errs = np.array(errs).reshape(shape1)
    poss = np.array(poss).reshape(shape0)

    return  [crbs, vals, errs, poss, sigmas]

def doplot(prefix='/media/scratch/peri/bad/platonic-form',
        images='/media/scratch/peri/does_matter/platonic-form-images.pkl',
        forms=['ev-exact-gaussian', 'ev-constrained-cubic', 'ev-lerp', 'ev-logistic'],
        labels=['EG', 'CC', 'LP', 'LG']):

    print "Generating a image comparison"
    images = pickle.load(open(images))
    im,pos = images[0]
    s,im = create_comparison_state(im, pos)
    common.set_image(s, im, 0.0001)
    h,l = runner.do_samples(s, 20, 0, stepout=0.1, quiet=True)
    h = h.mean(axis=0)
    s.obj.pos = np.array([h[:3]])
    s.obj.rad = np.array([h[3]])
    s.reset()

    nn = np.s_[:,:,im.shape[2]/2]
    diff = (im - s.get_model_image()[s.inner])
    image0, image1 = im[nn], diff[nn]

    print "Plotting"
    xs, crbs, errors = [], [], []
    for i, form in enumerate(forms):
        fn = prefix+'-'+form+'.pkl'
        crb, val, err, pos, time = pickle.load(open(fn))
        pos += 16

        xs.append(time)
        crbs.append(common.dist(crb))
        errors.append(common.errs(val, pos))

    common.doplot(image0, image1, xs, crbs, errors, labels, diff_image_scale=0.05,
        dolabels=True, multiple_crbs=False, xlim=None, ylim=None, highlight=None,
        detailed_labels=True, title='Platonic form', xlabel=r"$1/\rm{SNR}$")
