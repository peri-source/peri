import pickle
import numpy as np
import scipy as sp
import scipy.ndimage as nd

import common
from peri.test import init

def zjitter(jitter=0.0, radius=5):
    """
    scan jitter is in terms of the fractional pixel difference when
    moving the laser in the z-direction
    """
    psfsize = np.array([2.0, 1.0, 3.0])

    # create a base image of one particle
    s0 = init.create_single_particle_state(imsize=4*radius, 
            radius=radius, psfargs={'params': psfsize, 'error': 1e-6})
    sl = np.s_[s0.pad:-s0.pad,s0.pad:-s0.pad,s0.pad:-s0.pad]

    # add up a bunch of trajectories
    finalimage = 0*s0.get_model_image()[sl]
    position = 0*s0.obj.pos[0]

    for i in xrange(finalimage.shape[0]):
        offset = jitter*np.random.randn(3)*np.array([1,0,0])
        s0.obj.pos[0] = np.array(s0.image.shape)/2 + offset
        s0.reset()

        finalimage[i] = s0.get_model_image()[sl][i]
        position += s0.obj.pos[0]

    position /= float(finalimage.shape[0])

    # place that into a new image at the expected parameters
    s = init.create_single_particle_state(imsize=4*radius, sigma=0.05,
            radius=radius, psfargs={'params': psfsize, 'error': 1e-6})
    s.reset()

    # measure the true inferred parameters
    return s, finalimage, position

def dorun(SNR=20, njitters=20, samples=10, noise_samples=10, sweeps=20, burn=10):
    """
    we want to display the errors introduced by pixelation so we plot:
        * CRB, sampled error vs exposure time

    a = dorun(ntimes=10, samples=5, noise_samples=5, sweeps=20, burn=8)
    """
    jitters = np.logspace(-6, np.log10(0.5), njitters)
    crbs, vals, errs, poss = [], [], [], []

    for i,t in enumerate(jitters):
        print '###### jitter', i, t

        for j in xrange(samples):
            print 'image', j, '|', 
            s,im,pos = zjitter(jitter=t)

            # typical image
            common.set_image(s, im, 1.0/SNR)
            crbs.append(crb(s))

            val, err = sample(s, im, 1.0/SNR, N=noise_samples, sweeps=sweeps, burn=burn)
            poss.append(pos)
            vals.append(val)
            errs.append(err)


    shape0 = (njitters, samples, -1)
    shape1 = (njitters, samples, noise_samples, -1)

    crbs = np.array(crbs).reshape(shape0)
    vals = np.array(vals).reshape(shape1)
    errs = np.array(errs).reshape(shape1)
    poss = np.array(poss).reshape(shape0)

    return  [crbs, vals, errs, poss, jitters]

def doplot(prefix='/media/scratch/peri/does_matter/z-jitter', snrs=[20,50,200,500]):

    s,im,pos = zjitter(jitter=0.1, radius=5)
    diff = (im - s.get_model_image()[s.inner])
    nn = np.s_[:,:,im.shape[2]/2]
    image0, image1 = im[nn], diff[nn]

    xs, crbs, errors = [], [], []
    for i, snr in enumerate(snrs):
        fn = prefix+'-snr-'+str(snr)+'.pkl'
        crb, val, err, pos, time = pickle.load(open(fn))

        xs.append(time)
        crbs.append(common.dist(crb))
        errors.append(common.errs(val, pos))

    labels = ['SNR %i' % i for i in snrs]

    common.doplot(image0, image1, xs, crbs, errors, labels, diff_image_scale=0.05,
        dolabels=True, multiple_crbs=True, xlim=(0,time[-1]), ylim=(1e-4,1e0), highlight=None,
        detailed_labels=False, title=r"$z$-scan jitter", xlabel=r"$z$-scan NSR")
