import numpy as np
import scipy as sp

from cbamf.comp import psfs, ilms, objs
from cbamf.test import poissondisks
from cbamf import states

#=======================================================================
# Generating fake data
#=======================================================================
def create_state_random_packing(imsize, radius=5.0, phi=0.5, sigma=0.05,
        psf=(1.6, 3.0), seed=None):
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if not hasattr(imsize, '__iter__'):
        imsize = np.array((imsize,)*3)

    blank = np.zeros(imsize, dtype='float')
    disks = poissondisks.DiskCollection(imsize-radius, 2*radius)
    xstart = disks.get_positions() + radius/4
    image, pos, rad = states.prepare_for_state(blank, xstart, radius)
    
    obj = objs.SphereCollectionRealSpace(pos=pos, rad=rad, shape=image.shape)
    psf = psfs.AnisotropicGaussian(psf, shape=image.shape)
    ilm = ilms.Polynomial3D(order=(1,1,1), shape=image.shape)
    s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm, sigma=sigma)
    s.model_to_true_image()
    
    return s

def create_single_particle_state(imsize, radius=5.0, sigma=0.05,
        psf=(2.0, 4.0), seed=None):
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if not hasattr(imsize, '__iter__'):
        imsize = np.array((imsize,)*3)

    image, pos, rad = states.prepare_for_state(np.zeros(imsize),
            imsize.reshape(-1,3)/2.0, radius)
    
    imsize = image.shape
    obj = objs.SphereCollectionRealSpace(pos=pos, rad=rad, shape=imsize)
    psf = psfs.AnisotropicGaussian(psf, shape=imsize)
    ilm = ilms.Polynomial3D(order=(1,1,1), shape=imsize)
    s = states.ConfocalImagePython(image, obj=obj, psf=psf, ilm=ilm, sigma=sigma)
    s.model_to_true_image()
    return s
