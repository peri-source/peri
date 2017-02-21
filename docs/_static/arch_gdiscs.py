from __future__ import print_function

import peri.util
import peri.models
import peri.states
import peri.comp.comp

from peri.fft import fft, fftkwargs, fftnorm
from peri.comp.comp import GlobalScalar
from peri.opt import optimize as opt

import numpy as np
import matplotlib.pyplot as pl

class GaussianDiscModel(peri.models.Model):
    def __init__(self):
        # gives human readable labels to equation variables
        varmap = {'P': 'psf', 'D': 'disc', 'C': 'off'}
        
        # gives the full model equation of applying the psf P to an image
        # of discs D and adding a background field
        modelstr = {'full' : 'P(D) + C'}

        # calls the super-class' init
        super(GaussianDiscModel, self).__init__(
            modelstr=modelstr, varmap=varmap, registry=None
        )

class GaussianPSF(peri.comp.comp.Component):
    def __init__(self, sigmas=(1.0, 1.0)):
        # setup the parameters and values which will be passed to super
        super(GaussianPSF, self).__init__(
            params=['psf-sx', 'psf-sy'], values=sigmas, category='psf'
        )

    def get(self):
        """
        Since we wish to use the GaussianPSF in the model by calling P(D), the
        get function will simply return this object and we will override
        __call__ so that we can use P(...).
        """
        return self

    def __call__(self, field):
        """
        Accept a field, apply the point-spread-function, and return
        the resulting image of a blurred field
        """
        # in order to avoid translations from the psf, we must create
        # real-space vectors that are zero in the corner and go positive
        # in one direction and negative on the other side of the image
        tile = peri.util.Tile(field.shape)
        rx, ry = tile.kvectors(norm=1.0/tile.shape)

        # get the sigmas from ourselves
        sx, sy = self.values

        # calculate the real-space psf from the r-vectors and sigmas
        # normalize based on the calculated values, no the usual normalization
        psf = np.exp(-((rx/sx)**2 + (ry/sy)**2)/2)
        psf = psf / psf.sum()
        self.psf = psf

        # perform the convolution with ffts and return the result
        out = fft.fftn(fft.ifftn(field)*fft.ifftn(psf))
        return fftnorm(np.real(out))

    def get_padding_size(self, tile):
        # claim that the necessary padding size for the convolution is
        # the size of the padding of the image itself for now
        return peri.util.Tile(self.inner.l)

    def get_update_tile(self, params, values):
        # if we update the psf params, we must update the entire image
        return self.shape

class PlatonicDiscs(peri.comp.comp.Component):
    def __init__(self, positions, radii):
        comp = ['x', 'y', 'a']
        params, values = [], []

        # apply using naming scheme to the parameters associated with the
        # individual discs in the object pos and rad
        for i, (pos, rad) in enumerate(zip(positions, radii)):
            params.extend(['disc-{}-{}'.format(i, c) for c in comp])
            values.extend([pos[0], pos[1], rad])

        # use our super-class structure to keep track of these parameters
        self.N = len(positions)
        super(PlatonicDiscs, self).__init__(
            params=params, values=values, category='disc',
        )

    def draw_disc(self, rvec, i):
        # get the position and radii parameters cooresponding to this particle
        pparams = ['disc-{}-{}'.format(i, c) for c in ['x', 'y']]
        rparams = 'disc-{}-a'.format(i)

        # get the actual values of these parameters
        pos = np.array(self.get_values(pparams))
        rad = self.get_values(rparams)

        # draw the disc using the provided rvecs and now pos and rad
        dist = np.sqrt(((rvec - pos)**2).sum(axis=-1))
        return 1.0/(1.0 + np.exp(5.0*(dist-rad)))

    def get(self):
        # get the coordinates of all pixels in the image. however, make sure
        # that zero starts in the interior of the image where the padding stops
        rvec = self.shape.translate(-self.inner.l).coords(form='vector')

        # add up the images of many particles to get the platonic image
        self.image = np.array([
            self.draw_disc(rvec, i) for i in xrange(self.N)
        ]).sum(axis=0)

        # return the image in the current tile
        return self.image[self.tile.slicer]

    def get_update_tile(self, params, values):
        # for now, if we update a parameter update the entire image
        return self.shape

def initialize():
    N = 32

    # create a NullImage, which means that the model image will be used for data
    img = peri.util.NullImage(shape=(N,N))

    # setup the initial conditions for the parameters of our model
    pos = [[10., 10.], [15., 18.], [8.0, 24.0]]
    rad = [5.0, 3.5, 2.2]
    sig = [2.0, 1.5]

    # make each of the components separately
    d = PlatonicDiscs(positions=pos, radii=rad)
    p = GaussianPSF(sigmas=sig)
    c = GlobalScalar(name='off', value=0.0)

    # join them with the model into a state
    s = peri.states.ImageState(img, [d, p, c], mdl=GaussianDiscModel(), pad=10)
    return s

def show_derivative(s, param, ax=None):
    # if there is no axis supplied, create a new one
    ax = ax or pl.figure().gca()

    # calculate the derivative of the model
    deriv = s.gradmodel(params=[param], flat=False)[0]

    # plot it in a sane manner using matplotlib
    scale = max(np.abs([deriv.min(), deriv.max()]))
    ax.imshow(deriv, vmin=-scale, vmax=scale, cmap='RdBu_r')
    ax.set_title(param)
    ax.set_xticks([])
    ax.set_yticks([])

def show_jtj(s):
    # plot the JTJ with properly labeled axes
    p = s.params
    pl.imshow(np.log10(np.abs(s.JTJ())), cmap='bone')
    pl.xticks(np.arange(len(p)), p, rotation='vertical')
    pl.yticks(np.arange(len(p)), p, rotation='horizontal')
    pl.title(r'$J^T J$')

def rattle_and_fit(s):
    # grab the original values
    values = np.array(s.values).copy()

    # update the model with random parameters then optimize back
    s.update(s.params, values + np.random.randn(len(values)))
    opt.do_levmarq(s, s.params, run_length=12)

    # calculate the crb for all parameters
    crb = s.crb()

    # print a table comparing inferred values
    print(' {:^6s} += {:^5s} | {:^8s}'.format('Fit', 'CRB', 'Actual'))
    print('-'*27)
    for v0, c, v1 in zip(s.values, crb, values):
        print('{:7.3f} += {:4.3f} | {:7.3f}'.format(v0, c, v1))
