import os
import numpy as np
import cPickle as pickle

from cbamf import const
from cbamf import initializers
from cbamf.util import Tile, amin, amax, ProgressBar, RawImage
from cbamf.priors import overlap

class State:
    def __init__(self, nparams, state=None, logpriors=None):
        self.nparams = nparams
        self.state = state if state is not None else np.zeros(self.nparams, dtype='double')
        self.stack = []
        self.logpriors = logpriors

    def _update_state(self, block, data):
        self.state[block] = data.astype(self.state.dtype)
        return True

    # TODO -- should this be a context manager?
    # with s.push_update(b, d):
    #   s.loglikelihood()
    def push_update(self, block, data):
        curr = self.state[block].copy()
        self.stack.append((block, curr))
        self.update(block, data)

    def pop_update(self):
        block, data = self.stack.pop()
        self.update(block, data)

    def update(self, block, data):
        return self._update_state(block, data)

    def block_all(self):
        return np.ones(self.nparams, dtype='bool')

    def block_none(self):
        return np.zeros(self.nparams, dtype='bool')

    def block_range(self, bmin, bmax):
        block = self.block_none()
        bmin = max(bmin, 0)
        bmax = min(bmax, self.nparams)
        block[bmin:bmax] = True
        return block

    def explode(self, block):
        inds = np.arange(block.shape[0])
        inds = inds[block]

        blocks = []
        for i in inds:
            tblock = self.block_none()
            tblock[i] = True
            blocks.append(tblock)
        return blocks

    def _grad_single_param(self, block, dl, action='ll'):
        self.push_update(block, self.state[block]+dl)
        loglr = self.loglikelihood()
        self.pop_update()

        self.push_update(block, self.state[block]-dl)
        logll = self.loglikelihood()
        self.pop_update()

        return (loglr - logll) / (2*dl)

    def _hess_two_param(self, b0, b1, dl):
        self.push_update(b0, self.state[b0]+dl)
        self.push_update(b1, self.state[b1]+dl)
        logl_01 = self.loglikelihood()
        self.pop_update()
        self.pop_update()

        self.push_update(b0, self.state[b0]+dl)
        logl_0 = self.loglikelihood()
        self.pop_update()

        self.push_update(b1, self.state[b1]+dl)
        logl_1 = self.loglikelihood()
        self.pop_update()

        logl = self.loglikelihood()

        return (logl_01 - logl_0 - logl_1 + logl) / (dl**2)

    def loglikelihood(self):
        loglike = self.dologlikelihood()
        if self.logpriors is not None:
            loglike += self.logpriors()
        return loglike

    def gradloglikelihood(self, dl=1e-3, blocks=None, progress=False):
        if blocks is None:
            blocks = self.explode(self.block_all())
        grad = np.zeros(len(blocks))

        p = ProgressBar(len(blocks), display=progress)
        for i, b in enumerate(blocks):
            p.increment()
            grad[i] = self._grad_single_param(b, dl)
        p.end()

        return grad

    def hessloglikelihood(self, dl=1e-3, blocks=None, progress=False, jtj=False):
        if jtj:
            grad = self.gradloglikelihood(dl=dl, blocks=blocks, progress=progress)
            return grad.T[None,:] * grad[:,None]
        else:
            if blocks is None:
                blocks = self.explode(self.block_all())
            hess = np.zeros((len(blocks), len(blocks)))

            p = ProgressBar(len(blocks), display=progress)
            for i, bi in enumerate(blocks):
                p.increment()
                for j, bj in enumerate(blocks[i:]):
                    J = j + i
                    thess = self._hess_two_param(bi, bj, dl)
                    hess[i,J] = thess
                    hess[J,i] = thess
            p.end()
            return hess

    def negloglikelihood(self):
        return -self.loglikelihood()

    def neggradloglikelihood(self):
        return -self.gradloglikelihood()

"""
class JointState(State):
    def __init__(self, states, shared_params):
        self.states = states
        self.shared_params = shared_params

    def update(self, params, values):
        for param in params:
            if param in self.shared_params:
                pass
"""

class LinearFit(State):
    def __init__(self, x, y, sigma=1, *args, **kwargs):
        State.__init__(self, nparams=3, *args, **kwargs)
        self.dx, self.dy = (np.array(i) for i in zip(*sorted(zip(x, y))))

        self.b_sigma = self.explode(self.block_all())[-1]
        self.state[self.b_sigma] = sigma

    def plot(self, state):
        import pylab as pl
        pl.figure()
        pl.plot(self.dx, self.dy, 'o')
        pl.plot(self.dx, self.docalculate(state), '-')
        pl.show()

    def _calculate(self):
        return self.state[0]*self.dx + self.state[1]

    def dologlikelihood(self):
        sig = self.state[self.b_sigma]
        return (-((self._calculate() - self.dy)**2).sum() / (2*sig**2) + 
                -(np.log(abs(sig)) + np.log(np.sqrt(2*np.pi))))

    def reset(self):
        self.state *= 0

def prepare_image(image, imz=(0,None), imsize=None, invert=False, pad=const.PAD, dopad=True):
    image = initializers.normalize(image[imz[0]:imz[1],:imsize,:imsize], invert)
    if dopad:
        image = np.pad(image, pad, mode='constant', constant_values=const.PADVAL)
    return image

def prepare_for_state(image, pos, rad, invert=False, pad=const.PAD, dopad=True,
        remove_overlaps=False):
    """
    Prepares a set of positions, radii, and a test image for use
    in the ConfocalImagePython object

    Parameters:
    -----------
    image : (Nz, Ny, Nx) ndarray
        the raw image from which to feature pos, rad, etc.

    pos : (N,3) ndarray
        list of positions of particles in pixel units, where the positions
        are listed in the same order as the image, (pz, py, px)

    rad : (N,) ndarray | float
        list of particle radii in pixel units if ndarray.  If float, 
        a list of radii of that value will be returned

    invert : boolean (optional)
        True indicates that the raw image is light particles on dark background,
        and needs to be inverted, otherwise, False

    pad : integer (optional)
        The amount of padding to add to the raw image, should be at least
        2 times the PSF size.  Not recommended to set manually

    remove_overlaps : boolean
        whether to remove overlaps from the pos, rad given.  not recommended,
        has bugs
    """
    # normalize and pad the image, add the same offset to the positions
    image = initializers.normalize(image, invert)
    if dopad:
        image = np.pad(image, pad, mode='constant', constant_values=const.PADVAL)
        pos = pos.copy() + pad

    if not isinstance(rad, np.ndarray):
        rad = rad*np.ones(pos.shape[0])

    bound_left = np.zeros(3)
    bound_right = np.array(image.shape)

    # clip particles that are outside of the image or have negative radii
    keeps = np.ones(rad.shape[0]).astype('bool')
    for i, (p, r) in enumerate(zip(pos, rad)):
        if (p < bound_left).any() or (p > bound_right).any():
            print "Particle %i out of bounds at %r, removing..." % (i, p)
            keeps[i] = False
        if r < 0:
            print "Particle %i with negative radius %f, removing..." % (i, r)
            keeps[i] = False

    pos = pos[keeps]
    rad = rad[keeps]

    # change overlapping particles so that the radii do not coincide
    if remove_overlaps:
        initializers.remove_overlaps(pos, rad)
    return image, pos, rad


class ConfocalImagePython(State):
    def __init__(self, image, obj, psf, ilm, zscale=1, offset=1,
            sigma=0.04, doprior=False, constoff=True,
            varyn=False, allowdimers=False, nlogs=True, difference=True,
            pad=const.PAD, sigmapad=True, slab=None, *args, **kwargs):
        """
        The state object to create a confocal image.  The model is that of
        a spatially varying illumination field, from which platonic particle
        shapes are subtracted.  This is then spread with a point spread function
        (PSF).  Summarized as follows:

            Image = \int PSF(x-x') (ILM(x)*(1-SPH(x))) dx'

        Parameters:
        -----------
        image : (Nz, Ny, Nx) ndarray OR `cbamf.util.RawImage` object
            The raw image with which to compare the model image from this class.
            This image should have been prepared through prepare_for_state, which
            does things such as padding necessary for this class. In the case of the
            RawImage, paths are used to keep track of the image object to save
            on pickle size.

        obj : component
            A component object which handles the platonic image creation, e.g., 
            cbamf.comp.objs.SphereCollectionRealSpace.  Also, needs to be created
            after prepare_for_state.

        psf : component
            The PSF component which has the same image size as padded image.

        ilm : component
            Illumination field component from cbamf.comp.ilms

        zscale : float, typically (1, 1.5) [default: 1]
            The initial zscaling for the pixel sizes.  Bigger is more compressed.

        offset : float, typically (0, 1) [default: 1]
            The level that particles inset into the illumination field

        doprior: boolean [default: True]
            Whether or not to turn on overlap priors using neighborlists

        constoff: boolean [default: True]
            Changes the model so to:

                Image = \int PSF(x-x') (ILM(x)*-OFF*SPH(x)) dx'

        varyn: boolean [default: False]
            allow the variation of particle number (only recommended in that case)

        allowdimers: boolean [default: False]
            allow dimers to be created out of two separate overlapping particles

        nlogs: boolean [default: False]
            Include in the Loglikelihood calculate the term:

                LL = -(p_i - I_i)^2/(2*\sigma^2) - \log{\sqrt{2\pi} \sigma} 

        difference : boolean [default: True]
            To only modify difference images (thanks to linear FTs).  Set True by
            default because ~8x faster.

        pad : integer (optional)
            No recommended to set by hand.  The padding level of the raw image needed
            by the PSF support.

        sigmapad : boolean [default: True]
            If True, varies the sigma values at the edge of the image, changing them
            slowly to zero over the size of the psf support

        slab : `cbamf.comp.objs.Slab` [default: None]
            If not None, include a slab in the model image and associated analysis.
            This object should be from the platonic components module.
        """
        self.pad = pad
        self.index = None
        self.sigma = sigma
        self.doprior = doprior
        self.constoff = constoff
        self.allowdimers = allowdimers
        self.nlogs = nlogs
        self.varyn = varyn
        self.difference = difference
        self.sigmapad = sigmapad

        self.psf = psf
        self.ilm = ilm
        self.obj = obj
        self.slab = slab
        self.zscale = zscale
        self.rscale = 1.0
        self.offset = offset
        self.N = self.obj.N

        self._build_state()
        self.set_image(image)

    def reset(self):
        self._build_state()
        if self.rawimage is not None:
            self.set_image(self.rawimage)
        else:
            self.set_image(self.padded_image())

    def set_obj(self, obj):
        self.obj = obj
        self.reset()

    def set_psf(self, psf):
        self.psf = psf
        self.reset()

    def set_ilm(self, ilm):
        self.ilm = ilm
        self.reset()

    def set_pos_rad(self, pos, rad):
        self.obj.set_pos_rad(pos, rad)
        self.reset()

    def padded_image(self):
        o = self.image.copy()
        o[self.image_mask == 0] = const.PADVAL
        return o

    def set_image(self, image):
        """
        Update the current comparison (real) image
        """
        if isinstance(image, RawImage):
            self.rawimage = image
            image = image.get_padded_image(self.pad)
        else:
            self.rawimage = None

        self.image = image.copy()
        self.image_mask = (image > const.PADVAL).astype('float')
        self.image *= self.image_mask

        self.inner = (np.s_[self.pad:-self.pad],)*3
        self._build_internal_variables()
        self._initialize()

    def model_to_true_image(self):
        """
        In the case of generating fake data, use this method to add
        noise to the created image (only for fake data) and rotate
        the model image into the true image
        """
        im = self.get_model_image()
        im = im + self.image_mask * np.random.normal(0, self.sigma, size=self.image.shape)
        im = im + (1 - self.image_mask) * const.PADVAL
        self.set_image(im)

    def get_model_image(self):
        return self.model_image * self.image_mask

    def get_true_image(self):
        return self.image * self.image_mask

    def get_difference_image(self, doslice=True):
        o = self.get_true_image() - self.get_model_image()
        if doslice:
            return o[self.inner]
        return o

    def _build_state(self):
        self.param_dict = {
            'pos': 3*self.obj.N,
            'rad': self.obj.N,
            'typ': self.obj.N*self.varyn,
            'psf': len(self.psf.get_params()),
            'ilm': len(self.ilm.get_params()),
            'slab': len(self.slab.get_params()) if self.slab else 0,
            'off': 1,
            'rscale': 1,
            'zscale': 1,
            'sigma': 1,
        }

        self.param_order = [
            'pos', 'rad', 'typ', 'psf', 'ilm', 'off', 'slab',
            'rscale', 'zscale', 'sigma'
        ]
        self.param_lengths = [self.param_dict[k] for k in self.param_order]

        out = []
        for param in self.param_order:
            if self.param_dict[param] == 0:
                continue

            if param == 'pos':
                out.append(self.obj.get_params_pos())
            if param == 'rad':
                out.append(self.obj.get_params_rad())
            if param == 'typ':
                out.append(self.obj.get_params_typ())
            if param == 'psf':
                out.append(self.psf.get_params())
            if param == 'ilm':
                out.append(self.ilm.get_params())
            if param == 'slab':
                out.append(self.slab.get_params())
            if param == 'off':
                out.append(self.offset)
            if param == 'rscale':
                out.append(self.rscale)
            if param == 'zscale':
                out.append(self.zscale)
            if param == 'sigma':
                out.append(self.sigma)

        self.nparams = sum(self.param_lengths)
        State.__init__(self, nparams=self.nparams)

        self.state = np.hstack(out).astype('float')

        self.b_pos = self.create_block('pos')
        self.b_rad = self.create_block('rad')
        self.b_typ = self.create_block('typ')
        self.b_psf = self.create_block('psf')
        self.b_ilm = self.create_block('ilm')
        self.b_off = self.create_block('off')
        self.b_slab = self.create_block('slab')
        self.b_rscale = self.create_block('rscale')
        self.b_zscale = self.create_block('zscale')
        self.b_sigma = self.create_block('sigma')

    def _build_internal_variables(self):
        self.model_image = np.zeros_like(self.image)

        self._logprior = 0
        self._loglikelihood = 0
        self._loglikelihood_field = np.zeros(self.image.shape)
        self._build_sigma_field()

    def _build_sigma_field(self):
        self._sigma_field = np.zeros(self.image.shape)

        if not self.sigmapad:
            self._sigma_field += self.sigma
        else:
            top = self.image.shape[0] - self.pad

            p = self.psf.get_support_size(top)/4
            l = self.pad + p
            self._sigma_field[:] = 3*self.sigma #FIXME -- this should be 2 ??
            self._sigma_field[l[0]:-l[0], l[1]:-l[1], l[2]:-l[2]] = self.sigma

            self.psf.set_tile(Tile(self._sigma_field.shape))
            self._sigma_field = self.psf.execute(self._sigma_field)

        self._sigma_field_log = np.log(self._sigma_field)#*(self._sigma_field > 0) + 1e-16)

    def set_state(self, state):
        self.obj.pos = state[self.b_pos].reshape(-1,3)
        self.obj.rad = state[self.b_rad]
        self.ilm.params = state[self.b_ilm]
        self.psf.params = state[self.b_psf]
        self._initialize()

    def _initialize(self):
        if self.doprior:
            bounds = (np.array([0,0,0]), np.array(self.image.shape))
            self.nbl = overlap.HardSphereOverlapCell(self.obj.pos, self.obj.rad, self.obj.typ,
                    zscale=self.zscale, bounds=bounds, cutoff=2.2*self.obj.rad.max())
            self._logprior = self.nbl.logprior() + const.ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

        self.psf.update(self.state[self.b_psf])
        self.obj.initialize(self.zscale)
        self.ilm.initialize()

        if self.slab:
            self.slab.initialize()

        self._update_global()

    def _tile_from_particle_change(self, p0, r0, t0, p1, r1, t1):
        pref = 1 if self.difference else 2
        extr = 1 if self.difference else 0

        sl, sr = self.obj.get_support_size(p0, r0, t0, p1, r1, t1, self.zscale)

        tl = self.psf.get_support_size(sl)
        tr = self.psf.get_support_size(sr)
        t_xy = np.max([tl, tr], axis=0)[1:]
        tl[1:] = t_xy
        tr[1:] = t_xy

        pl = sl - pref*tl - extr
        pr = sr + pref*tr + extr
        pl = np.floor(pl).astype('int')
        pr = np.ceil (pr).astype('int')

        pl += pl % 2
        pr += pr % 2
        outer = Tile(pl, pr, 0, self.image.shape)

        if self.difference:
            inner = Tile(pl+extr, pr-extr, extr, np.array(self.image.shape)-extr)
            ioslice = tuple([np.s_[extr:-extr] for i in xrange(3)])
        else:
            ipl = np.ceil(self.psf.get_support_size(sl)).astype('int')
            ipr = np.ceil(self.psf.get_support_size(sr)).astype('int')

            inner = Tile(pl+ipl, pr-ipr, ipl, np.array(self.image.shape)-ipr)
            ioslice = tuple([np.s_[ipsc[i]:-ipsc[i]] for i in xrange(3)])

        return outer, inner, ioslice

    def _tile_global(self):
        outer = Tile(0, self.image.shape)
        inner = Tile(1, np.array(self.image.shape)-1)
        ioslice = (np.s_[1:-1],)*3
        return outer, inner, ioslice

    def _update_ll_field(self, data=None, slicer=np.s_[:]):
        if data is None:
            self._loglikelihood = 0
            self._loglikelihood_field *= 0
            data = self.get_model_image()

        s = slicer
        oldll = self._loglikelihood_field[s].sum()

        # make temporary variables since slicing is 'hard'
        im = self.image[s]
        sig = self._sigma_field[s]
        lsig = self._sigma_field_log[s]
        tmask = self.image_mask[s]

        self._loglikelihood_field[s] = (
                -tmask * (data - im)**2 / (2*sig**2)
                -tmask * (lsig + np.log(np.sqrt(2*np.pi)))*self.nlogs
            )

        newll = self._loglikelihood_field[s].sum()
        self._loglikelihood += newll - oldll

    def _update_global(self):
        self._update_tile(*self._tile_global(), difference=False)

    def _update_tile(self, otile, itile, ioslice, difference=False):
        self._last_slices = (otile, itile, ioslice)

        self.obj.set_tile(otile)
        self.ilm.set_tile(otile)
        self.psf.set_tile(otile)

        if self.slab:
            self.slab.set_tile(otile)

        islice = itile.slicer

        if difference:
            platonic = self.obj.get_diff_field()

            if self.constoff:
                replacement = self.offset*platonic
            else:
                replacement = self.ilm.get_field() * self.offset*platonic
        else:
            platonic = self.obj.get_field()

            if self.slab:
                # cannot be += since platonic is linked to self.obj.field
                # FIXME -- should be copy operator above?
                platonic = platonic + self.slab.get_field()

            if self.allowdimers:
                platonic = np.clip(platonic, -1, 1)

            if self.constoff:
                replacement = self.ilm.get_field() - self.offset*platonic
            else:
                replacement = self.ilm.get_field() * (1 - self.offset*platonic)

        replacement = self.psf.execute(replacement)

        if difference:
            self.model_image[islice] -= replacement[ioslice]
        else:
            self.model_image[islice] = replacement[ioslice]

        self._update_ll_field(self.model_image[islice], islice)

    def update(self, block, data):
        prev = self.state.copy()

        # TODO, instead, push the change in case we need to pop it
        self._update_state(block, data)

        pmask = block[self.b_pos].reshape(-1, 3).any(axis=-1)
        rmask = block[self.b_rad]
        tmask = block[self.b_typ] if self.varyn else (0*rmask).astype('bool')

        particles = np.arange(self.obj.N)[pmask | rmask | tmask]

        self._logprior = 0

        # FIXME -- obj.create_diff_field not guaranteed to work for multiple particles
        # if the particle was changed, update locally
        if len(particles) > 0:
            pos0 = prev[self.b_pos].copy().reshape(-1,3)[particles]
            rad0 = prev[self.b_rad].copy()[particles]

            pos = self.state[self.b_pos].copy().reshape(-1,3)[particles]
            rad = self.state[self.b_rad].copy()[particles]

            if self.varyn:
                typ0 = prev[self.b_typ].copy()[particles]
                typ = self.state[self.b_typ].copy()[particles]
            else:
                typ0 = np.ones(len(particles))
                typ = np.ones(len(particles))

            # Do a bunch of checks to make sure that we can safetly modify
            # the image since that is costly and we would reject
            # this state eventually otherwise
            if (pos < 0).any() or (pos > np.array(self.image.shape)).any():
                self.state[block] = prev[block]
                return False

            tiles = self._tile_from_particle_change(pos0, rad0, typ0, pos, rad, typ)
            #for tile in tiles[:2]:
            #    top = self.image.shape[0] - self.pad
            #    if (np.array(tile.shape) < 2*self.psf.get_support_size(top)).any():
            #        self.state[block] = prev[block]
            #        return False

            if self.doprior:
                self.nbl.update(particles, pos, rad, typ)
                self._logprior = self.nbl.logprior() + const.ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

                if self._logprior < const.PRIORCUT:
                    self.nbl.update(particles, pos0, rad0, typ0)
                    self._logprior = self.nbl.logprior() + const.ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

                    self.state[block] = prev[block]
                    return False

            if (typ0 == 0).all() and (typ == 0).all():
                return False

            # Finally, modify the image
            self.obj.update(particles, pos, rad, typ, self.zscale, difference=self.difference)
            self._update_tile(*tiles, difference=self.difference)
        else:
            docalc = False

            if self.slab and block[self.b_slab].any():
                self.slab.update(self.state[self.b_slab])
                docalc = True

            # if the psf was changed, update globally
            if block[self.b_psf].any():
                self.psf.update(self.state[self.b_psf])
                self._build_sigma_field()
                docalc = True

            # update the background if it has been changed
            if block[self.b_ilm].any():
                self.ilm.update(block[self.b_ilm], self.state[self.b_ilm])
                docalc = True

            # we actually don't have to do anything if the offset is changed
            if block[self.b_off].any():
                self.offset = self.state[self.b_off]
                docalc = True

            if block[self.b_sigma].any():
                self.sigma = self.state[self.b_sigma]

                if self.sigma <= 0:
                    self.state[block] = prev[block]
                    self.sigma = self.state[self.b_sigma]
                    return False

                self._build_sigma_field()
                self._update_ll_field()

            if block[self.b_zscale].any():
                self.zscale = self.state[self.b_zscale][0]

                if self.doprior:
                    bounds = (np.array([0,0,0]), np.array(self.image.shape))
                    tnbl = overlap.HardSphereOverlapCell(self.obj.pos, self.obj.rad, self.obj.typ,
                            zscale=self.zscale, bounds=bounds, cutoff=2.2*self.obj.rad.max())

                    if tnbl.logprior() < const.PRIORCUT:
                        self.state[block] = prev[block]
                        return False

                    self.nbl = tnbl
                    self._logprior = self.nbl.logprior() + const.ZEROLOGPRIOR*(self.state[self.b_rad] < 0).any()

                self.obj.initialize(self.zscale)
                self._update_global()

            if block[self.b_rscale].any():
                new_rscale = self.state[self.b_rscale][0]
                f = new_rscale / self.rscale

                self.obj.rad *= f
                self.obj.initialize(self.zscale)
                self._update_global()

                self.rscale = new_rscale

            if docalc:
                self._update_global()

        return True

    def add_particle(self, p, r):
        n = self.obj.typ.argmin()

        bp = self.block_particle_pos(n)
        br = self.block_particle_rad(n)
        bt = self.block_particle_typ(n)

        self.update(bp, p)
        self.update(br, np.array([r]))
        self.update(bt, np.array([1]))

        return n

    def remove_particle(self, n):
        bt = self.block_particle_typ(n)
        self.update(bt, np.array([0]))
        return self.obj.pos[n], self.obj.rad[n]

    def closest_particle(self, x):
        return ((self.obj.typ==0)*1e5 + ((self.obj.pos - x)**2).sum(axis=-1)).argmin()

    def remove_closest_particle(self, x):
        return self.remove_particle(self.closest_particle(x))

    def isactive(self, particle):
        if self.varyn:
            return self.state[self.block_particle_typ(particle)] == 1
        return True

    def active_particles(self):
        if self.varyn:
            return np.arange(self.N)[self.state[self.b_typ]==1.]
        return np.arange(self.N)

    def blocks_particle(self, index):
        p_ind, r_ind = 3*index, 3*self.obj.N + index

        blocks = []
        for t in xrange(p_ind, p_ind+3):
            blocks.append(self.block_range(t, t+1))
        blocks.append(self.block_range(r_ind, r_ind+1))
        return blocks

    def block_particle_pos(self, index):
        a = self.block_none()
        a[3*index:3*index+3] = True
        return a

    def block_particle_rad(self, index):
        a = self.block_none()
        a[3*self.obj.N + index] = True
        return a

    def block_particle_typ(self, index):
        a = self.block_none()
        a[4*self.obj.N + index] = True
        return a

    def create_block(self, typ='all'):
        return self.block_range(*self._block_offset_end(typ))

    def _block_offset_end(self, typ='pos'):
        index = self.param_order.index(typ)
        off = sum(self.param_lengths[:index])
        end = off + self.param_lengths[index]
        return off, end

    def _grad_image(self, bl, dl=1e-3):
        self.push_update(bl, self.state[bl]+dl)
        m1 = self.get_model_image()
        self.pop_update()

        self.push_update(bl, self.state[bl]-dl)
        m0 = self.get_model_image()
        self.pop_update()

        return (m1 - m0) / (2*dl)

    def _grad_residuals(self, bl, dl=1e-3):
        self.push_update(bl, self.state[bl]+dl)
        m1 = self.residuals()
        self.pop_update()

        self.push_update(bl, self.state[bl]-dl)
        m0 = self.residuals()
        self.pop_update()

        return (m1 - m0) / (2*dl)

    def residuals(self, masked=False):
        if not masked:
            return self.image_mask*(self.image - self.get_model_image())
        return (self.image - self.get_model_image())[self.image_mask == 1.0]

    def fisher_information(self, blocks=None, dl=1e-3):
        if blocks is None:
            blocks = self.explode(self.block_all())
        fish = np.zeros((len(blocks), len(blocks)))

        for i, bi in enumerate(blocks):
            for j, bj in enumerate(blocks[i:]):
                J = j + i
                di = self._grad_image(bi, dl=dl)
                dj = self._grad_image(bj, dl=dl)
                tfish = (di*dj).sum()
                fish[i,J] = tfish
                fish[J,i] = tfish

        return fish / self.sigma**2

    def jac(self, blocks=None, dl=1e-3, maxmem=1e9, flat=True):
        if blocks is None:
            blocks = self.explode(self.block_all())

        if self._loglikelihood_field.nbytes * len(blocks) > maxmem:
            raise AttributeError("Maximum memory would be violated" +
                    ", please select fewer blocks")

        J = np.zeros((len(blocks), self._loglikelihood_field.size))

        for i, bi in enumerate(blocks):
            tJ = self._grad_residuals(bi, dl=dl).flatten()
            if flat:
                J[i] = tJ.flatten()
            else:
                J[i] = tJ
        return J

    def jtj(self, blocks=None, dl=1e-3, maxmem=1e9):
        J = self.jac(blocks=blocks, dl=dl, maxmem=maxmem)
        return J.dot(J.T)

    def loglikelihood(self):
        return self._logprior + self._loglikelihood

    def todict(self, samples=None):
        """
        Transform ourselves (state) or a state vector into a dictionary for
        better storage options.  In the form {block_name: parameters}.

        Essentially the output of this function is of the form
        {b:samples[...,self.create_block(b)] for b in self.param_order}
        but we need to special handle some cases, so `for` loop
        """
        local = False
        if samples is None:
            samples = self.state.copy()
            local = True

        active = tuple(self.active_particles())
        nactive = len(active)

        out = {}
        for b in self.param_order:
            d = samples[...,self.create_block(b)]

            # reshape positions properly
            if b == 'pos':
                if not local:
                    d = d.reshape(-1,self.N,3)
                else:
                    d = d.reshape(self.N, 3)
                d = d[...,active,:]

            if b == 'rad':
                d = d[...,active]

            # transform scalars to lists
            if b in ['off', 'rscale', 'zscale', 'sigma']:
                if local:
                    d = np.array([d])
                d = np.squeeze(d)

            out[b] = d
        return out

    def __getstate__(self):
        return {}

    def __setstate__(self, idct):
        pass

    def __getinitargs__(self):
        # FIXME -- unify interface for RawImage
        if self.rawimage is not None:
            im = self.rawimage
        else:
            im = self.padded_image()

        return (im,
            self.obj, self.psf, self.ilm, self.zscale, self.offset,
            self.sigma, self.doprior, self.constoff,
            self.varyn, self.allowdimers, self.nlogs, self.difference,
            self.pad, self.sigmapad, self.slab)

def save(state, filename=None, desc='', extra=None):
    """
    Save the current state with extra information (for example samples and LL
    from the optimization procedure).

    state : cbamf.states.ConfocalImagePython
        the state object which to save

    filename : string
        if provided, will override the default that is constructed based on
        the state's raw image file.  If there is no filename and the state has
        a RawImage, the it is saved to RawImage.filename + "-peri-save.pkl"

    desc : string
        if provided, will augment the default filename to be 
        RawImage.filename + '-peri-' + desc + '.pkl'

    extra : list of pickleable objects
        if provided, will be saved with the state
    """
    if state.rawimage is not None:
        desc = desc or 'save'
        filename = filename or state.rawimage.filename + '-peri-' + desc + '.pkl'
    else:
        if not filename:
            raise AttributeError, "Must provide filename since RawImage is not used"

    if extra is None:
        save = state
    else:
        save = [state] + extra

    pickle.dump(save, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))
