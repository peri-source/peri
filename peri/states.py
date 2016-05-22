import os
import numpy as np
import cPickle as pickle
from contextlib import contextmanager

from peri import const, util
from peri.comp import Component

class State(Component):
    def __init__(self, params, values, logpriors=None):
        self.stack = []
        self.logpriors = logpriors

    @property
    def data(self):
        pass

    @property
    def model(self):
        pass

    @property
    def residual(self):
        pass

    def residuals_sample(self, inds):
        return residual.ravel[inds]

    def loglikelihood(self):
        loglike = self.dologlikelihood()
        if self.logpriors is not None:
            loglike += self.logpriors()
        return loglike

    def update(self, params, values):
        super(State, self).update(params, values)

    def push_update(self, params, values):
        curr = self.get_values(params)
        self.stack.append((params, curr))
        self.update(params, values)

    def pop_update(self):
        params, values = self.stack.pop()
        self.update(params, values)

    @contextmanager
    def temp_update(self, params, values):
        self.push_update(params, values)
        yield
        self.pop_update()

    def block_all(self):
        return self.params

    def _grad_one_param(self, func, p, dl=1e-3, f0=None, rts=True, **kwargs):
        vals = self.get_values(p)
        f0 = util.callif(func(**kwargs)) if f0 is None else f0

        self.update(p, vals+dl)
        f1 = util.callif(func(**kwargs))

        if rts:
            self.update(p, vals)

        return (f1 - f0) / dl

    def _hess_two_param(self, func, p0, p1, dl=1e-3, f0=None, rts=True, **kwargs):
        vals0 = self.get_values(p0)
        vals1 = self.get_values(p1)

        f00 = util.callif(func(**kwargs)) if f0 is None else f0

        self.update(p0, vals0+dl)
        f10 = util.callif(func(**kwargs))

        self.update(p1, vals1+dl)
        f11 = util.callif(func(**kwargs))

        self.update(p0, vals0)
        f01 = util.callif(func(**kwargs))

        if rts:
            self.update(p0, vals0)
            self.update(p1, vals1)

        return (f11 - f10 - f01 + f00) / (dl**2)

    def _grad(self, func, ps=None, dl=1e-3, **kwargs):
        if ps is None:
            ps = self.block_all()

        ps = listifty(ps)
        f0 = util.callif(func(**kwargs))

        grad = []
        for i, p in enumerate(ps):
            grad.append(self._grad_one_param(func, p, dl, f0=f0, **kwargs))
        return np.array(grad)

    def _jtj(self, func, ps=None, dl=1e-3, **kwargs):
        grad = self._grad(func=func, ps=ps, dl=dl, **kwargs)
        return np.dot(grad.T, grad)

    def _hess(self, func, ps=None, dl=1e-3, **kwargs):
        if ps is None:
            ps = self.block_all()

        ps = listifty(ps)
        f0 = util.callif(func(**kwargs))

        hess = [[0]*len(ps)]*len(ps)
        for i, pi in enumerate(ps):
            for j, pj in enumerate(ps[i:]):
                J = j + i
                thess = self._hess_two_param(func, bi, bj, dl=dl, f0=f0, **kwargs)
                hess[i,J] = thess
                hess[J,i] = thess
        return np.array(hess)

    gradloglikelihood = partial(_grad, func=self.loglikelihood)
    hessloglikelihood = partial(_hess, func=self.loglikelihood)
    J = partial(_grad, func=self.residuals)
    JTJ = partial(_jtj, func=self.residuals)
    Jp = partial(_grad, func=self.residuals_sample)
    JTJp = partial(_jtj, func=self.residuals_sample)

class ConfocalImagePython(State):
    def __init__(self, image, obj, psf, ilm, zscale=1, offset=0,
            sigma=0.04, doprior=False, constoff=False,
            varyn=False, nlogs=True, difference=True,
            pad=const.PAD, slab=None, newconst=True, bkg=None,
            method=1, *args, **kwargs):
        """
        The state object to create a confocal image.  The model is that of
        a spatially varying illumination field, from which platonic particle
        shapes are subtracted.  This is then spread with a point spread function
        (PSF).  Summarized as follows:

            Image = \int PSF(x-x') (ILM(x)*(1-SPH(x))) dx'

        Parameters:
        -----------
        image : (Nz, Ny, Nx) ndarray OR `peri.util.RawImage` object
            The raw image with which to compare the model image from this class.
            This image should have been prepared through prepare_for_state, which
            does things such as padding necessary for this class. In the case of the
            RawImage, paths are used to keep track of the image object to save
            on pickle size.

        obj : component
            A component object which handles the platonic image creation, e.g., 
            peri.comp.objs.SphereCollectionRealSpace.  Also, needs to be created
            after prepare_for_state.

        psf : component
            The PSF component which has the same image size as padded image.

        ilm : component
            Illumination field component from peri.comp.ilms

        zscale : float, typically (1, 1.5) [default: 1]
            The initial zscaling for the pixel sizes.  Bigger is more compressed.

        offset : float, typically (0, 1) [default: 1]
            The level that particles inset into the illumination field

        doprior: boolean [default: False]
            Whether or not to turn on overlap priors using neighborlists

        constoff: boolean [default: False]
            Changes the model so to:

                Image = \int PSF(x-x') (ILM(x)*-OFF*SPH(x)) dx'

        varyn: boolean [default: False]
            allow the variation of particle number (only recommended in that case)

        nlogs: boolean [default: False]
            Include in the Loglikelihood calculate the term:

                LL = -(p_i - I_i)^2/(2*\sigma^2) - \log{\sqrt{2\pi} \sigma} 

        difference : boolean [default: True]
            To only modify difference images (thanks to linear FTs).  Set True by
            default because ~8x faster.

        pad : integer (optional)
            No recommended to set by hand.  The padding level of the raw image needed
            by the PSF support.

        slab : `peri.comp.objs.Slab` [default: None]
            If not None, include a slab in the model image and associated analysis.
            This object should be from the platonic components module.

        newconst : boolean [default: True]
            If True, overrides constoff to implement a image formation of the form:

                Image = \int PSF(x-x') (ILM(x)*(1-SPH(X)) + OFF*SPH(x)) dx'
                      = \int PSF(x-x') (ILM(x) + (OFF-ILM(x))*SPH(x)) dx'

        bkg : `peri.comp.ilms.*`
            a polynomial that represents the background field in dark parts of
            the image

        method : int

            a int that describes the model that we wish to employ (how to
            combine the components into a model image). Given the following
            symbols

                I=ILM, B=BKG, c=OFFSET, P=platonic, p=particles, s=slab, H=PSF

            then the following ints correspond to the listed equations (x is
            convolution)

                1 : [I(1-P)]xH + B
                2 : [I(1-p)]xH + sxH + B
                3 : I[(1-p)xH - sxH] + B
        """
        self.pad = pad
        self.index = None
        self.sigma = sigma
        self.doprior = doprior
        self.constoff = constoff
        self.nlogs = nlogs
        self.varyn = varyn
        self.difference = difference
        self.newconst = newconst
        self.method = method

        self.dollupdate = True

        self.psf = psf
        self.ilm = ilm
        self.bkg = bkg
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

    def set_bkg(self, bkg):
        self.bkg = bkg
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
        if isinstance(image, util.RawImage):
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
        self.N = self.obj.N

        self.param_dict = {
            'pos': 3*self.obj.N,
            'rad': self.obj.N,
            'typ': self.obj.N*self.varyn,
            'psf': len(self.psf.get_params()),
            'ilm': len(self.ilm.get_params()),
            'bkg': len(self.bkg.get_params()) if self.bkg else 0,
            'slab': len(self.slab.get_params()) if self.slab else 0,
            'off': 1,
            'rscale': 1,
            'zscale': 1,
            'sigma': 1,
        }

        self.param_order = [
            'pos', 'rad', 'typ', 'psf', 'ilm', 'bkg', 'off', 'slab',
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
            if param == 'bkg':
                out.append(self.bkg.get_params())
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
        self.b_bkg = self.create_block('bkg')
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
        self._sigma_field += self.sigma
        self._sigma_field_log = np.log(self._sigma_field)

    def _tile_from_particle_change(self, p0, r0, t0, p1, r1, t1):
        # FIXME -- self.difference is ignored in this function

        # FIXME -- this should really be called get_update_tile ??
        # so that psf really is support size since it does not have translation
        sl, sr = self.obj.get_support_size(p0, r0, t0, p1, r1, t1, self.zscale)

        tl = self.psf.get_support_size(sl)
        tr = self.psf.get_support_size(sr)

        # FIXME -- more ceil functions?
        # get the object's tile and the psf tile size
        otile = util.Tile(sl, sr, 0, self.image.shape)
        ptile = util.Tile.boundingtile([util.Tile(np.ceil(i)) for i in [tl, tr]])

        # now remove the part of the tile that is outside the image and
        # pad the interior part with that overhang
        img = util.Tile(self.image.shape)

        # reflect the necessary padding back into the image itself for
        # the outer slice which we will call outer
        outer = otile.pad(ptile.shape/2+1)
        inner, outer = outer.reflect_overhang(img)
        iotile = inner.translate(-outer.l)

        return outer, inner, iotile.slicer

    def _tile_global(self):
        outer = util.Tile(0, self.image.shape)
        inner = util.Tile(0, self.image.shape)
        iotile = inner.translate(outer.l)
        return outer, inner, iotile.slicer

    def _update_ll_field(self, data=None, slicer=np.s_[:]):
        if not self.dollupdate:
            return

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
        """
        Actually calculates the model in a section of image given by the slice.
        If using difference images, then that is usually for platonic update
        so calculate replacement differences
        """
        self._last_slices = (otile, itile, ioslice)
        self._set_tiles(otile)
        islice = itile.slicer

        # unpack a few variables to that this is easier to read, nice compact
        # formulas coming up, B = bkg, I = ilm, C = off
        B = self.bkg.get_field() if self.bkg else None
        I = self.ilm.get_field()
        C = self.offset

        if not difference:
            # unpack one more variable, the platonic image (with slab)
            P = self._platonic_image()

            # Section QUX -- notes on this section:
            #   * for a formula to be correct it must have a term I(1-P)
            #   * sometimes C and B are both used to improve compatibility
            #   * terms B*P are to improve convergence times since it makes them
            #       independent of I(1-P) terms
            if self.method == 1:
                # [I(1-P)]xH + B
                replacement = I*(1-P) + C*P
            elif self.method == 2:
                # [I(1-p)]xH + sxH + B
                pass
            elif self.method == 3:
                # I[(1-p)xH - sxH] + B
                pass
            elif self.newconst and self.bkg is None:
                # 3. the first correct formula, but which sets the illumation to
                # zero where there are particles and adds a constant there
                replacement = I*(1-P) + C*P
            elif self.bkg and self.newconst:
                # 5. the correct formula with a background field. C, while degenerate,
                # is kept around so that other rewrites are unnecessary.
                replacement = I*(1-P) + (C+B)*P

            # TODO -- these last three are incorrect, remove them
            elif self.bkg and not self.newconst:
                # 4. the first bkg formula, but based on 1,2 so incorrect
                replacement = I*(1-C*P) + B*P
            elif self.constoff:
                # 2. the second formula, but still has P dependent on I
                replacement = I - C*P
            else:
                # 1. the original formula, but has P dependent on I since it takes out
                # a fraction of the original illumination, not all of it.
                replacement = I*(1-C*P)

            replacement = self.psf.execute(replacement)

            if self.method == 1:
                self.model_image[islice] = replacement[ioslice] + B[ioslice]
            else:
                self.model_image[islice] = replacement[ioslice]
        else:
            # this section is currently only run for particles
            # unpack one more variable, the change in the platonic image
            dP = self.obj.get_diff_field()

            # each of the following formulas are derived from the section marked
            # QUX (search for it), taking the variation with respect to P, for
            # example:
            #       M = I*(1-P) + C*P
            #      dM = (C-I)dP
            if self.method == 1:
                replacement = (C-I)*dP
            elif self.newconst and self.bkg is None:
                replacement = (C-I)*dP
            elif self.bkg and self.newconst:
                replacement = (C+B-I)*dP
            elif self.bkg and not self.newconst:
                replacement = (B-C*I)*dP
            elif self.constoff:
                replacement = C*dP
            else:
                replacement = -C*I*dP

            # FIXME -- if we move to other local updates, need to fix the last
            # section here with self.method switches, also with dP changing so
            # lolz at that
            replacement = self.psf.execute(replacement)
            self.model_image[islice] += replacement[ioslice]

        self._update_ll_field(self.model_image[islice], islice)

    def _block_to_particles(self, block):
        """ Get the particles affected by block `block` by index """
        pmask = block[self.b_pos].reshape(-1, 3).any(axis=-1)
        rmask = block[self.b_rad]
        tmask = block[self.b_typ] if self.varyn else (0*rmask).astype('bool')
        particles = np.arange(self.obj.N)[pmask | rmask | tmask]
        return particles

    def update_many(self, blocks, data):
        """
        Update many blocks at once by updating locally each parameter then at
        the very end ask for the convolution of the components.

        Parameters
        ----------
        blocks : list, ndarray
            if list, explodes into a list of ndarrays representing the blocks

        data : ndarray
            each element corresponding to new values for the blocks
        """
        otiles, itiles = [], []

        if not isinstance(blocks, list):
            bl = s.explode(blocks)

        for b, d in zip(blocks, data):
            otile, itile, _ = self.update(b, d, update_model=False, return_tiles=True)
            otiles.append(otile)
            itiles.append(itile)

        otile = util.Tile.boundingtile(otiles)
        itile = util.Tile.boundingtile(itiles)
        iotile = itile.translate(-otile.l)

        self._update_tile(otile, itile, iotile.slicer, difference=True)

    def update(self, block, data, update_model=True, return_tiles=False):
        if block.sum() > 1:
            raise AttributeError("Currently we only support 1 variable updates now")

        prev = self.state.copy()

        # TODO, instead, push the change in case we need to pop it
        self._update_state(block, data)
        self._logprior = 0

        particles = self._block_to_particles(block)

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

            if update_model:
                self._update_tile(*tiles, difference=self.difference)
            if return_tiles:
                return tiles
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

            # update the background if it has been changed
            if block[self.b_bkg].any():
                self.bkg.update(block[self.b_bkg], self.state[self.b_bkg])
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
                if update_model:
                    self._update_global()

            if block[self.b_rscale].any():
                new_rscale = self.state[self.b_rscale][0]
                f = new_rscale / self.rscale

                self.obj.rad *= f
                self.obj.initialize(self.zscale)
                if update_model:
                    self._update_global()

                self.rscale = new_rscale

            if docalc and update_model:
                self._update_global()
            if return_tiles:
                return self._tile_global()

        return True

    def add_particle(self, p, r):
        if not self.varyn:
            raise AttributeError("Add/remove particles not supported by state, varyn=False")

        n = self.obj.typ.argmin()

        bp = self.block_particle_pos(n)
        br = self.block_particle_rad(n)
        bt = self.block_particle_typ(n)

        bps = self.explode(bp)
        for i in xrange(bp.sum()):
            self.update(bps[i], np.array([p.ravel()[i]]))
        self.update(br, np.array([r]))
        self.update(bt, np.array([1]))

        return n

    def remove_particle(self, n):
        if not self.varyn:
            raise AttributeError("Add/remove particles not supported by state, varyn=False")

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

    """
    def _grad_func(self, bl, dl=1e-3, func=None, args=(), kwargs={}):
        self.push_update(bl, self.state[bl]+dl)
        m1 = func(*args, **kwargs)
        self.pop_update()

        self.push_update(bl, self.state[bl]-dl)
        m0 = func(*args, **kwargs)
        self.pop_update()

        return (m1 - m0) / (2*dl)

    def _build_funcs(self):
        self._grad_image = partial(self._grad_func, func=self.get_model_image)
        self._grad_residuals = partial(self._grad_func, func=self.residuals)
    """

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

    def residuals(self, masked=True):
        return self.get_difference_image(doslice=masked)

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

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

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
            self.varyn, self.nlogs, self.difference,
            self.pad, self.slab, self.newconst, self.bkg,
            self.method)

    @contextmanager
    def no_ll_update(self):
        try:
            self.dollupdate = False
            yield
        except Exception as e:
            raise
        finally:
            self.dollupdate = True

def save(state, filename=None, desc='', extra=None):
    """
    Save the current state with extra information (for example samples and LL
    from the optimization procedure).

    state : peri.states.ConfocalImagePython
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

    if os.path.exists(filename):
        ff = "{}-tmp-for-copy".format(filename)

        if os.path.exists(ff):
            os.remove(ff)

        os.rename(filename, ff)

    pickle.dump(save, open(filename, 'wb'))

def load(filename):
    """ Load the state from the given file, moving to the file's directory during load """
    path, name = os.path.split(filename)
    path = path or '.'

    with util.indir(path):
        return pickle.load(open(filename, 'rb'))
