import os
import numpy as np
import cPickle as pickle
from contextlib import contextmanager

from peri import const, util
from peri.comp import ComponentCollection

class State(ComponentCollection):
    def __init__(self, params, values, logpriors=None):
        self.stack = []
        self.logpriors = logpriors

    @property
    def data(self):
        """ Get the raw data of the model fit """
        pass

    @property
    def model(self):
        """ Get the current model fit to the data """
        pass

    @property
    def residuals(self):
        pass

    def residuals_sample(self, inds=None):
        if inds is not None:
            return self.residuals.ravel[inds]
        return self.residuals

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
    fisherinformation = partial(_jtj, func=self.model_sample) #FIXME -- sigma^2
    J = partial(_grad, func=self.residuals_sample)
    JTJ = partial(_jtj, func=self.residuals_sample)

    def crb(self, p):
        pass

class ConfocalImagePython(State):
    def __init__(self, image, comp, zscale=1, offset=0, sigma=0.04, priors=None,
            constoff=False, nlogs=True, pad=const.PAD, newconst=True, method=1):
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

        nlogs: boolean [default: False]
            Include in the Loglikelihood calculate the term:

                LL = -(p_i - I_i)^2/(2*\sigma^2) - \log{\sqrt{2\pi} \sigma} 

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

        super(ConfocalImagePython, self).__init__(comps=comps)
        self.set_image(image)

    def reset(self):
        if self.rawimage is not None:
            self.set_image(self.rawimage)
        else:
            self.set_image(self.padded_image())

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

    @property
    def model(self):
        return self.model_image * self.image_mask

    def get_true_image(self):
        return self.image * self.image_mask

    def get_difference_image(self, doslice=True):
        o = self.get_true_image() - self.get_model_image()
        if doslice:
            return o[self.inner]
        return o

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

    def get_support_size(self, params, values):
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

    def update(self, params, values):
        model = self.model.copy()

        for c in self.comps:
            c.update(params, values)

    def create_block(self, typ='all'):
        return self.block_range(*self._block_offset_end(typ))

    def residuals(self, masked=True):
        return self.get_difference_image(doslice=masked)

    def loglikelihood(self):
        return self._logprior + self._loglikelihood

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
            self.nlogs, self.difference,
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
