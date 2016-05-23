import os
import json
import types
import numpy as np
import cPickle as pickle

from functools import partial
from contextlib import contextmanager

from peri import const, util
from peri.comp import ParameterGroup, ComponentCollection

def superdoc(func):
    def wrapper(func):
        func.__doc__ = blah
        return func
    return wrapper(func)

class State(ParameterGroup):
    def __init__(self, params, values, logpriors=None, **kwargs):
        self.stack = []
        self.logpriors = logpriors

        super(State, self).__init__(params, values, **kwargs)
        self._build_funcs()

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

    def residuals_sample(self, inds=None, slicer=None):
        if inds is not None:
            return self.residuals.ravel()[inds]
        if slicer is not None:
            return self.residuals[slicer]
        return self.residuals

    def model_sample(self, inds=None, slicer=None):
        if inds is not None:
            return self.model.ravel()[inds]
        if slicer is not None:
            return self.model[slicer]
        return self.model

    def loglikelihood(self):
        loglike = self.dologlikelihood()
        if self.logpriors is not None:
            loglike += self.logpriors()
        return loglike

    def logprior(self):
        pass

    def update(self, params, values):
        return super(State, self).update(params, values)

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

        ps = util.listify(ps)
        f0 = util.callif(func(**kwargs))

        grad = []
        for i, p in enumerate(ps):
            grad.append(self._grad_one_param(func, p, dl, f0=f0, **kwargs))
        return np.array(grad)

    def _jtj(self, func, ps=None, dl=1e-3, **kwargs):
        grad = self._grad(func=func, ps=ps, dl=dl, **kwargs)
        return np.dot(grad, grad.T)

    def _hess(self, func, ps=None, dl=1e-3, **kwargs):
        if ps is None:
            ps = self.block_all()

        ps = util.listify(ps)
        f0 = util.callif(func(**kwargs))

        hess = [[0]*len(ps) for i in xrange(len(ps))]
        for i, pi in enumerate(ps):
            for j, pj in enumerate(ps[i:]):
                J = j + i
                thess = self._hess_two_param(func, pi, pj, dl=dl, f0=f0, **kwargs)
                hess[i][J] = thess
                hess[J][i] = thess
        return np.array(hess)

    def _build_funcs(self):
        self.gradloglikelihood = partial(self._grad, func=self.loglikelihood)
        self.hessloglikelihood = partial(self._hess, func=self.loglikelihood)
        self.fisherinformation = partial(self._jtj, func=self.model_sample) #FIXME -- sigma^2
        self.J = partial(self._grad, func=self.residuals_sample)
        self.JTJ = partial(self._jtj, func=self.residuals_sample)

        class _Statewrap(object):
            def __init__(self, obj):
                self.obj = obj
            def __getitem__(self, d):
                return self.obj.get_values(d)

        self.state = _Statewrap(self)

    def crb(self, p):
        pass

    def __str__(self):
        return "{}\n{}".format(self.__class__.__name__, json.dumps(self.param_dict, indent=2))

    def __repr__(self):
        return self.__str__()


class LinearFitState(State):
    def __init__(self, x, d, m=0.0, b=0.0, sigma=1.0):
        # FIXME -- add prior for sigma > 0
        self._data = d
        self._xpts = x
        super(LinearFitState, self).__init__(['m', 'b', 'sigma'], [m, b, sigma], ordered=False)

    @property
    def data(self):
        """ Get the raw data of the model fit """
        return self._data

    @property
    def model(self):
        """ Get the current model fit to the data """
        return self.param_dict['m']*self._xpts + self.param_dict['b']

    @property
    def residuals(self):
        return self.model - self.data

    def loglikelihood(self):
        sig = self.param_dict['sigma']
        return -((self.residuals)**2).sum() / (2*sig**2) - 0.5*np.log(2*np.pi*sig*sig)*self._data.shape[0]


class ConfocalImageState(State, ComponentCollection):
    def __init__(self, image, comp, zscale=1, offset=0, sigma=0.04, priors=None,
            constoff=False, nlogs=True, pad=const.PAD, newconst=True, method=1,
            modelnum=0, catmap=None, modelstr=None):
        """
        The state object to create a confocal image.  The model is that of
        a spatially varying illumination field, from which platonic particle
        shapes are subtracted.  This is then spread with a point spread function
        (PSF). 

        Parameters:
        -----------
        image : `peri.util.Image` object
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

        priors: boolean [default: False]
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

        """
        self.pad = pad
        self.sigma = sigma
        self.priors = priors

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

        ComponentCollection.__init__(self, comps=comps)
        self.set_image(image)
        self.set_model(modelnum, catmap, modelstr)

    def set_model(self, modelnum=None, catmap=None, modelstr=None):
        catmap0 = {
            'B': 'bkg', 'I': 'ilm', 'H': 'psf', 'P': 'platonic', 'C': 'offset'
        }
        modelstr0 = {
            'full' : 'H(I*(1-P)+C*P) + B',
            'dH' : 'H(I*(1-P)+C*P) + B',
            'dI' : 'H(dI*(1-P))',
            'dP' : 'H((C-I)*dP)',
            'dB' : 'dB'
        }

        catmaps = [catmap0]
        modelstrs = [modelstr0]

        if catmap is not None and modelstr is not None:
            self.catmap = catmap
            self.modelstr = modelstr
        elif catmap is not None or modelstr is not None:
            raise AttributeError('If catmap or modelstr is supplied, the other must as well')
        else:
            self.catmap = catmaps[methodnum]
            self.modelstr = modelstrs[methodnum]

        self.mapcat = {v:k for k,v in self.catmap.iteritems()}

        # FIXME -- make sure that the required comps are included in the list
        # of components supplied by the user. Also check that the parameters
        # are consistent across the many components.

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
        self.model_image = np.zeros_like(self.image)

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

    def get_io_tiles(self, otile, ptile):
        """
        Get the tiles corresponding to a particular section of image needed to
        be updated. Inputs are the update tile and padding tile. Returned is
        the padded tile, inner tile, and slicer to go between, but accounting
        for wrap with the edge of the image as necessary.
        """
        # now remove the part of the tile that is outside the image and
        # pad the interior part with that overhang
        img = util.Tile(self.image.shape)

        # reflect the necessary padding back into the image itself for
        # the outer slice which we will call outer
        outer = otile.pad((ptile.shape+1)/2)
        inner, outer = outer.reflect_overhang(img)
        iotile = inner.translate(-outer.l)

        return outer, inner, iotile.slicer

    def _map_vars(self, funcname, extra=None, *args, **kwargs):
        out = {}

        for c in self.comps:
            cat = c.category
            out[self.mapcat[cat]] = c.__dict__[funcname](*args, **kwargs)

        out.update(extra)
        return out

    def update(self, params, values):
        comps = self.affected_components(params)

        otile = self.get_update_tile(params, values)
        ptile = self.get_padding_size(otile)

        itile, otile, iotile = self.get_io_tiles(otile, ptile)
        self.set_tiles(otile)

        # here we diverge depending if there is only one component update
        # (so that we may calculate a variation / difference image) or if many
        # parameters are being update (should just update the whole model).
        if len(comps) == 1 and len(comps[0].category) == 1:
            comp = comps[0]
            compname = self.mapcat[comp.category]
            dcompname = 'd'+compname

            model0 = comp.get_field()
            super(ConfocalImageState, self).update(params, values)
            model1 = comp.get_field()

            diff = model1 - model0
            evar = self._map_vars('get_field', extra={dcompname: diff})
            diff = eval(self.modelstr[dcompname], globals=evar)

            self.model_image[itile.slicer] += diff[iotile.slicer]
        else:
            super(ConfocalImageState, self).update(params, values)

            # unpack a few variables to that this is easier to read, nice compact
            # formulas coming up, B = bkg, I = ilm, C = off
            evar = self._map_vars('get_field')
            diff = eval(self.modelstr['full'], globals=evar)
            self.model_image[itile.slicer] = diff[iotile.slicer]

        self._update_ll_field(self.model_image[islice], islice)

    def residuals(self, masked=True):
        return self.get_difference_image(doslice=masked)

    def loglikelihood(self):
        return self._logprior + self._loglikelihood

    @contextmanager
    def no_ll_update(self):
        try:
            self.dollupdate = False
            yield
        except Exception as e:
            raise
        finally:
            self.dollupdate = True

    def __str__(self):
        return "{} [\n    {}\n]".format(self.__class__.__name__,
            '\n    '.join([str(c) for c in self.comps])
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, idct):
        pass


def save(state, filename=None, desc='', extra=None):
    """
    Save the current state with extra information (for example samples and LL
    from the optimization procedure).

    state : peri.states.ConfocalImageState
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
