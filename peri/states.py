import os
import re
import copy
import json
import numpy as np
import cPickle as pickle

from functools import partial
from contextlib import contextmanager

from peri import util, comp, models
from peri.logger import log as baselog
log = baselog.getChild('states')

class UpdateError(Exception):
    pass

def sample(field, inds=None, slicer=None, flat=True):
    """
    Take a sample from a field given flat indices or a shaped slice

    Parameters:
    -----------
    inds : list of indices
        One dimensional (raveled) indices to return from the field

    slicer : slice object
        A shaped (3D) slicer that returns a section of image

    flat : boolean
        Whether to flatten the sampled item before returning
    """
    if inds is not None:
        out = field.ravel()[inds]
    elif slicer is not None:
        out = field[slicer].ravel()
    else:
        out = field

    if flat:
        return out.ravel()
    return out

#=============================================================================
# Super class of State, has all basic components and structure
#=============================================================================
class State(comp.ParameterGroup):
    def __init__(self, params, values, logpriors=None, **kwargs):
        self.stack = []
        self.logpriors = logpriors

        super(State, self).__init__(params, values, **kwargs)
        self.build_funcs()

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
        """ Get the model residuals wrt data """
        return self.data - self.model

    @property
    def error(self):
        return np.dot(self.residuals.flat, self.residuals.flat)

    @property
    def loglikelihood(self):
        pass

    @property
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

    def param_all(self):
        return self.params

    _graddoc = \
    """
    Parameters:
    -----------
    func : callable
        Function wrt to take a derivative, should return a nparray that is
        the same shape for all params and values

    params : string or list of strings
        Paramter(s) to take the derivative wrt

    dl : float
        Derivative step size for numerical deriv

    rts : boolean
        Return To Start. Return the state to how you found it when done,
        needs another update call, so can be ommitted sometimes (small dl).
        If True, functions return the final answer along with the final func
        evaluation so that it may be passed onto other calls.

    **kwargs :
        Arguments to `func`
    """

    _sampledoc = \
    """
    kwargs (supply only one):
    -----------------------------
    inds : list of indices
        One dimensional (raveled) indices to return from the field

    slicer : slice object
        A shaped (3D) slicer that returns a section of image

    flat : boolean
        Whether to flatten the sampled item before returning
    """

    def _grad_one_param(self, funct, p, dl=2e-5, rts=False, **kwargs):
        """
        Gradient of `func` wrt a single parameter `p`. (see _graddoc)
        """
        vals = self.get_values(p)
        f0 = funct(**kwargs)

        self.update(p, vals+dl)
        f1 = funct(**kwargs)

        if rts:
            self.update(p, vals)
        return (f1 - f0) / dl

    def _hess_two_param(self, funct, p0, p1, dl=2e-5, rts=False, **kwargs):
        """
        Hessian of `func` wrt two parameters `p0` and `p1`. (see _graddoc)
        """
        vals0 = self.get_values(p0)
        vals1 = self.get_values(p1)

        f00 = funct(**kwargs)

        self.update(p0, vals0+dl)
        f10 = funct(**kwargs)

        self.update(p1, vals1+dl)
        f11 = funct(**kwargs)

        self.update(p0, vals0)
        f01 = funct(**kwargs)

        if rts:
            self.update(p0, vals0)
            self.update(p1, vals1)
        return (f11 - f10 - f01 + f00) / (dl**2)

    def _grad(self, funct, params=None, dl=2e-5, rts=False, **kwargs):
        """
        Gradient of `func` wrt a set of parameters params. (see _graddoc)
        """
        if params is None:
            params = self.param_all()

        ps = util.listify(params)
        f0 = funct(**kwargs)

        # get the shape of the entire gradient to return and make an array
        shape = f0.shape if isinstance(f0, np.ndarray) else (1,)
        shape = (len(ps),) + shape
        grad = np.zeros(shape)

        for i, p in enumerate(ps):
            grad[i] = self._grad_one_param(funct, p, dl=dl, rts=rts, **kwargs)
        return grad #was np.squeeze(grad)

    def _jtj(self, funct, params=None, dl=2e-5, rts=False, **kwargs):
        """
        jTj of a `func` wrt to parmaeters `params`. (see _graddoc)
        """
        grad = self._grad(funct=funct, params=params, dl=dl, rts=rts, **kwargs)
        return np.dot(grad, grad.T)

    def _hess(self, funct, params=None, dl=2e-5, rts=False, **kwargs):
        """
        Hessian of a `func` wrt to parmaeters `params`. (see _graddoc)
        """
        if params is None:
            params = self.param_all()

        ps = util.listify(params)
        f0 = funct(**kwargs)

        # get the shape of the entire hessian, allocate an array
        shape = f0.shape if isinstance(f0, np.ndarray) else (1,)
        shape = (len(ps), len(ps)) + shape
        hess = np.zeros(shape)

        for i, pi in enumerate(ps):
            for j, pj in enumerate(ps[i:]):
                J = j + i
                thess = self._hess_two_param(funct, pi, pj, dl=dl, rts=rts, **kwargs)
                hess[i][J] = thess
                hess[J][i] = thess
        return np.squeeze(hess)

    def _dograddoc(self, f):
        f.im_func.func_doc += self._graddoc

    def build_funcs(self):
        """
        Here, we build gradient and hessian functions based on the properties
        of a state that are generally wanted. For each one, we fill in _grad or
        _hess with a function that takes care of various options such as
        slicing and flattening. For example, `m` below takes the model, selects
        different indices from it, maybe flattens it and copies it. This is
        then used in the fisherinformation, gradmodel, and hessmodel functions.
        """
        # create essentially lambda functions, but with a nice signature
        def m(inds=None, slicer=None, flat=True):
            return sample(self.model, inds=inds, slicer=slicer, flat=flat).copy()

        def r(inds=None, slicer=None, flat=True):
            return sample(self.residuals, inds=inds, slicer=slicer, flat=flat).copy()

        def l():
            return self.loglikelihood

        # set the member functions using partial
        self.fisherinformation = partial(self._jtj, funct=m)
        self.gradloglikelihood = partial(self._grad, funct=l)
        self.hessloglikelihood = partial(self._hess, funct=l)
        self.gradmodel = partial(self._grad, funct=m)
        self.hessmodel = partial(self._hess, funct=m)
        self.JTJ = partial(self._jtj, funct=r)
        self.J = partial(self._grad, funct=r)

        # add the appropriate documentation to the following functions
        self.fisherinformation.__doc__ = self._graddoc + self._sampledoc
        self.gradloglikelihood.__doc__ = self._graddoc
        self.hessloglikelihood.__doc__ = self._graddoc
        self.gradmodel.__doc__ = self._graddoc + self._sampledoc
        self.hessmodel.__doc__ = self._graddoc + self._sampledoc
        self.JTJ.__doc__ = self._graddoc + self._sampledoc
        self.J.__doc__ = self._graddoc + self._sampledoc

        # add documentation to the private functions as well. this is done
        # slightly differently, hence the function call
        self._dograddoc(self._grad_one_param)
        self._dograddoc(self._hess_two_param)
        self._dograddoc(self._grad)
        self._dograddoc(self._hess)

        # the state object is a workaround so that other interfaces still
        # work. this should probably be removed in the long run
        class _Statewrap(object):
            def __init__(self, obj):
                self.obj = obj
            def __getitem__(self, d=None):
                if d is None:
                    d = self.obj.params
                return util.delistify(self.obj.get_values(d), d)

        self.state = _Statewrap(self)

    def crb(self, params=None, *args, **kwargs):
        """
        Calculate the diagonal elements of the minimum covariance of the model
        with respect to parameters params. *args and **kwargs go to fisherinformation.
        """
        fish = self.fisherinformation(params=params, *args, **kwargs)
        return np.sqrt(np.diag(np.linalg.inv(fish))) * self.sigma

    def __str__(self):
        return "{}\n{}".format(self.__class__.__name__, json.dumps(self.param_dict, indent=2))

    def __repr__(self):
        return self.__str__()


class PolyFitState(State):
    def __init__(self, x, y, order=2, coeffs=None, sigma=1.0):
        # FIXME -- add prior for sigma > 0
        self._data = y
        self._xpts = x

        params = ['c-%i' %i for i in xrange(order)]
        values = coeffs if coeffs is not None else [0.0]*order

        params.append('sigma')
        values.append(1.0)

        super(PolyFitState, self).__init__(
            params=params, values=values, ordered=False
        )

        self.update(self.params, self.values)

    def update(self, params, values):
        super(PolyFitState, self).update(params, values)
        self._model = np.polyval(self.values, self._xpts)

    @property
    def data(self):
        """ Get the raw data of the model fit """
        return self._data

    @property
    def model(self):
        """ Get the current model fit to the data """
        return self._model

    @property
    def loglikelihood(self):
        sig = self.param_dict['sigma']
        return (
            -(0.5 * (self.residuals/sig)**2).sum()
            -np.log(np.sqrt(2*np.pi)*sig)*self._data.oshape[0]
        )

    @property
    def logprior(self):
        return 1.0

#=============================================================================
# Image state which specializes to components with regions, etc.
#=============================================================================
class ImageState(State, comp.ComponentCollection):
    def __init__(self, image, comps, mdl=models.ConfocalImageModel(), sigma=0.04,
            priors=None, pad=24, model_as_data=False):
        """
        The state object to create a confocal image.  The model is that of
        a spatially varying illumination field, from which platonic particle
        shapes are subtracted.  This is then spread with a point spread function
        (PSF).

        Parameters:
        -----------
        image : `peri.util.Image` object
            The raw image with which to compare the model image from this
            class.  This image should have been prepared through
            prepare_for_state, which does things such as padding necessary for
            this class. In the case of the RawImage, paths are used to keep
            track of the image object to save on pickle size.

        comp : list of `peri.comp.Component`s or `peri.comp.ComponentCollection`s
            Components used to make up the model image. Each separate component
            must be of a different category, otherwise combining them would be
            ambiguous. If you desire multiple Components of one category,
            combine them using a ComponentCollection (which has functions for
            combining) and supply that to the comps list.

            The component types must match the list of categories in the
            ImageState.catmap which tells how components are matched to
            parts of the model equation.

        mdl : `peri.models.Model` object
            Model defining how to combine different Components into a single
            model.

        priors: list of `peri.priors` [default: ()]
            Whether or not to turn on overlap priors using neighborlists

        pad : integer or tuple of integers (optional)
            No recommended to set by hand.  The padding level of the raw image
            needed by the PSF support.

        model_as_data : boolean
            Whether to use the model image as the true image after initializing
        """
        self.dim = len(image.get_image().shape)

        self.sigma = sigma
        self.priors = priors
        self.pad = util.aN(pad, dim=self.dim)
        self.model_as_data = model_as_data

        comp.ComponentCollection.__init__(self, comps=comps)

        self.set_model(mdl=mdl)
        self.set_image(image)
        self.build_funcs()

        if self.model_as_data:
            self.model_to_data(self.sigma)

    def set_model(self, mdl):
        """
        Setup the image model formation equation and corresponding objects into
        their various objects. `mdl` is a `peri.models.Model` object
        """
        self.mdl = mdl
        self.mdl.check_inputs(self.comps)

        for c in self.comps:
            setattr(self, '_comp_'+c.category, c)

    def set_image(self, image):
        """
        Update the current comparison (real) image
        """
        if isinstance(image, np.ndarray):
            image = util.Image(image)

        if isinstance(image, util.NullImage):
            self.model_as_data = True
        else:
            self.model_as_data = False

        self.image = image
        self._data = self.image.get_padded_image(self.pad)

        # set up various slicers and Tiles associated with the image and pad
        self.oshape = util.Tile(self._data.shape)
        self.ishape = self.oshape.pad(-self.pad)
        self.inner = self.ishape.slicer

        for c in self.comps:
            c.set_shape(self.oshape, self.ishape)

        self._model = np.zeros(self._data.shape, dtype=np.float64)
        self._residuals = np.zeros(self._data.shape, dtype=np.float64)
        self.calculate_model()

    def set_tile_full(self):
        self.set_tile(self.oshape)

    def model_to_data(self, sigma=0.0):
        """ Switch out the data for the model's recreation of the data. """
        im = self.model.copy()
        im += sigma*np.random.randn(*im.shape)
        self.set_image(util.NullImage(image=im))

    def reset(self):
        for c in self.comps:
            c.initialize()
        self.calculate_model()

    def calculate_model(self):
        self._model[:] = self._calc_model()
        self._residuals[:] = self._calc_residuals()
        self._loglikelihood = self._calc_loglikelihood()
        self._logprior = self._calc_logprior()

    @property
    def data(self):
        """ Get the raw data of the model fit """
        return self._data[self.inner]

    @property
    def model(self):
        """ Get the current model fit to the data """
        return self._model[self.inner]

    @property
    def residuals(self):
        return self._residuals[self.inner]

    @property
    def loglikelihood(self):
        return self._logprior + self._loglikelihood

    def get_update_io_tiles(self, params, values):
        """
        Get the tiles corresponding to a particular section of image needed to
        be updated. Inputs are the parameters and values. Returned is the
        padded tile, inner tile, and slicer to go between, but accounting for
        wrap with the edge of the image as necessary.
        """
        # get the affected area of the model image
        otile = self.get_update_tile(params, values)
        if otile is None:
            return [None]*3
        ptile = self.get_padding_size(otile) or util.Tile(0)

        otile = util.Tile.intersection(otile, self.oshape)

        if (otile.shape <= 0).any():
            raise UpdateError("update triggered invalid tile size")

        if (ptile.shape < 0).any() or (ptile.shape > self.oshape.shape).any():
            raise UpdateError("update triggered invalid padding tile size")

        # now remove the part of the tile that is outside the image and pad the
        # interior part with that overhang. reflect the necessary padding back
        # into the image itself for the outer slice which we will call outer
        outer = otile.pad((ptile.shape+1)/2)
        inner, outer = outer.reflect_overhang(self.oshape)
        iotile = inner.translate(-outer.l)

        outer = util.Tile.intersection(outer, self.oshape)
        inner = util.Tile.intersection(inner, self.oshape)
        return outer, inner, iotile

    def update(self, params, values):
        """
        Actually perform an image (etc) update based on a set of params and
        values. These parameter can be any present in the components in any
        number. If there is only one component affected then difference image
        updates will be employed.
        """
        comps = self.affected_components(params)

        if len(comps) == 0:
            return False

        # get the affected area of the model image
        otile, itile, iotile = self.get_update_io_tiles(params, values)

        if otile is None:
            return False

        # have all components update their tiles
        self.set_tile(otile)

        oldmodel = self._model[itile.slicer].copy()

        # here we diverge depending if there is only one component update
        # (so that we may calculate a variation / difference image) or if many
        # parameters are being update (should just update the whole model).
        if len(comps) == 1 and self.mdl.get_difference_model(comps[0].category):
            comp = comps[0]
            model0 = copy.deepcopy(comp.get())
            super(ImageState, self).update(params, values)
            model1 = copy.deepcopy(comp.get())

            diff = model1 - model0
            diff = self.mdl.evaluate(
                self.comps, 'get', diffmap={comp.category: diff}
            )

            if isinstance(model0, (float, int)):
                self._model[itile.slicer] += diff
            else:
                self._model[itile.slicer] += diff[iotile.slicer]
        else:
            super(ImageState, self).update(params, values)

            # allow the model to be evaluated using our components
            diff = self.mdl.evaluate(self.comps, 'get')
            self._model[itile.slicer] = diff[iotile.slicer]

        newmodel = self._model[itile.slicer].copy()

        # use the model image update to modify other class variables which
        # are hard to compute globally for small local updates
        self.update_from_model_change(oldmodel, newmodel, itile)
        return True

    def get(self, name):
        """ Return component by category name """
        for c in self.comps:
            if c.category == name:
                return c
        return None

    def set(self, name, obj):
        comp.ComponentCollection.set(self, name, obj)
        obj.set_shape(self.oshape, self.ishape)
        self.calculate_model()

    def _calc_model(self):
        self.set_tile_full()
        return self.mdl.evaluate(self.comps, 'get')

    def _calc_residuals(self):
        return self._data - self._model

    def _calc_logprior(self):
        return 1.0 # FIXME

    def _calc_loglikelihood(self, model=None, tile=None):
        if model is None:
            res = self.residuals
        else:
            res = model - self._data[tile.slicer]

        sig, isig = self.sigma, 1.0/self.sigma
        nlogs = -np.log(np.sqrt(2*np.pi)*sig)*res.size
        return -0.5*isig*isig*np.dot(res.flat, res.flat) + nlogs

    def update_from_model_change(self, oldmodel, newmodel, tile):
        """
        Update various internal variables from a model update from oldmodel to
        newmodel for the tile `tile`
        """
        self._loglikelihood -= self._calc_loglikelihood(oldmodel, tile=tile)
        self._loglikelihood += self._calc_loglikelihood(newmodel, tile=tile)
        self._residuals[tile.slicer] = self._data[tile.slicer] - newmodel

    def exports(self):
        raise NotImplementedError('inherited but not relevant')

    def register(self, parent):
        raise NotImplementedError('inherited but not relevant')

    def __str__(self):
        def _pad(s):
            return re.subn('(\n)', '\n    ', s)[0]

        stats = _pad('\nstats: E={} LL={}\n'.format(self.error, self.loglikelihood))
        model = _pad('model: {}\n'.format(str(self.mdl)))
        image = _pad('image: {}\n'.format(str(self.image)))
        comps = _pad('\n'.join([c.category+': '+str(c) for c in self.comps]))

        return "{} [{}{}{}{}{}\n]".format(
            self.__class__.__name__, stats, model, image, _pad('-'*70+'\n'), comps
        )

    def __getstate__(self):
        return {'image': self.image, 'comps': self.comps, 'mdl': self.mdl,
                'sigma': self.sigma, 'priors': self.priors, 'pad': self.pad,
                'model_as_data': self.model_as_data}

    def __setstate__(self, idct):
        self.__init__(**idct)

    def set_mem_level(self, mem_level='hi'):
        """
        Sets the memory usage level of the state.
            hi      : all mem's are np.float64
            med-hi  : image, platonic are float32, rest are float64
            med     : all mem's are float32
            med-lo  : image, platonic are float16, rest float32
            lo      : all are float16, which is bad for accuracy.
        Right now the PSF is not affected by the mem-level changes, which is
        OK for mem but it means that self._model, self._residuals are always
        float64, which can be a chunk of mem.
        """
        #A little thing to parse strings for convenience:
        key = ''.join(map(lambda c: c if c in 'mlh' else '', mem_level))
        if key not in ['h','mh','m','ml','m', 'l']:
            raise ValueError('mem_level must be one of hi, med-hi, med, med-lo, lo.')
        mem_levels = {  'h':     [np.float64, np.float64],
                        'mh': [np.float64, np.float32],
                        'm':   [np.float32, np.float32],
                        'ml':  [np.float32, np.float16],
                        'l':      [np.float16, np.float16]
                    }
        hi_lvl, lo_lvl = mem_levels[key]

        self.image.float_precision = hi_lvl
        self.image.image = self.image.image.astype(lo_lvl)
        self.set_image(self.image)

        for c in ['ilm','bkg']:
            self.get(c).float_precision = hi_lvl
        for c in self.get('obj').comps:
            c.float_precision = lo_lvl
        self._model = self._model.astype(hi_lvl)
        self._residuals = self._model.astype(hi_lvl)
        self.reset()


def save(state, filename=None, desc='', extra=None):
    """
    Save the current state with extra information (for example samples and LL
    from the optimization procedure).

    state : peri.states.ImageState
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
    if isinstance(state.image, util.RawImage):
        desc = desc or 'save'
        filename = filename or state.image.filename + '-peri-' + desc + '.pkl'
    else:
        if not filename:
            raise AttributeError("Must provide filename since RawImage is not used")

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
