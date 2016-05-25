import os
import re
import json
import types
import numpy as np
import cPickle as pickle

from functools import partial
from contextlib import contextmanager

from peri import const, util
from peri.comp import ParameterGroup, ComponentCollection
from peri.logger import log as baselog
log = baselog.getChild('state')

class ModelError(Exception):
    pass

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
        pass

    def random_pixels(self, N, flat=True):
        """
        Return N random pixel indices from the data shape. If flat, then
        return raveled coordinates only.
        """
        inds = np.random.choice(self.data.size, size=N, replace=False)
        if not flat:
            return np.unravel_index(inds, self.data.shape)
        return inds

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
        pass

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
        """
        Gradient of `func` wrt a single parameter `p`.

        Parameters:
        -----------
        func : callable
            Function wrt to take a derivative, should return a nparray that is
            the same shape for all params and values

        p : string
            Paramter to take the derivative wrt

        dl : float
            Derivative step size for numerical deriv

        f0 : ndarray or float (optional)
            Value at the current parameters. Useful when evaluating many derivs
            so that it is not recalculated for every parameter.

        rts : boolean
            Return To State. Return the state to how you found it when done,
            needs another update call, so can be ommitted sometimes (small dl)

        **kwargs :
            Arguments to `func`
        """
        vals = np.array(self.get_values(p))
        f0 = util.callif(func(**kwargs)) if f0 is None else f0

        self.update(p, vals+dl)
        f1 = util.callif(func(**kwargs))

        if rts:
            self.update(p, vals)

        return (f1 - f0) / dl

    def _hess_two_param(self, func, p0, p1, dl=1e-3, f0=None, rts=True, **kwargs):
        """
        Hessian of `func` wrt two parameters `p0` and `p1`. Parameters follow
        conventions of State._grad_one_param.
        """
        vals0 = np.array(self.get_values(p0))
        vals1 = np.array(self.get_values(p1))

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

    def _grad(self, func, params=None, dl=1e-3, rts=True, **kwargs):
        """
        Gradient of `func` wrt a set of parameters params. Parameters follow
        conventions of State._grad_one_param.
        """
        if params is None:
            params = self.block_all()

        ps = util.listify(params)
        f0 = util.callif(func(**kwargs))

        grad = []
        for i, p in enumerate(ps):
            tgrad = self._grad_one_param(
                func, p, dl=dl, f0=f0, rts=rts, **kwargs
            )
            grad.append(tgrad)
        return np.array(grad)

    def _jtj(self, func, params=None, dl=1e-3, rts=True, **kwargs):
        """
        jTj of a `func` wrt to parmaeters `params`. Parameters follow
        conventions of State._grad_one_param.
        """
        grad = self._grad(func=func, params=params, dl=dl, rts=rts, **kwargs)
        return np.dot(grad, grad.T)

    def _hess(self, func, params=None, dl=1e-3, rts=True, **kwargs):
        """
        Hessian of a `func` wrt to parmaeters `params`. Parameters follow
        conventions of State._grad_one_param.
        """
        if params is None:
            params = self.block_all()

        ps = util.listify(params)
        f0 = util.callif(func(**kwargs))

        hess = [[0]*len(ps) for i in xrange(len(ps))]
        for i, pi in enumerate(ps):
            for j, pj in enumerate(ps[i:]):
                J = j + i
                thess = self._hess_two_param(
                    func, pi, pj, dl=dl, f0=f0, rts=rts, **kwargs
                )
                hess[i][J] = thess
                hess[J][i] = thess
        return np.array(hess)

    def build_funcs(self):
        # FIXME -- docstrings
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

    @property
    def data(self):
        """ Get the raw data of the model fit """
        return self._data

    @property
    def model(self):
        """ Get the current model fit to the data """
        return np.polyval(self.values, self._xpts)

    @property
    def residuals(self):
        return self.model - self.data

    def loglikelihood(self):
        sig = self.param_dict['sigma']
        return (
            -(0.5 * (self.residuals/sig)**2).sum()
            -np.log(np.sqrt(2*np.pi)*sig)*self._data.shape[0]
        )

    def logprior(self):
        return 1.0

class ImageState(State, ComponentCollection):
    def __init__(self, image, comps, offset=0, sigma=0.04, priors=None,
            nlogs=True, pad=const.PAD, modelnum=0, catmap=None, modelstr=None):
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

        comp : list of `peri.comp.Component`s or `peri.comp.ComponentCollection`s
            Components used to make up the model image. Each separate component
            must be of a different category, otherwise combining them would be
            ambiguous. If you desire multiple Components of one category,
            combine them using a ComponentCollection (which has functions for
            combining) and supply that to the comps list.

            The component types must match the list of categories in the
            ImageState.catmap which tells how components are matched to
            parts of the model equation.

        offset : float, typically (0, 1) [default: 1]
            The level that particles inset into the illumination field

        priors: list of `peri.priors` [default: ()]
            Whether or not to turn on overlap priors using neighborlists

        nlogs: boolean [default: True]
            Include in the Loglikelihood calculate the term:

                LL = -(p_i - I_i)^2/(2*\sigma^2) - \log{\sqrt{2\pi} \sigma} 

        pad : integer (optional)
            No recommended to set by hand.  The padding level of the raw image needed
            by the PSF support.

        modelnum : integer
            Use of the supplied models given by an index. Currently there is:

                0 : H(I*(1-P) + C*P) + B
                1 : H(I*(1-P) + C*P + B)

        catmap : dict
            (Using a custom model) The mapping of variables in the modelstr
            equations to actual component types. For example:

                {'P': 'obj', 'H': 'psf'}

        modelstr : dict of strings
            (Using a custom model) The actual equations used to defined the model.
            At least one eq. is required under the key `full`. Other partial
            update equations can be added where the variable name (e.g. 'dI')
            denotes a partial update of I, for example:

                {'full': 'H(P)', 'dP': 'H(dP)'}
        """
        self.pad = pad
        self.sigma = sigma
        self.priors = priors
        self.dollupdate = True

        self.rscale = 1.0
        self.offset = offset

        ComponentCollection.__init__(self, comps=comps)
        self.set_model(modelnum=modelnum, catmap=catmap, modelstr=modelstr)
        self.set_image(image)
        self.build_funcs()

    def set_model(self, modelnum=None, catmap=None, modelstr=None):
        """
        Setup the image model formation equation and corresponding objects into
        their various objects. See the ImageState __init__ docstring
        for information on these parameters.
        """
        N = 2
        catmaps = [0]*N
        modelstrs = [0]*N

        catmaps[0] = {
            'B': 'bkg', 'I': 'ilm', 'H': 'psf', 'P': 'obj', 'C': 'offset'
        }
        modelstrs[0] = {
            'full' : 'H(I*(1-P)+C*P) + B',
            'dI' : 'H(dI*(1-P))',
            'dP' : 'H((C-I)*dP)',
            'dB' : 'dB'
        }

        catmaps[1] = {
            'P': 'obj', 'H': 'psf', 'C': 'offset'
        }
        modelstrs[1] = {
            'full': 'H(P) + C',
            'dP': 'H(dP)',
            'dC': 'dC'
        }

        if catmap is not None and modelstr is not None:
            self.catmap = catmap
            self.modelstr = modelstr
        elif catmap is not None or modelstr is not None:
            raise AttributeError('If catmap or modelstr is supplied, the other must as well')
        else:
            self.catmap = catmaps[modelnum]
            self.modelstr = modelstrs[modelnum]

        self.mapcat = {v:k for k,v in self.catmap.iteritems()}
        self.check_inputs(self.catmap, self.modelstr, self.comps)

        for c in self.comps:
            setattr(self, '_comp_'+c.category, c)

    def check_inputs(self, catmap, model, comps):
        """
        Make sure that the required comps are included in the list of
        components supplied by the user. Also check that the parameters are
        consistent across the many components.
        """
        error = False
        compcats = [c.category for c in comps]
        regex = re.compile('([a-zA-Z_][a-zA-Z0-9_]*)')

        if not model.has_key('full'):
            raise ModelError(
                'Model must contain a `full` key describing '
                'the entire image formation'
            )

        # Check that the two model descriptors are consistent
        for name, eq in model.iteritems():
            var = regex.findall(eq)
            for v in var:
                # remove the derivative signs
                v = re.sub(r"^d", '', v)
                if v not in catmap:
                    log.error(
                        "Variable '%s' (eq. '%s': '%s') not found in category map %r" %
                        (v, name, eq, catmap)
                    )
                    error = True

        if error:
            raise ModelError('Inconsistent catmap and model descriptions')

        # Check that the components are all provided, given the categories
        for k,v in catmap.iteritems():
            if k not in model['full']:
                log.warn('Component (%s : %s) not used in model.' % (k,v))

            if not v in compcats:
                log.error('Map component (%s : %s) not found in list of components.' % (k,v))
                error = True

        if error:
            raise ModelError('Component list incomplete or incorrect')

    def set_image(self, image):
        """
        Update the current comparison (real) image
        """
        self.rawimage = image
        self._data = self.rawimage.get_padded_image(self.pad)

        #self.image_mask = (image > const.PADVAL).astype('float')
        #self.image *= self.image_mask

        self.inner = (np.s_[self.pad:-self.pad],)*3
        self.reset()

    def reset(self):
        for c in self.comps:
            c.initialize()

        self._model = self._calc_model()
        self._residuals = self._calc_residuals()
        self._loglikelihood = self._calc_loglikelihood()
        #self._logprior = self._calc_logprior()

    def model_to_true_image(self):
        """
        In the case of generating fake data, use this method to add
        noise to the created image (only for fake data) and rotate
        the model image into the true image
        """
        im = self.model.copy()
        im = im + self.image_mask * np.random.normal(0, self.sigma, size=self.image.shape)
        im = im + (1 - self.image_mask) * const.PADVAL
        self.set_image(im)

    @property
    def data(self):
        """ Get the raw data of the model fit """
        return self._data

    @property
    def model(self):
        """ Get the current model fit to the data """
        return self._model

    @property
    def residuals(self):
        return self._residuals

    def get_io_tiles(self, otile, ptile):
        """
        Get the tiles corresponding to a particular section of image needed to
        be updated. Inputs are the update tile and padding tile. Returned is
        the padded tile, inner tile, and slicer to go between, but accounting
        for wrap with the edge of the image as necessary.
        """
        # now remove the part of the tile that is outside the image and
        # pad the interior part with that overhang
        img = util.Tile(self.data.shape)

        # reflect the necessary padding back into the image itself for
        # the outer slice which we will call outer
        outer = otile.pad((ptile.shape+1)/2)
        inner, outer = outer.reflect_overhang(img)
        iotile = inner.translate(-outer.l)

        return outer, inner, iotile

    def _map_vars(self, funcname, extra=None, *args, **kwargs):
        out = {}
        extra = extra or {}

        for c in self.comps:
            cat = c.category
            out[self.mapcat[cat]] = getattr(c, funcname)(*args, **kwargs)

        out.update(extra)
        return out

    def update(self, params, values):
        """
        Actually perform an image (etc) update based on a set of params and
        values. These parameter can be any present in the components in any
        number. If there is only one component affected then difference image
        updates will be employed.
        """
        comps = self.affected_components(params)

        # get the affected area of the model image
        otile = self.get_update_tile(params, values)
        ptile = self.get_padding_size(otile)
        itile, otile, iotile = self.get_io_tiles(otile, ptile)

        # have all components update their tiles
        self.set_tile(otile)

        oldmodel = self.model[itile.slicer].copy()

        # here we diverge depending if there is only one component update
        # (so that we may calculate a variation / difference image) or if many
        # parameters are being update (should just update the whole model).
        if len(comps) == 1 and len(comps[0].category) == 1:
            comp = comps[0]
            compname = self.mapcat[comp.category]
            dcompname = 'd'+compname

            # FIXME -- check that self.modelstr[dcompname] exists first

            model0 = comp.get_field()
            super(ImageState, self).update(params, values)
            model1 = comp.get_field()

            diff = model1 - model0
            evar = self._map_vars('get_field', extra={dcompname: diff})
            diff = eval(self.modelstr[dcompname], evar)

            self._model[itile.slicer] += diff[iotile.slicer]
        else:
            super(ImageState, self).update(params, values)

            # unpack a few variables to that this is easier to read, nice compact
            # formulas coming up, B = bkg, I = ilm, C = off
            evar = self._map_vars('get_field')
            diff = eval(self.modelstr['full'], evar)
            self._model[itile.slicer] = diff[iotile.slicer]

        newmodel = self.model[itile.slicer].copy()

        # use the model image update to modify other class variables which
        # are hard to compute globally for small local updates
        self.update_from_model_change(oldmodel, newmodel, itile)

    def _calc_model(self):
        self.set_tile(util.Tile(self.data.shape))
        var = self._map_vars('get_field')
        return eval(self.modelstr['full'], var)

    def _calc_residuals(self):
        return self.model - self.data

    def _calc_loglikelihood(self, model=None, tile=None):
        if model is None:
            res = self.residuals
        else:
            res = model - self.data[tile.slicer]

        sig = 0.1#self.get_values('sigma')
        return -0.5*((res/sig)**2).sum() - np.log(np.sqrt(2*np.pi)*sig)*res.size

    def update_from_model_change(self, oldmodel, newmodel, tile):
        """
        Update various internal variables from a model update from oldmodel to
        newmodel for the tile `tile`
        """
        self._loglikelihood -= self._calc_loglikelihood(oldmodel, tile=tile)
        self._loglikelihood += self._calc_loglikelihood(newmodel, tile=tile)
        self._residuals[tile.slicer] = newmodel - self.data[tile.slicer]

    def loglikelihood(self):
        return self._logprior + self._loglikelihood

    def __str__(self):
        return "{} [\n    {}\n]".format(self.__class__.__name__,
            '\n    '.join([c.category+': '+str(c) for c in self.comps])
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, idct):
        pass


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
