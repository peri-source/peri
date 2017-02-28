from future.utils import iteritems
from builtins import object

import re

from peri.comp import (
    ComponentCollection, GlobalScalar, ilms, psfs, objs, exactpsf
)
from peri.logger import log as baselog
log = baselog.getChild('models')

allfields = {
    'const': GlobalScalar,
    'poly3d': ilms.Polynomial3D,
    'leg3d': ilms.LegendrePoly3D,
    'poly2p1d': ilms.Polynomial2P1D,
    'leg2p1d': ilms.LegendrePoly2P1D,
    'cheb2p1d': ilms.ChebyshevPoly2P1D,
    'barnesleg2p1d': ilms.BarnesStreakLegPoly2P1D,
    'combo': ComponentCollection,
}

allpsfs = {
    'identity': psfs.IdentityPSF,
    'gauss2d': psfs.AnisotropicGaussian,
    'gauss3d': psfs.AnisotropicGaussianXYZ,
    'gauss4d': psfs.Gaussian4DPoly,
    'gauss4d-leg': psfs.Gaussian4DLegPoly,
    'gauss4d-mom': psfs.GaussianMomentExpansion,
    'linescan': exactpsf.ExactLineScanConfocalPSF,
    'cheb-linescan': exactpsf.ChebyshevLineScanConfocalPSF,
    'cheb-linescan-fixedss': exactpsf.FixedSSChebLinePSF,
}

allobjs = {
    'spheres': objs.PlatonicSpheresCollection,
    'slab': objs.Slab,
    'combo': ComponentCollection,
}

class ModelError(Exception):
    pass

class Model(object):
    def __init__(self, modelstr, varmap, registry={}):
        """
        An abstraction for defining how to combine components into a complete
        model as well as derivatives with respect to different variables.

        Parameters
        -----------
        modelstr : dict of strings
            (Using a custom model) The actual equations used to defined the model.
            At least one eq. is required under the key `full`. Other partial
            update equations can be added where the variable name (e.g. 'dI')
            denotes a partial update of I, for example::

                {'full': 'H(P)', 'dP': 'H(dP)'}

        varmap : dict
            (Using a custom model) The mapping of variables in the modelstr
            equations to actual component types. For example::

                {'P': 'obj', 'H': 'psf'}

        registry : dict
            A mapping of categories to components which this model will
            accept. For example::

                {
                    'obj': [objs.PlatonicSpheresCollection],
                    'psf': [psfs.Gaussian4DPoly, psfs.GaussianMomentExpansion]
                }
        """
        self.modelstr = modelstr
        self.varmap = varmap
        self.registry = registry
        self.ivarmap = {v:k for k, v in iteritems(self.varmap)}
        self.check_consistency()

    def check_consistency(self):
        """
        Make sure that the required comps are included in the list of
        components supplied by the user. Also check that the parameters are
        consistent across the many components.
        """
        error = False
        regex = re.compile('([a-zA-Z_][a-zA-Z0-9_]*)')

        # there at least must be the full model, not necessarily partial updates
        if 'full' not in self.modelstr:
            raise ModelError(
                'Model must contain a `full` key describing '
                'the entire image formation'
            )

        # Check that the two model descriptors are consistent
        for name, eq in iteritems(self.modelstr):
            var = regex.findall(eq)
            for v in var:
                # remove the derivative signs if there (dP -> P)
                v = re.sub(r"^d", '', v)
                if v not in self.varmap:
                    log.error(
                        "Variable '%s' (eq. '%s': '%s') not found in category map %r" %
                        (v, name, eq, self.varmap)
                    )
                    error = True

        if error:
            raise ModelError('Inconsistent varmap and modelstr descriptions')

    def check_inputs(self, comps):
        """
        Check that the list of components `comp` is compatible with both the
        varmap and modelstr for this Model
        """
        error = False
        compcats = [c.category for c in comps]

        # Check that the components are all provided, given the categories
        for k, v in iteritems(self.varmap):
            if k not in self.modelstr['full']:
                log.warn('Component (%s : %s) not used in model.' % (k,v))

            if v not in compcats:
                log.error('Map component (%s : %s) not found in list of components.' % (k,v))
                error = True

        if error:
            raise ModelError('Component list incomplete or incorrect')

    def diffname(self, name):
        """ Transform a variable name into a derivative """
        return 'd'+name

    def get_base_model(self):
        """ The complete model, no derivatives """
        return self.modelstr['full']

    def get_difference_model(self, category):
        """
        Get the equation corresponding to a variation wrt category. For example
        if::

            modelstr = {
                'full' :'H(I) + B',
                'dH' : 'dH(I)',
                'dI' : 'H(dI)',
                'dB' : 'dB'
            }

            varmap = {'H': 'psf', 'I': 'obj', 'B': 'bkg'}

        then ``get_difference_model('obj') == modelstr['dI'] == 'H(dI)'``
        """
        name = self.diffname(self.ivarmap[category])
        return self.modelstr.get(name)

    def map_vars(self, comps, funcname='get', diffmap=None, **kwargs):
        """
        Map component function ``funcname`` result into model variables
        dictionary for use in eval of the model. If ``diffmap`` is provided then
        that symbol is translated into 'd'+diffmap.key and is replaced by
        diffmap.value. ``**kwargs` are passed to the ``comp.funcname(**kwargs)``.
        """
        out = {}
        diffmap = diffmap or {}

        for c in comps:
            cat = c.category

            if cat in diffmap:
                symbol = self.diffname(self.ivarmap[cat])
                out[symbol] = diffmap[cat]
            else:
                symbol = self.ivarmap[cat]
                out[symbol] = getattr(c, funcname)(**kwargs)

        return out

    def evaluate(self, comps, funcname='get', diffmap=None, **kwargs):
        """
        Calculate the output of a model. It is recommended that at some point
        before using `evaluate`, that you make sure the inputs are valid using
        :class:`~peri.models.Model.check_inputs`

        Parameters
        -----------
        comps : list of :class:`~peri.comp.comp.Component`
            Components which will be used to evaluate the model

        funcname : string (default: 'get')
            Name of the function which to evaluate for the components which
            represent their output. That is, when each component is used in the
            evaluation, it is really their ``attr(comp, funcname)`` which is used.

        diffmap : dictionary
            Extra mapping of derivatives or other symbols to extra variables.
            For example, the difference in a component has been evaluated as
            diff_obj so we set ``{'I': diff_obj}``

        ``**kwargs``:
            Arguments passed to ``funcname`` of component objects
        """
        evar = self.map_vars(comps, funcname, diffmap=diffmap)

        if diffmap is None:
            return eval(self.get_base_model(), evar)
        else:
            compname = list(diffmap.keys())[0]
            return eval(self.get_difference_model(compname), evar)

    def __str__(self):
        return "{} : {}".format(self.__class__.__name__, self.get_base_model())

    def __repr__(self):
        return self.__str__()

class ConfocalImageModel(Model):
    def __init__(self):
        """
        Confocal microscope image with simplifications made for performance.
        In particular, the sensing and illumination point spread functions
        are combined into one:

        .. math::
            \\mathcal{M} = H(I(1-P) + CP) + B

        """
        varmap = {
            'B': 'bkg', 'I': 'ilm', 'H': 'psf', 'P': 'obj', 'C': 'offset'
        }
        modelstr = {
            'full' : 'H(I*(1-P)+C*P) + B',
            'dI' : 'H(dI*(1-P))',
            'dP' : 'H((C-I)*dP)',
            'dC' : 'H(dC*P)',
            'dB' : 'dB',
        }
        registry = {
            'bkg': allfields,
            'ilm': allfields,
            'psf': allpsfs,
            'obj': allobjs,
            'offset': {'const': GlobalScalar}
        }
        Model.__init__(self, modelstr=modelstr, varmap=varmap, registry=registry)

class SmoothFieldModel(Model):
    """
    A smooth field with one component:

    .. math::
        \\mathcal{M} = I
    """
    def __init__(self):
        varmap = {'I': 'ilm'}
        modelstr = {'full': 'I', 'dI': 'dI'}
        registry = {'ilm': allfields}
        Model.__init__(self, modelstr=modelstr, varmap=varmap, registry=registry)

class BlurredFieldModel(Model):
    """
    A blurred field, with two component:

    .. math::
        \\mathcal{M} = H(I)
    """
    def __init__(self):
        varmap = {'H': 'psf', 'I': 'ilm'}
        modelstr = {'full': 'H(I)', 'dI': 'H(dI)'}
        registry = {'psf': allpsfs, 'ilm': allfields}
        Model.__init__(self, modelstr=modelstr, varmap=varmap, registry=registry)

class ParticlesOnlyModel(Model):
    """
    A model of only particles

    .. math::
        \\mathcal{M} = P
    """
    def __init__(self):
        varmap = {'P': 'obj'}
        modelstr = {'full': 'P', 'dP': 'dP'}
        registry = {'obj': allobjs}
        Model.__init__(self, modelstr=modelstr, varmap=varmap, registry=registry)

class ConfocalDyedParticlesModel(Model):
    """
    A model of particles blurred by a PSF, with a background and scale:

    .. math::
        \\mathcal{M} = H(I*P) + B
    """
    def __init__(self):
        varmap = {
            'P': 'obj', 'H': 'psf', 'I': 'ilm', 'B': 'bkg'
        }
        modelstr = {
            'full': 'H(I*P) + B',
            'dP': 'H(I*dP)',
            'dS': 'H(dI*P)',
            'dB': 'dB',
        }
        registry = {
            'psf': allpsfs,
            'obj': allobjs,
            'bkg': allfields,
            'ilm': allfields,
        }
        Model.__init__(self, modelstr=modelstr, varmap=varmap, registry=registry)

class BrightfieldImageModel(Model):
    """
    Brightfield microscope image with simplifications made for performance.
    In particular, the illumination from the condenser is presumed to be
    Koehler and therefore slowly-varying compared to the PSF.

    .. math::
        \\mathcal{M} = I*(1+c*H(P)) + B
    """
    def __init__(self):
        varmap = {
            'B': 'bkg', 'I': 'ilm', 'H': 'psf', 'P': 'obj', 'c':'contrast'
        }
        modelstr = {
            'full' :'I*(1+c*H(P)) + B',
            'dI' : 'dI*(1+c*H(P))',
            'dP' : 'I*c*H(dP)',
            'dc' : 'I*dc*H(P)',
            'dB' : 'dB',
        }
        registry = {
            'bkg': allfields,
            'ilm': allfields,
            'psf': allpsfs,  #FIXME should be 2d psfs
            'obj': allobjs,
            'contrast': {'const': GlobalScalar}
        }
        super(BrightfieldImageModel, self).__init__(modelstr=modelstr, varmap=varmap)#registry=registry)

class IlluminatedParticlesModel(Model):
    """
    A model of particles illuminated by light, with a background.

    .. math::
        \\mathcal{M} = I * P + B
    """
    def __init__(self):
        varmap = {'P': 'obj', 'I':'ilm', 'B':'bkg'}
        modelstr = {
            'full' : 'I*P + B',
            'dI' : 'dI*P',
            'dP' : 'I*dP',
            'dB' : 'dB',
        }
        registry = {
            'bkg': allfields,
            'ilm': allfields,
            'obj': allobjs,
        }
        super(IlluminatedParticlesModel, self).__init__(
                modelstr=modelstr, varmap=varmap, registry=registry)


models = {
    'confocal-dyedfluid': ConfocalImageModel,
    'confocal-dyedobjects': ConfocalDyedParticlesModel,
    'smooth-field': SmoothFieldModel,
    'blurred-field': BlurredFieldModel
}

