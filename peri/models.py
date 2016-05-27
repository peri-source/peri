import re

class ModelError(Exception):
    pass

class Model(object):
    def __init__(self, modelstr, varmap):
        """
        An abstraction for defining how to combine components into a complete
        model as well as derivatives with respect to different variables.

        Parameters:
        -----------
        modelstr : dict of strings
            (Using a custom model) The actual equations used to defined the model.
            At least one eq. is required under the key `full`. Other partial
            update equations can be added where the variable name (e.g. 'dI')
            denotes a partial update of I, for example:

                {'full': 'H(P)', 'dP': 'H(dP)'}

        varmap : dict
            (Using a custom model) The mapping of variables in the modelstr
            equations to actual component types. For example:

                {'P': 'obj', 'H': 'psf'}
        """
        self.modelstr = modelstr
        self.varmap = varmap
        self.ivarmap = {v:k for k,v in self.varmap.iteritems()}
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
        if not self.modelstr.has_key('full'):
            raise ModelError(
                'Model must contain a `full` key describing '
                'the entire image formation'
            )

        # Check that the two model descriptors are consistent
        for name, eq in self.modelstr.iteritems():
            var = regex.findall(eq)
            for v in var:
                # remove the derivative signs if there (dP -> P)
                v = re.sub(r"^d", '', v)
                if v not in self.varmap:
                    log.error(
                        "Variable '%s' (eq. '%s': '%s') not found in category map %r" %
                        (v, name, eq, varmap)
                    )
                    error = True

        if error:
            raise ModelError('Inconsistent catmap and model descriptions')

    def check_inputs(self, comps):
        error = False
        compcats = [c.category for c in comps]

        # Check that the components are all provided, given the categories
        for k,v in self.varmap.iteritems():
            if k not in self.modelstr['full']:
                log.warn('Component (%s : %s) not used in model.' % (k,v))

            if not v in compcats:
                log.error('Map component (%s : %s) not found in list of components.' % (k,v))
                error = True

        if error:
            raise ModelError('Component list incomplete or incorrect')

    def diffname(self, name):
        return 'd'+name

    def get_base_model(self):
        """ The complete model, no variation """
        return self.modelstr['full']

    def get_difference_model(self, category):
        """ Get the equation corresponding to a variation wrt category """
        name = self.diffname(self.ivarmap[category])
        return self.modelstr.get(name)

    def map_vars(self, comps, funcname, diffvar=None, **kwargs):
        """
        Map component function `funcname` result into model variables
        dictionary for use in eval of the model. If diffvar is provided then
        that symbol is translated into 'd'+symbol. **kwargs are passed
        to the comp.funcname(**kwargs).
        """
        out = {}

        for c in comps:
            cat = c.category

            if cat == diffvar:
                symbol = self.diffname(self.ivarmap[cat])
            else:
                symbol = self.ivarmap[cat]

            out[symbol] = getattr(c, funcname)(**kwargs)

        return out


class ConfocalImageModel(Model):
    def __init__(self):
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
        Model.__init__(self, modelstr=modelstr, varmap=varmap)

class ParticlesModel(Model):
    def __init__(self):
        varmap = {
            'P': 'obj', 'H': 'psf', 'C': 'offset'
        }
        modelstr = {
            'full': 'H(P) + C',
            'dP': 'H(dP)',
            'dC': 'dC'
        }
        Model.__init__(self, modelstr=modelstr, varmap=varmap)


