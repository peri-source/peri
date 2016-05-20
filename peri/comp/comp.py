from collections import OrderedDict, defaultdict

from peri.util import listify, delistify

#=============================================================================
# A base class for parameter groups (components and priors)
#=============================================================================
class ParameterGroup(object):
    category = 'param'

    def __init__(self, params=None, values=None):
        """
        Set up a parameter group, which is essentially an OrderedDict of param
        -> values. However, it is generalized so that this structure is not
        strictly enforced for all ParameterGroup subclasses.
        """
        if params is not None and values is not None:
            self._params = OrderedDict()
            for p,v in zip(params, values):
                self._params[p] = v
        elif params is not None and values is None:
            self._params = params
        else:
            self._params = OrderedDict()

    def update(self, params, values):
        """
        Update the a single (param, value) combination, or a list or tuple of
        params and corresponding values for the object.
        """
        self.set_values(params, values)

    def get_values(self, params):
        """ Get the value of a list or single parameter """
        values = delistify([self._params[p] for p in listify(params)])
        return values

    def set_values(self, params, values):
        """
        Directly set a single (param, value) combination, or a list or tuple of
        params and corresponding values for the object.
        """
        for p, v in zip(listify(params), listify(values)):
            self._params[p] = v

    @property
    def params(self):
        return self._params.keys()

    @property
    def values(self):
        return self._params.values()


#=============================================================================
# Component class == model components for an image
#=============================================================================
class Component(ParameterGroup):
    def __init__(self, params, values):
        super(Component, self).__init__(params, values)

    def get_support_size(self, params, values):
        """
        This method returns a `peri.util.Tile` object defining the region of
        a field that has to be modified by the update of (params, values).

        Parameters:
        -----------
        params : single param, list of params
            A single parameter or list of parameters to be updated

        values : single value, list of values
            The values corresponding to the params

        Returns:
        --------
        tile : `peri.util.Tile`
            A tile corresponding to the image region
        """
        raise NotImplementedError("get_support_size required for components")

    def get_padding_size(self, params, values):
        raise NotImplementedError("get_padding_size required for components")

    def get_field(self):
        raise NotImplementedError("get_field required for components")

    def set_tile(self, tile):
        raise NotImplementedError("set_tile required for components")

    # TODO make all components serializable via _getinitargs_

#=============================================================================
# Component class == model components for an image
#=============================================================================
class ComponentCollection(Component):
    def __init__(self, comps):
        comps = comps
        pmap = defaultdict(set)

        for c in comps:
            for p in c.params:
                pmap[p].update([c])

        self.comps = comps
        self.pmap = pmap

    def split_params(self, params, values=None):
        pc, vc = [], []

        returnvalues = values is not None
        if values is None:
            values = [0]*len(listify(params))

        for c in self.comps:
            tp, tv = [], []
            for p,v in zip(listify(params), listify(values)):
                if c in self.pmap[p]:
                    tp.append(p)
                    tv.append(v)

            pc.append(tp)
            vc.append(tv)

        if returnvalues:
            return pc, vc
        return pc

    def update(self, params, values):
        plist, vlist = self.split_params(params, values)
        for c, p, v in zip(self.comps, plist, vlist):
            c.update(plist, vlist)

    def get_values(self, params):
        values = delistify([self._params[p] for p in listify(params)])
        return values

    def set_values(self, params, values):
        for p, v in zip(listify(params), listify(values)):
            self._params[p] = v

    @property
    def params(self):
        return self._params.keys()

    @property
    def values(self):
        return self._params.values()

    def get_support_size(self, params, values):
        raise NotImplementedError("get_support_size required for components")

    def get_padding_size(self, params, values):
        raise NotImplementedError("get_padding_size required for components")

    def get_field(self):
        raise NotImplementedError("get_field required for components")

    def set_tile(self, tile):
        raise NotImplementedError("set_tile required for components")

