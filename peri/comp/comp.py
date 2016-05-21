from operator import add
from collections import OrderedDict, defaultdict

from peri.util import listify, delistify, Tile

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
            self.param_dict = OrderedDict()
            for p,v in zip(params, values):
                self.param_dict[p] = v
        elif params is not None and values is None:
            self.param_dict = params
        else:
            self.param_dict = OrderedDict()

    def update(self, params, values):
        """
        Update the a single (param, value) combination, or a list or tuple of
        params and corresponding values for the object.
        """
        self.set_values(params, values)

    def get_values(self, params):
        """ Get the value of a list or single parameter """
        values = delistify([self.param_dict[p] for p in listify(params)])
        return values

    def set_values(self, params, values):
        """
        Directly set a single (param, value) combination, or a list or tuple of
        params and corresponding values for the object.
        """
        for p, v in zip(listify(params), listify(values)):
            self.param_dict[p] = v

    @property
    def params(self):
        return self.param_dict.keys()

    @property
    def values(self):
        return self.param_dict.values()

    # Functions that begin with block_ will be passed through to the states and
    # other functions so that nice interfaces are present
    # def block_particle_positions(self):
    #   return parameters
    #
    # Also, functions that start with (unknown) will get passed through too

#=============================================================================
# Component class == model components for an image
#=============================================================================
class Component(ParameterGroup):
    def __init__(self, params, values):
        super(Component, self).__init__(params, values)

    def initialize(self):
        """ Begin anew and initialize the component """
        raise NotImplementedError("initialize required for components")

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
    def __init__(self, comps, field_reduce_func=None):
        """
        Group a number of components into a single coherent object which a
        single interface for each function. Obvious reductions are performed in
        places such as get_support_size which takes the bounding tile of all
        constituent get_support_size.  The only reduction which is not
        straight-forward is get_field, but by default it adds all fields. This
        class has the same interface as Component itself.

        Parameters:
        -----------
        comps : list of `peri.comp.Component`
            The components to group together

        field_reduce_func : function (list of ndarrays)
            Reduction function for get_field object of collection
        """
        comps = comps
        pmap = defaultdict(set)
        lmap = defaultdict(list)

        for c in comps:
            for p in c.params:
                pmap[p].update([c])
                lmap[p].extend([c])

        self.comps = comps
        self.pmap = pmap
        self.lmap = lmap

        if not field_reduce_func:
            field_reduce_func = lambda x: reduce(add, x)
        self.field_reduce_func = field_reduce_func

    def initialize(self):
        for c in self.comps:
            c.initialize()

    def split_params(self, params, values=None):
        """
        Split params, values into groups that correspond to the ordering in
        self.comps. For example, given a sphere collection and slab,

        [
            (spheres) [pos rad etc] [pos val, rad val, etc]
            (slab) [slab params] [slab vals]
        ]
        """
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
            c.update(p, v)

    def get_values(self, params):
        vals = []
        for p in listify(params):
            vals.append(self.lmap[p][0].get_values(p))
        return vals

    def set_values(self, params, values):
        plist, vlist = self.split_params(params, values)
        for c, p, v in zip(self.comps, plist, vlist):
            c.set_values(p, v)

    @property
    def params(self):
        pv = OrderedDict()
        for c in self.comps:
            for p,v in zip(c.params, c.values):
                pv[p] = v
        return pv.keys()

    @property
    def values(self):
        pv = OrderedDict()
        for c in self.comps:
            for p,v in zip(c.params, c.values):
                pv[p] = v
        return pv.values()

    def get_support_size(self, params, values):
        sizes = []

        plist, vlist = self.split_params(params, values)
        for c, p, v in zip(self.comps, plist, vlist):
            sizes.append(c.get_support_size(p, v))

        return Tile.boundingtile(sizes)

    def get_padding_size(self, params, values):
        sizes = []

        plist, vlist = self.split_params(params, values)
        for c, p, v in zip(self.comps, plist, vlist):
            sizes.append(c.get_padding_size(p, v))

        return Tile.boundingtile(sizes)

    def get_field(self):
        fields = [c.get_field() for c in self.comps]
        return self.field_reduce_func(fields)

    def set_tile(self, tile):
        for c in self.comps:
            c.set_tile(tile)

