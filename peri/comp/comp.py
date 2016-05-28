import re
import inspect
from operator import add
from collections import OrderedDict, defaultdict

from peri import util

class NotAParameterError(Exception):
    pass

#=============================================================================
# A base class for parameter groups (components and priors)
#=============================================================================
class ParameterGroup(object):
    category = 'param'

    def __init__(self, params=None, values=None, ordered=True):
        """
        Set up a parameter group, which is essentially an OrderedDict of param
        -> values. However, it is generalized so that this structure is not
        strictly enforced for all ParameterGroup subclasses.
        """
        gen = OrderedDict if ordered else dict

        if params is not None and values is not None:
            self.param_dict = gen()
            for p,v in zip(params, values):
                self.param_dict[p] = v
        elif params is not None and values is None:
            self.param_dict = params
        else:
            self.param_dict = gen()

    def update(self, params, values):
        """
        Update the a single (param, value) combination, or a list or tuple of
        params and corresponding values for the object.
        """
        self.set_values(params, values)
        return True

    def get_values(self, params):
        """ Get the value of a list or single parameter """
        values = util.delistify([self.param_dict[p] for p in util.listify(params)])
        return values

    def set_values(self, params, values):
        """
        Directly set a single (param, value) combination, or a list or tuple of
        params and corresponding values for the object.
        """
        for p, v in zip(util.listify(params), util.listify(values)):
            self.param_dict[p] = v

    @property
    def params(self):
        return self.param_dict.keys()

    @property
    def values(self):
        return self.param_dict.values()

    def __str__(self):
        return "{} [{}]".format(self.__class__.__name__, self.param_dict)

    def __repr__(self):
        return self.__str__()

    # Functions that begin with param_ will be passed through to the states and
    # other functions so that nice interfaces are present
    # def param_particle_positions(self):
    #   return parameters
    #
    # Also, functions that start with (unknown) will get passed through too

#=============================================================================
# Component class == model components for an image
#=============================================================================
class Component(ParameterGroup):
    category = 'comp'

    def __init__(self, params, values, ordered=True):
        self._parent = None
        super(Component, self).__init__(params, values, ordered=ordered)

    def initialize(self):
        """ Begin anew and initialize the component """
        pass

    def get_update_tile(self, params, values):
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
        pass

    def get_padding_size(self, tile):
        """
        Get the amount of padding required for this object when calculating
        about a tile `tile`. Padding size is the total size, so twice that
        on each side.

        Parameters:
        -----------
        tile : `peri.util.Tile`
            A tile defining the region of interest
        """
        pass

    def get_field(self):
        """ Return the current field as determined by self.set_tile """
        pass

    def set_tile(self, tile):
        """ Set the currently active tile region for the calculation """
        self.tile = tile

    def set_shape(self, shape, inner):
        """
        Set the overall shape of the calculation area. The total shape of that
        the calculation can possibly occupy, in pixels. The second, inner, is
        the region of interest within the image.
        """
        self.shape = shape
        self.inner = inner
        self.initialize()

    def execute(self, field):
        """ Perform its routine, whatever that may be """
        pass

    def get(self):
        """
        Return the `natural` part of the model. In the case of most elements it
        is the get_field method, for others it is the object itself.
        """
        return self.get_field()

    # functions that allow better handling of component collections
    def exports(self):
        return []

    def register(self, obj):
        self._parent = obj

    def trigger_parameter_change(self):
        if self._parent:
            self._parent.trigger_parameter_change()

    def trigger_update(self, params, values):
        if self._parent:
            self._parent.update(params, values)

    def __call__(self, field):
        return self.execute(field)

    # TODO make all components serializable via _getinitargs_

class GlobalScalar(Component):
    category = 'scalar'

    def __init__(self, name, value, shape=None):
        self.shape = shape
        self.category = name
        super(GlobalScalar, self).__init__([name], [value], ordered=False)

    def get(self):
        return self.values[0]

    def get_field(self):
        return self.values[0]

    def get_update_tile(self, params, values):
        return self.shape

#=============================================================================
# Component class == model components for an image
#=============================================================================
class ComponentCollection(Component):
    def __init__(self, comps, field_reduce_func=None, category=None):
        """
        Group a number of components into a single coherent object which a
        single interface for each function. Obvious reductions are performed in
        places such as get_update_tile which takes the bounding tile of all
        constituent get_update_tile.  The only reduction which is not
        straight-forward is get_field, but by default it adds all fields. This
        class has the same interface as Component itself.

        Parameters:
        -----------
        comps : list of `peri.comp.Component`
            The components to group together

        field_reduce_func : function (list of ndarrays)
            Reduction function for get_field object of collection
        """
        self.comps = comps
        self.category = category

        if not field_reduce_func:
            field_reduce_func = lambda x: reduce(add, x)
        self.field_reduce_func = field_reduce_func

        self.setup_params()
        self._passthrough_func()

    def initialize(self):
        for c in self.comps:
            c.initialize()

    def setup_params(self):
        pmap = defaultdict(set)
        lmap = defaultdict(list)

        for c in self.comps:
            c.register(self)

            if not isinstance(c, (Component, ComponentCollection)):
                raise AttributeError("%r is not a valid Component or ComponentCollection" % c)
            for p in c.params:
                pmap[p].update([c])
                lmap[p].extend([c])

        self.pmap = pmap
        self.lmap = lmap
        self.sync_params()

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
            values = [0]*len(util.listify(params))

        for c in self.comps:
            tp, tv = [], []
            for p,v in zip(util.listify(params), util.listify(values)):
                if not p in self.lmap:
                    raise NotAParameterError("%r does not belong to %r" % (p, self))

                if c in self.pmap[p]:
                    tp.append(p)
                    tv.append(v)

            pc.append(tp)
            vc.append(tv)

        if returnvalues:
            return pc, vc
        return pc

    def affected_components(self, params):
        comps = []
        plist = self.split_params(params)
        for c, p in zip(self.comps, plist):
            if len(p) > 0:
                comps.append(c)
        return comps

    def update(self, params, values):
        plist, vlist = self.split_params(params, values)
        for c, p, v in zip(self.comps, plist, vlist):
            if len(p) > 0:
                c.update(p, v)
        return True

    def get_values(self, params):
        vals = []
        for p in util.listify(params):
            if not p in self.lmap:
                raise NotAParameterError("%r does not belong to %r" % (p, self))
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

    def get_update_tile(self, params, values):
        sizes = []

        plist, vlist = self.split_params(params, values)
        for c, p, v in zip(self.comps, plist, vlist):
            if len(p) > 0:
                tile = c.get_update_tile(p, v)
                if tile is not None:
                    sizes.append(tile)

        if len(sizes) == 0:
            return None
        return util.Tile.boundingtile(sizes)

    def get_padding_size(self, tile):
        sizes = []

        for c in self.comps:
            pad = c.get_padding_size(tile)
            if pad is not None:
                sizes.append(pad)

        if len(sizes) == 0:
            return None
        return util.Tile.boundingtile(sizes)

    def get_field(self):
        """ Combine the fields from all components """
        fields = [c.get_field() for c in self.comps]
        return self.field_reduce_func(fields)

    def set_tile(self, tile):
        """ Set the current working tile for components """
        for c in self.comps:
            c.set_tile(tile)

    def set_shape(self, shape, inner):
        """ Set the shape for all components """
        for c in self.comps:
            c.set_shape(shape, inner)

    def sync_params(self):
        """ Ensure that shared parameters are the same value everywhere """
        pass # FIXME

    def trigger_parameter_change(self):
        self.setup_params()

    def _passthrough_func(self):
        """
        Inherit some functions from the components that we own. In particular,
        let's grab all functions that begin with `param_` so the super class
        knows how to get parameter groups. Also, take anything that is listed
        under Component.exports and rename with the category type, i.e.,
        SphereCollection.add_particle -> Component.obj_add_particle
        """
        for c in self.comps:
            # take all member functions that start with 'param_'
            funcs = inspect.getmembers(c, predicate=inspect.ismethod)
            for func in funcs:
                if func[0].startswith('param_'):
                    setattr(self, func[0], func[1])

            # add everything from exports
            funcs = c.exports()
            for func in funcs:
                newname = c.category + '_' + func.im_func.func_name
                setattr(self, newname, func)

    def exports(self):
        return [i for c in self.comps for i in c.exports()]

    def __str__(self):
        def _pad(s):
            return re.subn('(\n)', '\n   ', s)[0]

        return "{} [\n    {}\n]".format(self.__class__.__name__, 
            _pad('\n'.join([c.category+': '+str(c) for c in self.comps]))
        )

    def __repr__(self):
        return self.__str__()

util.patch_docs(GlobalScalar, Component)
util.patch_docs(ComponentCollection, Component)

