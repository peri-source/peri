from builtins import range, str, object
from future.utils import iteritems

import re
import inspect
from operator import add
from collections import OrderedDict, defaultdict
from functools import reduce

from peri import util

class NotAParameterError(Exception):
    pass

#=============================================================================
# A base class for parameter groups (components and priors)
#=============================================================================
class ParameterGroup(object):
    def __init__(self, params=None, values=None, ordered=True, category='param'):
        """
        Any object which computes something based on parameters and values can
        be considered a ``ParameterGroup``. This class provides a common
        interface since ``ParameterGroup`` appears throughout ``PERI``
        including ``Components``, ``Priors``, ``States``. In the very basic
        form, a ``ParameterGroup`` is a ``dict`` or ``OrderedDict`` of::

            { parameter_name: parameter_value, ... }

        The use of a dictionary is strictly optional -- as long as the following
        methods are provided, the parameters and values may be stored in any
        format that is convenient:
        
            * :func:`~peri.comp.comp.ParameterGroup.params`
            * :func:`~peri.comp.comp.ParameterGroup.values`
            * :func:`~peri.comp.comp.ParameterGroup.get_values`
            * :func:`~peri.comp.comp.ParameterGroup.set_values`
            * :func:`~peri.comp.comp.ParameterGroup.update`

        Parameters
        ----------
        params : string, list of strings
            The names of the parameters, in the proper order

        values : number, list of numbers
            The values corresponding to the parameter names

        ordered : boolean (default: True)
            If True, uses an OrderedDict so that parameter order is
            deterministic independent of number of parameters

        category : string (default: 'param')
            Name of the category associated with this ParameterGroup.
            
            .. warning::
            
                FIXME : should only be a property of Component
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

        self.category = category
        self.ordered = ordered

    def update(self, params, values):
        """
        Update the calculation of the class based on a pair or pairs
        of parameters and associated values.

        Parameters
        ----------
        params : string, list of strings
            name of parameters to update

        values : number, list of numbers
            cooresponding values to update
        """
        self.set_values(params, values)
        return True

    def get_values(self, params):
        """
        Get the value of a list or single parameter.

        Parameters
        ----------
        params : string, list of string
            name of parameters which to retrieve
        """
        return util.delistify(
            [self.param_dict[p] for p in util.listify(params)], params
        )

    def set_values(self, params, values):
        """
        Directly set the values corresponding to certain parameters.
        This does not necessarily trigger and update of the calculation,
        
        See also
        --------
        :func:`~peri.comp.comp.ParameterGroup.update` : full update func
        """
        for p, v in zip(util.listify(params), util.listify(values)):
            self.param_dict[p] = v

    @property
    def params(self):
        """ The list of parameters """
        return list(self.param_dict.keys())

    @property
    def values(self):
        """ The list of values """
        return list(self.param_dict.values())

    def nopickle(self):
        """
        Elements of the class that should not be included in pickled objects.
        If inheriting a new class, should be::

            super(Class, self).nopickle() + ['other1', 'other2', ...]

        Returns
        -------
        elements : list of strings
            The name of class member variables which should not be pickled
        """
        return []

    def initargs(self):
        """
        Pickling helper method which returns a dictionary of function
        parameters which get passed to pickle via `__getinitargs__
        <https://docs.python.org/2/library/pickle.html#object.__getinitargs__>`_
        
        Returns
        -------
        arg_dict : dictionary
            ``**kwargs`` to be passed to the __init__ func after unpickling
        """
        return {"params": self.params, "values": self.values, "ordered": self.ordered}

    def __str__(self):
        return "{} [{}]".format(self.__class__.__name__, self.param_dict)

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return self.initargs()

    def __setstate__(self, idct):
        self.__init__(**idct)


#=============================================================================
# Component class == model components for an image
#=============================================================================
class Component(ParameterGroup, util.CompatibilityPatch):
    def __init__(self, params, values, ordered=True, category='comp'):
        """
        A :class:`~peri.comp.comp.ParameterGroup` which specifically computes
        over sections of an image for an :class:`~peri.states.ImageState`. To
        this end, we require the implementation of several new member functions:

            * :func:`~peri.comp.comp.Component.initialize`
            * :func:`~peri.comp.comp.Component.get_update_tile`
            * :func:`~peri.comp.comp.Component.get_padding_size`
            * :func:`~peri.comp.comp.Component.set_shape`
            * :func:`~peri.comp.comp.Component.set_tile`
            * :func:`~peri.comp.comp.Component.get`

        In order to facilitate optimizations such as caching and local updates,
        we must incorporate tiling in this object. 

        Parameters
        ----------
        params : string, list of strings
            The names of the parameters, in the proper order

        values : number, list of numbers
            The values corresponding to the parameter names

        ordered : boolean (default: True)
            If True, uses an OrderedDict so that parameter order is
            deterministic independent of number of parameters

        category : string (default: 'param')
            Name of the category associated with this ParameterGroup.
            
        """
        for attr in ['shape', 'inner', '_parent']:
            if not hasattr(self, attr):
                setattr(self, attr, None)  #Not sure if this is the best, since inner and shape are related
        super(Component, self).__init__(
            params, values, ordered=ordered, category=category
        )

    def initialize(self):
        """ Begin anew and initialize the component """
        pass

    def get_update_tile(self, params, values):
        """

        This method returns a :class:`~peri.util.Tile` object defining the
        region of a field that has to be modified by the update of (params,
        values). For example, if this Component is the point-spread-function,
        it might return a tile of entire image since every parameter affects
        the entire image::

            return self.shape

        

        Parameters
        -----------
        params : single param, list of params
            A single parameter or list of parameters to be updated

        values : single value, list of values
            The values corresponding to the params

        Returns
        -------
        tile : :class:`~peri.util.Tile`
            A tile corresponding to the image region
        """
        pass

    def get_padding_size(self, tile):
        """
        Get the amount of padding required for this object when calculating
        about a tile `tile`. Padding size is the total size, so half that
        on each side. For example, if this Component is a Gaussian point spread
        function, then the padding returned might be::

            peri.util.Tile(np.ceil(2*self.sigma))

        Parameters
        -----------
        tile : :class:`~peri.util.Tile`
            A tile defining the region of interest

        Returns
        -------
        pad : :class:`~peri.util.Tile`
            A tile corresponding to the required padding size
        """
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
        if self.shape != shape or self.inner != inner:
            self.shape = shape
            self.inner = inner
            self.initialize()

    def execute(self, *args, **kwargs):
        """ Perform its routine, whatever that may be """
        pass

    def get(self):
        """
        Return the `natural` part of the model. In the case of most elements it
        is the calculated field, for others it is the object itself.
        """
        pass

    # functions that allow better handling of component collections
    def exports(self):
        """ Which methods a class wants to expose to parent classes """
        return []

    def nopickle(self):
        """ Class attributes which should not be included in a pickle object """
        return ['_parent']

    def register(self, obj):
        """ Registery a parent object so that communication maybe happen upwards """
        self._parent = obj

    def trigger_parameter_change(self):
        """ Notify parents of a parameter change """
        if self._parent:
            self._parent.trigger_parameter_change()

    def trigger_update(self, params, values):
        """ Notify parent of a parameter change """
        if self._parent:
            self._parent.trigger_update(params, values)
        else:
            self.update(params, values)

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

class GlobalScalar(Component):
    def __init__(self, name, value, shape=None):
        self.shape = shape
        super(GlobalScalar, self).__init__(
            [name], [value], ordered=False, category=name
        )

    def initargs(self):
        return {
            'name': self.params[0], 'value': self.values[0],
            'shape': self.shape
        }

    def get(self):
        return self.values[0]

    def get_update_tile(self, params, values):
        return self.shape

#=============================================================================
# Component class == model components for an image
#=============================================================================
def reduce_add(x):
    return reduce(add, x)

def reduce_mul(x):
    return reduce(mul, x)

class ComponentCollection(Component):
    def __init__(self, comps, field_reduce_func=None, category='comp'):
        """
        Group a number of components into a single coherent object which a
        single interface for each function. Obvious reductions are performed in
        places such as get_update_tile which takes the bounding tile of all
        constituent get_update_tile.  The only reduction which is not
        straight-forward is get, but by default it adds all fields. This
        class has the same interface as Component itself.

        Parameters
        -----------
        comps : list of :class:`~peri.comp.comp.Component`
            The components to group together

        field_reduce_func : function(list of ndarrays)
            Reduction function for get object of collection, what happens when
            all comps are reduced to a single field (etc). For example, nested
            point spread functions may want convolution to be the reduce func::

                def reduce_conv(psfs):
                    ffts = np.array([fft.fftn(p) for p in psfs])
                    return fft.ifftn(np.prod(ffts, axis=0))

            Or, if it is just a simple field that needs to be added together::

                def reduce_add(x):
                    return reduce(add, x)

        category : string
            Name of the category associated with this ``ComponentCollection``
        """
        self.comps = comps
        self.category = category

        if not field_reduce_func:
            field_reduce_func = reduce_add
        self.field_reduce_func = field_reduce_func

        self.setup_params()
        self.setup_passthroughs()
        self._parent = None

    def initargs(self):
        """ Return arguments that are passed to init to setup the class again """
        return {
            'comps': self.comps,
            'category': self.category,
            'field_reduce_func': self.field_reduce_func,
        }

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
        self.comps. For example, given a sphere collection and slab::

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
        return util.delistify(vals, params)

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
        return list(pv.keys())

    @property
    def values(self):
        pv = OrderedDict()
        for c in self.comps:
            for p,v in zip(c.params, c.values):
                pv[p] = v
        return list(pv.values())

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

    def set(self, name, obj):
        for i, c in enumerate(self.comps):
            if c.category == name:
                self.comps[i] = obj
        self.trigger_parameter_change()

    def get(self):
        """ Combine the fields from all components """
        fields = [c.get() for c in self.comps]
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
        def _normalize(comps, param):
            vals = [c.get_values(param) for c in comps]
            diff = any([vals[i] != vals[i+1] for i in range(len(vals)-1)])

            if diff:
                for c in comps:
                    c.set_values(param, vals[0])

        for param, comps in iteritems(self.lmap):
            if isinstance(comps, list) and len(comps) > 1:
                _normalize(comps, param)

    def trigger_parameter_change(self):
        self.setup_params()
        self.setup_passthroughs()

        if self._parent:
            self._parent.trigger_parameter_change()

    def setup_passthroughs(self):
        """
        Inherit some functions from the components that we own. In particular,
        let's grab all functions that begin with `param_` so the super class
        knows how to get parameter groups. Also, take anything that is listed
        under Component.exports and rename with the category type, i.e.,
        SphereCollection.add_particle -> Component.obj_add_particle
        """
        self._nopickle = []

        for c in self.comps:
            # take all member functions that start with 'param_'
            funcs = inspect.getmembers(c, predicate=inspect.ismethod)
            for func in funcs:
                if func[0].startswith('param_'):
                    setattr(self, func[0], func[1])
                    self._nopickle.append(func[0])

            # add everything from exports
            funcs = c.exports()
            for func in funcs:
                newname = c.category + '_' + func.__func__.__name__
                setattr(self, newname, func)
                self._nopickle.append(newname)

    def exports(self):
        return [i for c in self.comps for i in c.exports()]

    def nopickle(self):
        return self._nopickle + super(ComponentCollection, self).nopickle()

    def __str__(self):
        def _pad(s):
            return re.subn('(\n)', '\n    ', s)[0]

        return "{} [\n    {}\n]".format(self.__class__.__name__,
            _pad('\n'.join([c.category+': '+str(c) for c in self.comps]))
        )

    def __repr__(self):
        return self.__str__()

util.patch_docs(Component, ParameterGroup)
util.patch_docs(GlobalScalar, Component)
util.patch_docs(ComponentCollection, Component)

