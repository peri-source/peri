from builtins import range, str

import numpy as np
from scipy.special import erf

try:
    from scipy.weave import inline
except ImportError as e:
    try:
        from weave import inline
    except ImportError as e:
        pass

from peri.special import functions
from peri.comp import Component
from peri.util import Tile, cdd, listify, delistify

# maximum number of iterations to get an exact volume
MAX_VOLUME_ITERATIONS = 10


#=============================================================================
# Superclass for collections of particles
#=============================================================================

class PlatonicParticlesCollection(Component):
    def __init__(self, pos, shape=None, param_prefix='sph', category='obj',
                support_pad=4, float_precision=np.float64):
        """
        Parent class for a large collection of particles, such as spheres or
        points or ellipsoids or rods.

        This class is good for a collection of objects which each have a
        position as well as (possibly) some other parameters, like particle
        radius, aspect ratio, or brightness. Its .get() method returns a
        field of the drawn particles, selected on the current tile. Any
        daughter classes need the following methods:

        * _draw_particle
        * _update_type
        * setup_variables
        * get_values
        * set_values
        * add_particle
        * remove_particle

        In addition, the following methods should be modified for particles
        with more parameters than just positions:

        * _drawargs
        * _tile
        * param_particle
        * exports
        * _p2i

        If you have a few objects to group, like 2 or 3 slabs, group them
        with a `peri.comp.ComponentCollection` instead.

        Parameters
        ----------
        pos : ndarray [N,d]
            Initial positions of the particles. Re-cast as float internally

        shape : ``peri.util.Tile``, optional
            Shape of the field over which to draw the platonic spheres.
            Default is None.

        param_prefix : string, optional
            Prefix for the particle parameter names. Default is `'sph'`

        category : string, optional
            Category, as in comp.Component. Default is `'obj'`.

        support_pad : Int, optional
            How much to pad the boundary of particles when calculating the
            support so that particles do not leak out the edges. Default is 4

        float_precision : numpy float datatype, optional
            One of numpy.float16, numpy.float32, numpy.float64; precision
            for precomputed arrays. Default is np.float64; make it 16 or 32
            to save memory.
        """
        if pos.ndim != 2:
            raise ValueError('pos must be of shape (N,d)')

        self.category = category
        self.support_pad = support_pad
        self.pos = pos.astype('float')
        self.param_prefix = param_prefix

        if float_precision not in (np.float64, np.float32, np.float16):
            raise ValueError('float_precision must be one of np.float64, ' +
                    'np.float32, np.float16')
        self.float_precision = float_precision

        self.shape = shape
        self.setup_variables()

        if self.shape:
            self.inner = self.shape.copy()
            self.tile = self.inner.copy()
            self.initialize()

    def _draw_particle(self, pos, sign=1):
        """
        Updates ``self.particles`` by drawing a particle at position ``pos``,
        with possible additional unnamed arguments between ``pos`` and
        ``sign``. If ``sign`` is -1, un-draws the particle instead.

        To be able to fit this component in a model, _draw_particle must
        create an image that is numerically continuous as pos changes --
        i.e. the edge of the particle must alias smoothly to 0.
        """
        raise NotImplementedError('Implement in subclasss')

    def _update_type(self, params):
        """
        Given a list of parameters, returns a bool of whether or not any of
        the parameters require a global update, and a list of particle indices
        which are included in ``params``, e.g.
        ``return doglobal, particles``
        """
        raise NotImplementedError('Implement in subclasss')

    def setup_variables(self):
        """Creates an ordered list of parameters and stores in self._params"""
        raise NotImplementedError('Implement in subclass')

    def get_values(self, params):
        """
        Returns a util.delisty-d assortment of values and parameters, e.g.
        (get values for the parameters, both particle positions and globals)
        (return delistify(values, params))
        """
        #FIXME seems stupid that this can't be done intelligently, like with
        #a Component's dict
        raise NotImplementedError('Implement in subclasss')

    def set_values(self, params, values):
        """Sets the parameters and values"""
        #FIXME seems stupid that this can't be done intelligently, like with
        #a Component's dict
        raise NotImplementedError('Implement in subclasss')

    def add_particle(pos):
        raise NotImplementedError('Implement in subclasss')

    def remove_particle(pos):
        raise NotImplementedError('Implement in subclasss')


    def _drawargs(self):
        """
        Returns a list of arguments for self._draw_particle, of the same
        length as `self.pos`. For example, if drawing a sphere, _drawargs
        could return a list of radii.
        """
        return [[] for p in self.pos]

    def _tile(self, n):
        """Get the update tile surrounding particle `n` """
        pos = self._trans(self.pos[n])
        return Tile(pos, pos).pad(self.support_pad)

    def param_particle(self, ind):
        return self.param_particle_pos(ind)

    def exports(self):
        return [
            self.add_particle, self.remove_particle, self.closest_particle,
            self.get_positions
        ]

    def _p2i(self, param):
        """
        Parameter to indices, returns (coord, index), e.g. for a pos
        pos     : ('x', 100)
        """
        g = param.split('-')
        if len(g) == 3:
            return g[2], int(g[1])
        else:
            raise ValueError('`param` passed as incorrect format')


    def initialize(self):
        """Start from scratch and initialize all objects / draw self.particles"""
        self.particles = np.zeros(self.shape.shape, dtype=self.float_precision)

        for p0, arg0 in zip(self.pos, self._drawargs()):
            self._draw_particle(p0, *listify(arg0))

    def get(self):
        return self.particles[self.tile.slicer]

    @property
    def N(self):
        return self.pos.shape[0]

    def _vps(self, inds):
        """Clips a list of inds to be on [0, self.N]"""
        return [j for j in inds if j >= 0 and j < self.N]

    def param_positions(self):
        """ Return params of all positions """
        return self.param_particle_pos(list(range(self.N)))

    def param_particle_pos(self, ind):
        """ Get position of one or more particles """
        #FIXME assumes 3D and x,y,z labels right now....
        ind = self._vps(listify(ind))
        return [self._i2p(i, j) for i in ind for j in ['z', 'y', 'x']]

    def _trans(self, pos):
        return pos + self.inner.l

    def get_positions(self):
        return self.pos.copy()

    def closest_particle(self, x):
        """ Get the index of the particle closest to vector `x` """
        return (((self.pos - x)**2).sum(axis=-1)).argmin()

    @property
    def params(self):
        return self._params

    @property
    def values(self):
        return self.get_values(self._params)

    def _i2p(self, ind, coord):
        """ Translate index info to parameter name """
        return '-'.join([self.param_prefix, str(ind), coord])

    def get_update_tile(self, params, values):
        """ Get the amount of support size required for a particular update."""
        doglobal, particles = self._update_type(params)
        if doglobal:
            return self.shape.copy()

        # 1) store the current parameters of interest
        values0 = self.get_values(params)
        # 2) calculate the current tileset
        tiles0 = [self._tile(n) for n in particles]

        # 3) update to newer parameters and calculate tileset
        self.set_values(params, values)
        tiles1 = [self._tile(n) for n in particles]

        # 4) revert parameters & return union of all tiles
        self.set_values(params, values0)
        return Tile.boundingtile(tiles0 + tiles1)

    def update(self, params, values):
        """
        Update the particles field given new parameter values
        """
        #1. Figure out if we're going to do a global update, in which
        #   case we just draw from scratch.
        global_update, particles = self._update_type(params)

        # if we are doing a global update, everything must change, so
        # starting fresh will be faster instead of add subtract
        if global_update:
            self.set_values(params, values)
            self.initialize()
            return

        # otherwise, update individual particles. delete the current versions
        # of the particles update the particles, and redraw them anew at the
        # places given by (params, values)
        oldargs = self._drawargs()
        for n in particles:
            self._draw_particle(self.pos[n], *listify(oldargs[n]), sign=-1)

        self.set_values(params, values)

        newargs = self._drawargs()
        for n in particles:
            self._draw_particle(self.pos[n], *listify(newargs[n]), sign=+1)

    def __str__(self):
        return "{} N={}".format(self.__class__.__name__, self.N)


#=============================================================================
# Forms of the platonic sphere interpolation function
#=============================================================================
def norm(a):
    return np.sqrt((a**2).sum(axis=-1))

def inner(r, p, a, zscale=1.0):
    eps = np.array([1,1,1])*1e-8
    s = np.array([zscale, 1.0, 1.0])

    d = (r-p-eps)*s
    n = norm(d)
    dhat = d / n[...,None]

    o = norm((d - a*dhat)/s)
    return o * np.sign(n - a)

def sphere_bool(dr, a, alpha):
    return 1.0*(dr < 0)

def sphere_lerp(dr, a, alpha):
    """ Linearly interpolate the pixels for the platonic object """
    return (1-np.clip((dr+alpha) / (2*alpha), 0, 1))

def sphere_logistic(dr, a, alpha):
    """ Classic logistic interpolation """
    return 1.0/(1.0 + np.exp(alpha*dr))

def sphere_triangle_cdf(dr, a, alpha):
    """ Cumulative distribution function for the traingle distribution """
    p0 = (dr+alpha)**2/(2*alpha**2)*(0 > dr)*(dr>-alpha)
    p1 = 1*(dr>0)-(alpha-dr)**2/(2*alpha**2)*(0<dr)*(dr<alpha)
    return (1-np.clip(p0+p1, 0, 1))


def sphere_analytical_gaussian(dr, a, alpha=0.2765):
    """
    Analytically calculate the sphere's functional form by convolving the
    Heavyside function with first order approximation to the sinc, a Gaussian.
    The alpha parameters controls the width of the approximation -- should be
    1, but is fit to be roughly 0.2765
    """
    term1 = 0.5*(erf((dr+2*a)/(alpha*np.sqrt(2))) + erf(-dr/(alpha*np.sqrt(2))))
    term2 = np.sqrt(0.5/np.pi)*(alpha/(dr+a+1e-10)) * (
                np.exp(-0.5*dr**2/alpha**2) - np.exp(-0.5*(dr+2*a)**2/alpha**2)
            )
    return term1 - term2

def sphere_analytical_gaussian_trim(dr, a, alpha=0.2765, cut=1.6):
    """
    See sphere_analytical_gaussian_exact.

    I trimmed to terms from the functional form that are essentially zero (1e-8)
    for r0 > cut (~1.5), a fine approximation for these platonic anyway.
    """
    m = np.abs(dr) <= cut

    # only compute on the relevant scales
    rr = dr[m]
    t = -rr/(alpha*np.sqrt(2))
    q = 0.5*(1 + erf(t)) - np.sqrt(0.5/np.pi)*(alpha/(rr+a+1e-10)) * np.exp(-t*t)

    # fill in the grid, inside the interpolation and outside where values are constant
    ans = 0*dr
    ans[m] = q
    ans[dr >  cut] = 0
    ans[dr < -cut] = 1
    return ans

def sphere_analytical_gaussian_fast(dr, a, alpha=0.2765, cut=1.20):
    """
    See sphere_analytical_gaussian_trim, but implemented in C with
    fast erf and exp approximations found at
        Abramowitz and Stegun: Handbook of Mathematical Functions
        A Fast, Compact Approximation of the Exponential Function

    The default cut 1.25 was chosen based on the accuracy of fast_erf
    """

    code = """
    double coeff1 = 1.0/(alpha*sqrt(2.0));
    double coeff2 = sqrt(0.5/pi)*alpha;

    for (int i=0; i<N; i++){
        double dri = dr[i];
        if (dri < cut && dri > -cut){
            double t = -dri*coeff1;
            ans[i] = 0.5*(1+fast_erf(t)) - coeff2/(dri+a+1e-10) * fast_exp(-t*t);
        } else {
            ans[i] = 0.0*(dri > cut) + 1.0*(dri < -cut);
        }
    }
    """

    shape = r.shape
    r = r.flatten()
    N = self.N
    ans = r*0
    pi = np.pi

    inline(code, arg_names=['dr', 'a', 'alpha', 'cut', 'ans', 'pi', 'N'],
            support_code=functions, verbose=0)
    return ans.reshape(shape)

def sphere_constrained_cubic(dr, a, alpha):
    """
    Sphere generated by a cubic interpolant constrained to be (1,0) on
    (r0-sqrt(3)/2, r0+sqrt(3)/2), the size of the cube in the (111) direction.
    """
    sqrt3 = np.sqrt(3)

    b_coeff = a*0.5/sqrt3*(1 - 0.6*sqrt3*alpha)/(0.15 + a*a)
    rscl = np.clip(dr, -0.5*sqrt3, 0.5*sqrt3)

    a, d = rscl + 0.5*sqrt3, rscl - 0.5*sqrt3
    return alpha*d*a*rscl + b_coeff*d*a - d/sqrt3

try:
    sphere_analytical_gaussian_fast(np.linspace(0,10,10), 5.0)
except Exception as e:
    sphere_analytical_gaussian_fast = sphere_analytical_gaussian_trim

def exact_volume_sphere(rvec, pos, radius, zscale=1.0, volume_error=1e-5,
        function=sphere_analytical_gaussian, max_radius_change=1e-2, args=()):
    """
    Perform an iterative method to calculate the effective sphere that perfectly
    (up to the volume_error) conserves volume.  Return the resulting image
    """
    vol_goal = 4./3*np.pi*radius**3 / zscale
    rprime = radius

    dr = inner(rvec, pos, rprime, zscale=zscale)
    t = function(dr, rprime, *args)
    for i in range(MAX_VOLUME_ITERATIONS):
        vol_curr = np.abs(t.sum())
        if np.abs(vol_goal - vol_curr)/vol_goal < volume_error:
            break

        rprime = rprime + 1.0*(vol_goal - vol_curr) / (4*np.pi*rprime**2)

        if np.abs(rprime - radius)/radius > max_radius_change:
            break

        dr = inner(rvec, pos, rprime, zscale=zscale)
        t = function(dr, rprime, *args)

    return t

#=============================================================================
# Actual sphere collection (and slab)
#=============================================================================
class PlatonicSpheresCollection(PlatonicParticlesCollection):
    def __init__(self, pos, rad, shape=None, zscale=1.0, support_pad=4,
            method='exact-gaussian-fast', alpha=None, user_method=None,
            exact_volume=True, volume_error=1e-5, max_radius_change=1e-2,
            param_prefix='sph', grouping='particle', category='obj',
            float_precision=np.float64):
        """
        A collection of spheres in real-space with positions and radii, drawn
        not necessarily on a uniform grid (i.e. scale factor associated with
        z-direction).  There are many ways to draw the sphere, currently
        supported  methods can be one of::

            [
                'bool', 'lerp', 'logistic', 'triangle', 'constrained-cubic',
                'exact-gaussian', 'exact-gaussian-trim', 'exact-gaussian-fast',
                'user-defined'
            ]

        Parameters
        ----------
        pos : ndarray [N,3]
            Initial positions of the spheres

        rad : ndarray [N] or float
            Initial radii of the spheres

        shape : tuple
            Shape of the field over which to draw the platonic spheres

        zscale : float
            scaling of z-pixels in the platonic image

        support_pad : int
            how much to pad the boundary of particles when calculating
            support so that there is not more contribution

        method : string
            The sphere drawing function to use, see above.

        alpha : float
            Parameter supplied to sphere drawing function, set to value to
            override default value

        user_method : tuple (function, parameters)
            Provide your own sphere function to the drawing method. First
            element of tuple is function with call signature `func(dr, a, *args)`
            where the second element is the `*args` that are not the distance
            to edge (dr) or particles radius (a). `method` must be set to
            'user-defined'.

        exact_volume : boolean
            whether to iterate effective particle size until exact volume
            (within volume_error) is achieved

        volume_error : float
            relative volume error tolerance in iteration steps

        max_radius_change : float
            maximum relative radius change allowed during iteration (due to
            edge particles and other confounding factors)

        grouping : string
            Either 'particle' or 'parameter' parameter grouping. If 'particle'
            then grouped by xyza,xyza if 'parameter' then xyz,xyz,a,a

        float_precision : numpy float datatype
            One of numpy.float16, numpy.float32, numpy.float64; precision
            for precomputed arrays. Default is np.float64; make it 16 or 32
            to save memory.

        """
        if isinstance(rad, (float, int)):
            rad = rad*np.ones(pos.shape[0])
        if rad.size != pos.shape[0]:
            raise ValueError('pos, rad must have the same number of particles.')
        if pos.ndim != 2:
            raise ValueError('pos must be of shape (N,3)')

        self.rad = rad.astype('float')
        self.zscale = zscale
        self.exact_volume = exact_volume
        self.volume_error = volume_error
        self.max_radius_change = max_radius_change
        self.user_method = user_method
        self.grouping = grouping

        self.set_draw_method(method=method, alpha=alpha, user_method=user_method)

        super(PlatonicSpheresCollection, self).__init__(pos=pos, shape=shape,
                param_prefix=param_prefix, category=category, support_pad=
                support_pad, float_precision=float_precision)

    def _drawargs(self):
        return self.rad

    def setup_variables(self):
        self._params = []
        if self.grouping == 'parameter':
            for i, p0 in enumerate(self.pos):
                self._params.extend([self._i2p(i, c) for c in ['z','y','x']])
            for i, r0 in enumerate(self.rad):
                self._params.extend([self._i2p(i, c) for c in ['a']])
        else:
            for i, (p0, r0) in enumerate(zip(self.pos, self.rad)):
                self._params.extend([self._i2p(i, c) for c in ['z','y','x','a']])
        self._params += ['zscale']

    def get_values(self, params):
        values = []
        for p in listify(params):
            typ, ind = self._p2i(p)
            if typ == 'zscale':
                values.append(self.zscale)
            elif typ == 'x':
                values.append(self.pos[ind][2])
            elif typ == 'y':
                values.append(self.pos[ind][1])
            elif typ == 'z':
                values.append(self.pos[ind][0])
            elif typ == 'a':
                values.append(self.rad[ind])
        return delistify(values, params)

    def set_values(self, params, values):
        for p,v in zip(listify(params), listify(values)):
            typ, ind = self._p2i(p)
            if typ == 'zscale':
                self.zscale = v
            elif typ == 'x':
                self.pos[ind][2] = v
            elif typ == 'y':
                self.pos[ind][1] = v
            elif typ == 'z':
                self.pos[ind][0] = v
            elif typ == 'a':
                self.rad[ind] = v

    def set_draw_method(self, method, alpha=None, user_method=None):
        self.methods = [
            'lerp', 'logistic', 'triangle', 'constrained-cubic',
            'exact-gaussian', 'exact-gaussian-trim', 'exact-gaussian-fast',
            'user-defined'
        ]

        self.sphere_functions = {
            'bool': sphere_bool,
            'lerp': sphere_lerp,
            'logistic': sphere_logistic,
            'triangle': sphere_triangle_cdf,
            'exact-gaussian': sphere_analytical_gaussian,
            'exact-gaussian-trim': sphere_analytical_gaussian_trim,
            'exact-gaussian-fast': sphere_analytical_gaussian_fast,
            'constrained-cubic': sphere_constrained_cubic
        }

        self.alpha_defaults = {
            'bool': 0,
            'lerp': 0.4539,
            'logistic': 6.5,
            'triangle': 0.6618,
            'exact-gaussian': 0.27595,
            'exact-gaussian-trim': 0.27595,
            'exact-gaussian-fast': 0.27595,
            'constrained-cubic': 0.84990,
        }

        if user_method:
            self.sphere_functions['user-defined'] = user_method[0]
            self.alpha_defaults['user-defined'] = user_method[1]

        self.method = method
        if alpha is not None:
            self.alpha = tuple(listify(alpha))
        else:
            self.alpha = tuple(listify(self.alpha_defaults[self.method]))

    def _draw_particle(self, pos, rad, sign=1):
        # we can't draw 0 radius particles correctly, abort
        if rad == 0.0:
            return

        # translate to its actual position in the padded image
        pos = self._trans(pos)

        p = np.round(pos)
        r = np.round(np.array([1.0/self.zscale,1,1])*np.ceil(rad)+self.support_pad)

        tile = Tile(p-r, p+r, 0, self.shape.shape)
        rvec = tile.coords(form='vector')

        # if required, do an iteration to find the best radius to produce
        # the goal volume as given by the particular goal radius
        if self.exact_volume:
            t = sign*exact_volume_sphere(
                rvec, pos, rad, zscale=self.zscale, volume_error=self.volume_error,
                function=self.sphere_functions[self.method], args=self.alpha,
                max_radius_change=self.max_radius_change
            )
        else:
            # calculate the anti-aliasing according to the interpolation type
            dr = inner(rvec, pos, rad, zscale=self.zscale)
            t = sign*self.sphere_functions[self.method](dr, rad, *self.alpha)

        self.particles[tile.slicer] += t

    def param_radii(self):
        """ Return params of all radii """
        return [self._i2p(i, 'a') for i in range(self.N)]

    def param_particle(self, ind):
        """ Get position and radius of one or more particles """
        ind = self._vps(listify(ind))
        return [self._i2p(i, j) for i in ind for j in ['z', 'y', 'x', 'a']]

    def param_particle_pos(self, ind):
        """ Get position of one or more particles """
        ind = self._vps(listify(ind))
        return [self._i2p(i, j) for i in ind for j in ['z', 'y', 'x']]

    def param_particle_rad(self, ind):
        """ Get radius of one or more particles """
        ind = self._vps(listify(ind))
        return [self._i2p(i, 'a') for i in ind]

    def add_particle(self, pos, rad):
        """
        Add a particle or list of particles given by a list of positions and
        radii, both need to be array-like.

        Parameters
        ----------
        pos : array-like [N, 3]
            Positions of all new particles

        rad : array-like [N]
            Corresponding radii of new particles

        Returns
        -------
        inds : N-element numpy.ndarray.
            Indices of the added particles.
        """
        rad = listify(rad)
        # add some zero mass particles to the list (same as not having these
        # particles in the image, which is true at this moment)
        inds = np.arange(self.N, self.N+len(rad))
        self.pos = np.vstack([self.pos, pos])
        self.rad = np.hstack([self.rad, np.zeros(len(rad))])

        # update the parameters globally
        self.setup_variables()
        self.trigger_parameter_change()

        # now request a drawing of the particle plz
        params = self.param_particle_rad(inds)
        self.trigger_update(params, rad)
        return inds

    def remove_particle(self, inds):
        """
        Remove the particle at index `inds`, may be a list.
        Returns [3,N], [N] element numpy.ndarray of pos, rad.
        """
        if self.rad.shape[0] == 0:
            return

        inds = listify(inds)

        # Here's the game plan:
        #   1. get all positions and sizes of particles that we will be
        #      removing (to return to user)
        #   2. redraw those particles to 0.0 radius
        #   3. remove the particles and trigger changes
        # However, there is an issue -- if there are two particles at opposite
        # ends of the image, it will be significantly slower than usual
        pos = self.pos[inds].copy()
        rad = self.rad[inds].copy()

        self.trigger_update(self.param_particle_rad(inds), np.zeros(len(inds)))

        self.pos = np.delete(self.pos, inds, axis=0)
        self.rad = np.delete(self.rad, inds, axis=0)

        # update the parameters globally
        self.setup_variables()
        self.trigger_parameter_change()
        return np.array(pos).reshape(-1,3), np.array(rad).reshape(-1)

    def get_radii(self):
        return self.rad.copy()

    def exports(self):
        return (super(PlatonicSpheresCollection, self).exports() +
                [self.get_radii])

    def _p2i(self, param):
        """
        Parameter to indices, returns (coord, index). Therefore, for a
        pos    : ('x', 100)
        rad    : ('a', 100)
        zscale : ('zscale, None)
        """
        g = param.split('-')
        if len(g) == 1:
            return 'zscale', None
        if len(g) == 3:
            return g[2], int(g[1])

    def _update_type(self, params):
        """ Returns dozscale and particle list of update """
        dozscale = False
        particles = []
        for p in listify(params):
            typ, ind = self._p2i(p)
            particles.append(ind)
            dozscale = dozscale or typ == 'zscale'
        particles = set(particles)
        return dozscale, particles

    def _tile(self, n):
        """ Get the tile surrounding particle `n` """
        zsc = np.array([1.0/self.zscale, 1, 1])
        pos, rad = self.pos[n], self.rad[n]
        pos = self._trans(pos)
        return Tile(pos - zsc*rad, pos + zsc*rad).pad(self.support_pad)

    def update(self, params, values):
        """Calls an update, but clips radii to be > 0"""
        # radparams = self.param_radii()
        params = listify(params)
        values = listify(values)
        for i, p in enumerate(params):
            # if (p in radparams) & (values[i] < 0):
            if (p[-2:] == '-a') and (values[i] < 0):
                values[i] = 0.0
        super(PlatonicSpheresCollection, self).update(params, values)

    def __str__(self):
        return "{} N={}, zscale={}".format(self.__class__.__name__, self.N,
                self.zscale)

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, super(PlatonicSpheresCollection, self).nopickle())
        cdd(odict, ['rvecs', 'particles', '_params'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        ##Compatibility patches...
        self.float_precision = self.__dict__.get('float_precision', np.float64)
        ##end compatibility patch
        self.setup_variables()
        if self.shape:
            self.initialize()

#=============================================================================
# Coverslip half plane class
#=============================================================================
class Slab(Component):
    def __init__(self, zpos=0, angles=(0,0), param_prefix='slab', shape=None,
            float_precision=np.float64, category='obj'):
        """
        A half plane corresponding to a cover-slip.

        Parameters
        ----------
        shape : tuple
            field shape over which to calculate

        zpos : float
            position of the center of the slab in pixels

        angles : tuple of float (2,), optional
            Euler-like Angles of rotation of the normal with respect to the
            z-axis, i.e. ``angles=(0., 0.)`` gives a slab with a normal
            along z. The first angle theta is the rotation about the x-axis;
            the second angle phi is the rotation about the y-axis. Default
            is (0,0).

        float_precision : numpy float datatype
            One of numpy.float16, numpy.float32, numpy.float64; precision
            for precomputed arrays. Default is np.float64; make it 16 or 32
            to save memory.
        """
        self.lbl_zpos = param_prefix+'-zpos'
        self.lbl_theta = param_prefix+'-theta'
        self.lbl_phi = param_prefix+'-phi'

        if float_precision not in (np.float64, np.float32, np.float16):
            raise ValueError('float_precision must be one of np.float64, ' +
                    'np.float32, np.float16')
        self.float_precision = float_precision

        params = [self.lbl_zpos, self.lbl_theta, self.lbl_phi]
        values = [float(i) for i in [zpos, angles[0], angles[1]]]
        super(Slab, self).__init__(params, values, ordered=False,
                category=category)

        if shape:
            inner = shape.copy()  #same default as Platonic Sphere Collection
            self.set_shape(shape, inner)
        self.set_tile(self.shape)

        if self.shape:
            self.initialize()

    def rmatrix(self):
        """
        Generate the composite rotation matrix that rotates the slab normal.

        The rotation is a rotation about the x-axis, followed by a rotation
        about the z-axis.
        """
        t = self.param_dict[self.lbl_theta]
        r0 = np.array([ [np.cos(t),  -np.sin(t), 0],
                        [np.sin(t), np.cos(t), 0],
                        [0, 0, 1]])

        p = self.param_dict[self.lbl_phi]
        r1 = np.array([ [np.cos(p), 0, np.sin(p)],
                        [0, 1, 0],
                        [-np.sin(p), 0, np.cos(p)]])
        return np.dot(r1, r0)

    def normal(self):
        return np.dot(self.rmatrix(), np.array([1,0,0]))

    def _setup(self):
        self.rvecs = self.shape.coords(form='broadcast')
        self.image = np.zeros(self.shape.shape, dtype=self.float_precision)

    def _draw_slab(self):
        # for the position at zpos, and the center in the x-y plane
        pos = np.array([
            self.param_dict[self.lbl_zpos], self.shape.shape[1]//2, self.shape.shape[2]//2
        ])
        pos = pos + self.inner.l

        p = (np.sum([r*n for r, n in zip(self.rvecs, self.normal())]) -
                pos.dot(self.normal()))
        m1 = p < -4.
        m0 = p > 4.
        mp = ~(m1 | m0)
        self.image[m1] = 1.
        self.image[mp] = 1.0/(1.0 + np.exp(7*p[mp]))  #FIXME why is this not an erf???
        self.image[m0] = 0.

    def initialize(self):
        self._setup()
        self._draw_slab()

    def set_tile(self, tile):
        self.tile = tile

    def update(self, params, values):
        super(Slab, self).update(params, values)
        self._draw_slab()

    def get(self):
        return self.image[self.tile.slicer]

    def get_update_tile(self, params, values):
        return self.shape.copy()

    def __getstate__(self):
        odict = self.__dict__.copy()
        cdd(odict, super(Slab, self).nopickle())
        cdd(odict, ['rvecs', 'image'])
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        ##Compatibility patches...
        self.float_precision = self.__dict__.get('float_precision', np.float64)
        ##end compatibility patch
        if self.shape:
            self.initialize()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} <{}>".format(
            str(self.__class__.__name__), self.param_dict
        )

