********************
Package architecture
********************

This document is a description of the overall package architecture elucidated
via a simple example of polynomial fitting. In this example, we will have noisy
data which we wish to fit with a polynomial. While a relatively simple problem,
we will implement it in the PERI framework including how to get the best
performance using various aspects of the package.

States
=======

Overview
--------

The basic structure needed to fit a model to data is the ``State`` object. This
structure holds the data and model and provides a common interface that allows
the ``peri`` package to optimize a set of parameters to match the two. All of
the common operations are implemented, leaving you to implement only a few
methods in order to have a functioning ``State``. In order to implement a
``State``, you must know about the following functions:

.. autoclass:: peri.states.State
    :members: data, model, update
    :undoc-members:
    :noindex:

Example: PolyFitState
---------------------------

To demonstrate this ``State`` class, let's implement a polynomial fit class
which fits a one dimensional curve to an arbitrary degree polynomial.  In the
following class, we subclass :class:`~peri.states.State` and implement the
properties and functions that we outlined in the previous section. In
particular, in the ``__init__``, we store the input ``x`` points as a member
variable and ``y`` values as data. We then set up parameter names and values
depending on whether the user supplied coefficients the init. We call the
superclasses' init and call update with these parameters and values to make
sure that the model is calculated.

.. code-block:: python

    class PolyFitState(peri.states.State):
        def __init__(self, x, y, order=2, coeffs=None):
            self._data = y
            self._xpts = x
    
            params = ['c-%i' %i for i in xrange(order)]
            values = coeffs if coeffs is not None else [0.0]*order
    
            super(PolyFitState, self).__init__(
                params=params, values=values, ordered=False
            )
    
            self.update(self.params, self.values)

        def update(self, params, values):
            super(PolyFitState, self).update(params, values)
            self._model = np.polyval(self.values, self._xpts)
    
        @property
        def data(self):
            return self._data
    
        @property
        def model(self):
            return self._model

The ``update`` function for this class simply uses numpy's ``polyval`` function
to evaluate the parameter values as a polynomial at the stored ``x`` values.
This model value is stored as a member variable and returned for the ``model``
property. The ``data`` property is simple the stored ``y`` values.

We can then make an instance of this ``PolyFitState`` and begin to fit fake
'data' with our model.

.. code-block:: python

    # noise level
    sigma = 0.3

    # num of coefficients, datapoints
    C, N = 8, 1000

    # generate data
    c = 2*np.random.rand(C) - 1
    x = np.linspace(0.0, 2.0, N)
    y = np.polyval(c, x) + sigma*np.random.randn(N)

    # create a state
    s = PolyFitState(x, y, order=C)

Properties
^^^^^^^^^^

We can check out some of the common functions provided to ``State`` objects
before we begin to optimize. For example, we can look at an approximation to
the sensitivity matrix :math:`J^T J`:

.. code-block:: python

    import matplotlib.pyplot as pl
    pl.imshow(s.JTJ())

or we can calculate the `Cramer-Rao bound
<https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound>`_ for the fit
parameters:

.. code-block:: python

    s.crb()

From here, we can optimize the parameters of the state along with the estimated
noise level. First, we will do so using Monte Carlo sampling, particularly with
a multidimensional slice sampler.

Optimization
^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as pl
    from peri.mc import sample

    # burn a number of samples, then collect the samples around the true value
    h = sample.sample_state(s, s.params, N=1000, doprint=True, procedure='uniform')
    h = sample.sample_state(s, s.params, N=30, doprint=True, procedure='uniform')

    pl.plot(s.data, 'o')
    pl.plot(s.model, '-')

    # distribution of fit parameter values
    h.get_histogram()

We can also optimize the ``PolyFitState`` using variations on nonlinear least
squares optimization with `Levenberg-Marquardt
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_.

.. code-block:: python

    import peri.opt.optimize as opt
    opt.do_levmarq(s, s.params[1:])

    pl.plot(s.data, 'o')
    pl.plot(s.model, '-')

Image states
=============

Since a common usage pattern of PERI is to optimize models of experimental
microscope images, we implemented a very flexible ``ImageState`` which
provides:

* Easy implementation of new model equations
* Compartmentalization of parts of an image
* Many optimizations including local image updates and better FFTs

On top of the :class:`~peri.states.State` class, we add several layers of
complexity.  We feel these levels of complexity help, rather than hinder, the
development of new image models and allows the flexibility to adapt to new
brands and types of microscopes and experimental systems. Here we will describe
these structures and along the way develop a very simple image model of
polydisperse discs in a plane imaged with microscope described by a Gaussian
point-spread-function (PSF). In particular, the model we will be creating is:

.. math::

    \mathcal{M}(\bvec{x}) = B(\bvec{x}) + \int P(\bvec{x} - \bvec{x}^{\prime}) S(\bvec{x}; \{\bvec{p}_i, a_i\}) \rm{d}\bvec{x}^{\prime}

where :math:`P` is the point spread function and :math:`S` is the shape
function which defines the *Platonic* solid, and :math:`B` is a spatially
varying background which may represent any number of confounding factors
in image formation:

.. math::

    P(\bvec{x}) &= \frac{1}{\sqrt{2\pi \sigma^2}} e^{ - \| \bvec{x}\|^2 / 2\sigma^2 } \\
    S(\bvec{x}; \{\bvec{p}_, a_i\}) &= \sum_{i=0}^{N_{\rm{particles}}}  \frac{1}{1 + e^{\alpha (\|\bvec{x} - \bvec{p}_i\| - a_i)}} \\
    B(\bvec{x}) &= \sum_{i=0}^{C_x} \sum_{j=0}^{C_y} c_{ij} L_i(x) L_j(y)

That is, we have a simple image model of circular discs influenced by
microscope optics with a non-uniform background. Since each of this functional
form seem distinct, we separate our model into small objects which we call
``components``. These ``components`` calculate part of the model (:math:`P`,
:math:`S`, ...) over a certain region, or ``Tile``, then get combined back into
the overall model. This philosophy can be expressed simply as:

* **Model** (:class:`~peri.models.Model`) -- The entire equation (and derivatives) describing the image formation :math:`\mathcal{M}`
* **Component** (:class:`~peri.comp.comp.Component`) -- Small subsections of the model e.g. :math:`P`, :math:`S`, :math:`B`
* **Tile** (:class:`~peri.util.Tile`) -- Regions of the image over which parts of the model are calculated.

ParameterGroups and Components
------------------------------

First, we split the model into small parts which break up the monolithic model
into managable pieces. We call these objects ``Components``
(:class:`~peri.comp.comp.Component`). In their most basic form, a ``Component``
is a :class:`~peri.comp.comp.ParameterGroup` which also knows about tiling.

ParameterGroup
^^^^^^^^^^^^^^

A :class:`~peri.comp.comp.ParameterGroup` is a container that provides a common
interface to any object which computes "something" based on a set of
``parameters`` (string names) and ``values`` (the values associated with those
names). In the most basic form, a ``ParameterGroup`` must care about the following
structure:

.. autoclass:: peri.comp.comp.ParameterGroup
    :members: get_values, set_values, update
    :noindex:

Component
^^^^^^^^^

A subclass of the ``ParameterGroup`` is a ``Component`` which specifically
computes part of the ``Model`` computed by ``ImageState``. Therefore, in
addition to the methods required by ``ParameterGroup``, it must also implement
functions pertaining to *tiling*, how the ``ImageState`` deals with local and
semi-local updates for best performance.

.. autoclass:: peri.comp.comp.Component
    :members: initialize, get_update_tile, get_padding_size, set_tile, set_shape, get
    :noindex:

ComponentCollections and Models
-------------------------------

.. autoclass:: peri.comp.comp.ComponentCollection
    :members: get
    :noindex:

.. autoclass:: peri.models.Model
    :members: evaluate
    :noindex:

Example: Gaussian-blurred discs
-------------------------------
