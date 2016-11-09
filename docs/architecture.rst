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
    :members: __init__, data, model, update
    :undoc-members:

Example: PolyFitState
---------------------------

To demonstrate this ``State`` class, let's implement a polynomial fit class
which fits a one dimensional curve to an arbitrary degree polynomial.  In the
following class, we subclass :class:`peri.states.State` and implement the
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

On top of the :class:`peri.states.State` class, we add several layers of
complexity.  We feel these levels of complexity help, rather than harm, the
development of new image models and allows the flexibility to adapt to new
brands and types of microscopes and experimental systems. Here we will describe
these structures and along the way develop a very simple image model. 

* **Model** -- 
* **Component** --
* **Tile** --
