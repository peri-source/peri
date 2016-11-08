********************
Package architecture
********************

This document is a description of the overall package architecture elucidated
via a simple example of polynomial fitting. In this example, we will have noisy
data which we wish to fit with a polynomial. While a relatively simple problem,
we will implement it in the PERI framework including how to get the best
performance using various aspects of the package.

Overview
========

The basic structure needed to fit a model to data is the ``State`` object. This
structure holds the data and model and provides a common interface that allows
the ``peri`` package to optimize a set of parameters to match the two. All of
the common operations are implemented, leaving you to implement only a few
methods in order to have a functioning ``State``. In order to implement a ``State``,
you must know about the following functions:

.. autoclass:: peri.states.State
    :members: __init__, data, model, update
    :undoc-members:

To demonstrate this ``State`` class, let's implement a polynomial fit class
which fits a one dimensional curve to an arbitrary degree polynomial.
In the following class, we subclass the 

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
            """ Get the raw data of the model fit """
            return self._data
    
        @property
        def model(self):
            """ Get the current model fit to the data """
            return self._model
