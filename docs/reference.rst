**********
Reference
**********

.. role:: python(code)
    :language: python

peri.conf
=========

.. automodule:: peri.conf
    :members:

peri.fft
========

.. automodule:: peri.fft
    :members:

peri.states
===========

peri.states.State
-----------------

.. autoclass:: peri.states.State
    :members:

peri.states.ImageState
-----------------------

.. autoclass:: peri.states.ImageState
    :members:

peri.states.{save,load}
-----------------------

.. autofunction:: peri.states.save

.. autofunction:: peri.states.load

peri.comp
=========

peri.comp.comp
--------------

.. autoclass:: peri.comp.comp.ParameterGroup
    :members:

.. autoclass:: peri.comp.comp.Component
    :members:

.. autoclass:: peri.comp.comp.ComponentCollection
    :members:

peri.comp.objs
--------------

.. autoclass:: peri.comp.objs.PlatonicSpheresCollection
    :members:

.. autoclass:: peri.comp.objs.Slab
    :members:

peri.comp.ilms
--------------

.. autoclass:: peri.comp.ilms.Polynomial3D
    :members:

.. autoclass:: peri.comp.ilms.Polynomial2P1D
    :members:

.. autoclass:: peri.comp.ilms.LegendrePoly2P1D
    :members:

.. autoclass:: peri.comp.ilms.BarnesStreakLegPoly2P1D
    :members:

peri.models
===========

.. autoclass:: peri.models.Model
    :members:

.. autoclass:: peri.models.ConfocalImageModel
    :members:

peri.opt
========

peri.opt.optimize
-----------------

.. autoclass:: peri.opt.optimize.LMEngine
    :members:

.. autofunction:: peri.opt.optimize.burn
    :members:

peri.opt.addsubtract
--------------------

.. autofunction:: peri.opt.addsubtract.add_subtract

.. autofunction:: peri.opt.addsubtract.feature_guess

peri.util
==========

peri.util.Tile
--------------

.. autoclass:: peri.util.Tile
    :members:

peri.util.Image
---------------

.. autoclass:: peri.util.Image
    :members:

peri.util.RawImage
------------------

.. autoclass:: peri.util.RawImage
    :members:

.. autoclass:: peri.util.NullImage
    :members:

peri.util.{*}
-------------

.. autoclass:: peri.util.ProgressBar

.. autofunction:: peri.util.oddify

.. autofunction:: peri.util.listify

.. autofunction:: peri.util.delistify

.. autofunction:: peri.util.aN

.. autofunction:: peri.util.patch_docs

.. autofunction:: peri.util.indir

peri.interpolation
==================

BarnesInterpolation1D
---------------------

.. autoclass:: peri.interpolation.BarnesInterpolation1D
    :members: __call__

ChebyshevInterpolation1D
------------------------

.. autoclass:: peri.interpolation.ChebyshevInterpolation1D
    :members:  __call__

