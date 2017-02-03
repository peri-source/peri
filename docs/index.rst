Welcome to PERI's documentation!
================================

.. topic:: Getting Started

    For an introduction to how ``peri`` works and how to use it, see the
    :doc:`walkthrough </walkthrough>`.

Parameter Extraction from Reconstruction of Images (PERI) is a package that
extracts features from microscope images by fitting a physics-based model to
data. It is a set of models and components that are used to recreate the
physics of image formation in order to extract desired quantities such as
particle sizes and positions at the information theoretic limit.  While a very
general software package, the primary model currently implemented is that of 3D
confocal microscopes, particularly of the line and point scan varities.

In these documents you will find information about:

* :doc:`Installation </installation>`
* :doc:`Package overview </architecture>`
* :ref:`ref-developing-models`
* :ref:`ref-implementing-components`

Documentation:
==============

.. toctree::
   :maxdepth: 3

   installation
   walkthrough
   architecture
   optimization
   parallel

Reference:
==========

.. toctree::
   :glob:
   :maxdepth: 3

   reference/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

