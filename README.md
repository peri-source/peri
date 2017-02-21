PERI : Parameter Extraction from Reconstruction of Images
=========================================================

A software which implements the generalized framework of extracting parameters
(e.g. locations and properties of objects in an image) by full reconstruction
of experimental images. The general framework is built on combining components
(e.g. illumination field, background, particles, point-spread function) into a
model which is then optimized given data.


Installation
------------

Very straightforward, simply use distutils' setup.py:

    python setup.py install

Testing
-------

Testing is done through nose, and can be performed with:

    nosetests

Documentation
-------------

Currently, the documentation is hosted at https://peri-source.github.io/peri-docs/
