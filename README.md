PERI : Parameter Extraction from Reconstruction of Images
=========================================================

A software which implements the generalized framework of extracting parameters
(positions, radii of spheres; microscope properties) by full reconstruction of
experimental images. The general framework is built on combining components
(illumination field, background, particles) into a model which is then
optimized given data.

Installation
------------

Very straightforward, simply use distutils' setup.py:

    python setup.py install

Testing
-------

Testing is done through nose, and can be performed with:

    nosetests

