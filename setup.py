#from setuptools import setup
from distutils.core import setup

setup(name='cbamf',
      license='MIT License',
      author='Matt Bierbaum, Brian Leahy, Alex Alemi',
      version='0.1.1',

      packages=['cbamf', 'cbamf.mc', 'cbamf.comp', 'cbamf.psf', 'cbamf.viz', 'cbamf.priors', 'cbamf.test', 'cbamf.opt'],
      install_requires=[
          "numpy>=1.8.1",
          "scipy>=0.14.0",
          "matplotlib>=1.0.0",
          "pyfftw>=0.9.1",
          "libtiff>=0.4",
      ],
      scripts=['bin/cbamf']
)
