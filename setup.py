#from setuptools import setup
from distutils.core import setup

setup(name='peri',
      license='MIT License',
      author='Matt Bierbaum, Brian Leahy, Alex Alemi',
      version='0.1.1',

      packages=['peri', 'peri.mc', 'peri.comp', 'peri.viz', 'peri.priors', 'peri.test', 'peri.opt'],
      install_requires=[
          "numpy>=1.8.1",
          "scipy>=0.14.0",
          "matplotlib>=1.0.0",
          "pyfftw>=0.9.1",
          "pillow>=1.1.7"
      ],
      scripts=['bin/peri']
)
