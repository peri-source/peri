#from distutils.core import setup
from setuptools import setup

try:
    desc = open('README.md').read()
except (IOError, FileNotFoundError) as e:
    desc = ''

setup(name='peri',
      url='http://github.com/peri-source/peri/',
      license='MIT License',
      author='Matt Bierbaum, Brian Leahy',
      version='0.1.3',

      packages=[
          'peri', 'peri.mc', 'peri.comp', 'peri.viz',
          'peri.priors', 'peri.test', 'peri.opt', 'peri.gui'
      ],
      scripts=['bin/peri'],
      install_requires=[
          "future>=0.15.0",
          "numpy>=1.8.1",
          "scipy>=0.14.0",
          "matplotlib>=1.0.0",
          "pillow>=1.1.7"
      ],
      package_data={
          'peri': ['../README.md', 'gui/*.ui']
      },

      platforms='any',
      classifiers = [
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Development Status :: 2 - Pre-Alpha',
          'Natural Language :: English',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      description='Parameter Extraction from the Reconstruction of Images',
      long_description=desc,
)
