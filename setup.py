from distutils.core import setup
#from setuptools import setup

desc = open('./README.md').read()
reqs = open('./requirements.txt').readlines()

setup(name='peri',
      url='http://github.com/peri-source/peri/',
      license='MIT License',
      author='Matt Bierbaum, Brian Leahy',
      version='0.1.1',

      packages=[
          'peri', 'peri.mc', 'peri.comp', 'peri.viz',
          'peri.priors', 'peri.test', 'peri.opt', 'peri.gui'
      ],
      package_data={'peri': ['gui/*.ui']},
      scripts=['bin/peri'],
      install_requires=reqs,

      platforms='any',
      classifiers = [
          'Programming Language :: Python',
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
