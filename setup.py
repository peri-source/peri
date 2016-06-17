#from setuptools import setup
from distutils.core import setup

reqs = open('./requirements.txt').readlines()

setup(name='peri',
      license='MIT License',
      author='Matt Bierbaum, Brian Leahy',
      version='0.1.1',

      packages=[
          'peri', 'peri.mc', 'peri.comp', 'peri.viz',
          'peri.priors', 'peri.test', 'peri.opt'
      ],
      install_requires=reqs,
      scripts=['bin/peri']
)
