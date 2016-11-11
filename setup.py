from distutils.core import setup
#from setuptools import setup
import peri

desc = open('./README.md').read()
reqs = open('./requirements.txt').readlines()

setup(name='peri',
      url='http://github.com/mattbierbaum/peri/',
      license='MIT License',
      author='Matt Bierbaum, Brian Leahy',
      version=peri.__version__,

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
          'Development Status :: 4 - Beta',
          'Natural Language :: English',
          'Environment :: Web Environment',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Software Development :: Libraries :: Application Frameworks',
          'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
      ],
      description='Parameter Extraction from the Reconstruction of Images',
      long_description=desc,
)
