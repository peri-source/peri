import os
import glob
from os.path import join as pjoin
#from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy
import shutil

#==============================================================
# thanks to the wonderful work by rmcgibbo for monkey-patching
# disutils to work with cuda compilation
# https://github.com/rmcgibbo/npcuda-example
#==============================================================
# Obtain the numpy include directory.  This logic works across numpy versions.
DOPLOT = True

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    default_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', 'nvcc')
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        default_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so
    self._compile = _compile

class CustomBuildExt(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA environment on the system
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

GLLIB = ['GL', 'GLU', 'glut'] if DOPLOT else []
CUARGS = {
    'swig_opts': ['-c++'],
    'library_dirs': [CUDA['lib64']],
    'runtime_library_dirs': [CUDA['lib64']],
    'include_dirs': [numpy_include, CUDA['include'], 'src'],
    'libraries': ['cudart', 'curand', 'cufft', 'cublas', 'm', 'rt'] + GLLIB,
    'extra_compile_args': {
        'gcc': ['-std=c99', '-O2', "-fPIC", "-DCUDA", "-march=native", '-pipe', '-Wall'],
        'nvcc': ['-DCUDA', '-O2', '-use_fast_math', '-arch=sm_20', '-c', '--compiler-options', "'-fPIC'"]#, '--compiler-options', "'-Wall'"]
    },
}

CARGS = {
    'include_dirs': [numpy_include, 'src'],
    'libraries': ['m', 'rt'] + GLLIB,
    'extra_compile_args': {
        'gcc': ['-std=c99', "-fPIC"],
    },
}

ext = Extension('cbamf.cu._nbl',
        sources=['src/nbl.i', 'src/nbl.cpp'], **CUARGS)

ext2 = Extension('cbamf.cu._fields',
        sources=['src/fields.i', 'src/fields.cu', 'src/util.cu'], **CUARGS)

extensions = [ext, ext2]
extensions = extensions if DOPLOT else extensions[:-1]

for f in glob.glob("src/*.py"):
    ff = os.path.basename(f)
    shutil.copyfile(f, os.path.join("./cbamf/cu", ff))

for f in glob.glob("build/lib*/cbamf/cu/*.so"):
    ff = os.path.basename(f)
    shutil.copyfile(f, os.path.join("./cbamf/cu", ff))

setup(name='cbamf',
      author='Matt Bierbaum',
      version='0.1',

      packages=['cbamf', 'cbamf.cu'],
      cmdclass={'build_ext': CustomBuildExt},
      ext_modules=extensions,
      zip_safe=False
)
