import os
import sysconfig
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import mpi4py
import ctypes

metadata = {
    'name'             : 'pypacmensl',
    'version'          : '0.0.1',
    'credits'          : 'Munsky Group',
    'keywords'         : 'chemical master equation \ finite state projection \ parallel computing',
    'license'          : 'BSD',
    'author'           : 'Huy Vo',
    'author_email'     : 'vdhuy91@gmail.com',
    'maintainer'       : 'Huy Vo',
    'maintainer_email' : 'vdhuy91@gmail.com',
    }

extra_compile_args = ['-std=c++11', '-Wall', '-Wextra', '-stdlib=libc++', '-mmacosx-version-min=10.9']
mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpic++ --showme:link").read().strip().split(' ')
extra_compile_args += mpi_compile_args

print(extra_compile_args)

ext = Extension(name='pypacmensl.PACMENSL',
                sources=["src/pypacmensl/pypacmensl.PACMENSL.pyx"],
                language="c++",
                libraries=['pacmensl','petsc'],
                extra_link_args = mpi_link_args,
                extra_compile_args = extra_compile_args,
                include_dirs=[os.environ['PETSC_DIR']+'/include', np.get_include(), "src/pypacmensl/PACMENSL/include"],
                library_dirs=[os.environ['PETSC_DIR']+'/lib'],
              )

setup(
      packages = ['pypacmensl'],
      package_dir = {'pypacmensl' : 'src/pypacmensl'},
      ext_modules=cythonize(ext, language_level=3),
      **metadata
      )
