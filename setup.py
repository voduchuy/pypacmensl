import os
from sys import platform
from distutils.core import setup, Extension
from Cython.Build import cythonize
from glob import glob
import numpy as np
from os.path import join

metadata = {
        'name':             'pypacmensl',
        'version':          '0.0.1',
        'keywords':         'chemical master equation \ finite state projection \ parallel computing',
        'license':          'BSD',
        'author':           'Huy Vo',
        'author_email':     'vdhuy91@gmail.com',
        'maintainer':       'Huy Vo',
        'maintainer_email': 'vdhuy91@gmail.com',
}

# %% Set MPI-wrapped compilers and extra C++ options
os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpic++'
extra_compile_args = ['-std=c++11', '-Wall', '-Wextra']
extra_links = []
# %%


pypacmensl_dirs = ['./src/pypacmensl']
#
pypacmensl_ext=[]

pypacmensl_subpackages=['callbacks', 'arma', 'fsp_solver', 'sensitivity', 'smfish', 'state_set',
                        # 'stationary',
                        'utils', 'ssa']

for folder in pypacmensl_subpackages:
    extensions = cythonize('src/pypacmensl/'+ folder + '/*.pyx', language_level=3, include_path=pypacmensl_dirs)
    for ext in extensions:
        ext.language = "c++"
        ext.libraries = ['pacmensl', 'petsc']
        ext.extra_compile_args = extra_compile_args
        ext.include_dirs = [os.environ['PETSC_DIR'] + '/include', np.get_include()]
        ext.library_dirs = [join(os.environ['PETSC_DIR'], 'lib'), join(np.get_include(), '..', '..', 'random', 'lib')]
        ext.extra_link_args=extra_links

    pypacmensl_ext.extend(extensions)


# %%

setup(
        packages=['pypacmensl', 'pypacmensl.fsp_solver', 'pypacmensl.state_set', 'pypacmensl.sensitivity',
                  'pypacmensl.stationary'],
        package_dir={'pypacmensl':             'src/pypacmensl',
                     'pypacmensl.fsp_solver':  'src/pypacmensl/fsp_solver',
                     'pypacmensl.state_set':   'src/pypacmensl/state_set',
                     'pypacmensl.sensitivity': 'src/pypacmensl/sensitivity',
                     # 'pypacmensl.stationary':  'src/pypacmensl/stationary'
                     },
        ext_modules= pypacmensl_ext,
        **metadata
)
