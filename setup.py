import os
from sys import platform
from distutils.core import setup, Extension
from Cython.Build import cythonize
from glob import glob
import numpy as np

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

os.environ['CC'] = 'mpicc'
os.environ['CXX'] = 'mpic++'

extra_compile_args = ['-std=c++11', '-Wall', '-Wextra']
if platform == 'darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']

pypacmensl_srcs = glob('**/*.pyx', recursive=True)

pypacmensl_dirs = ['./src/pypacmensl']

extensions = cythonize('**/*.pyx', language_level=3, include_path=pypacmensl_dirs)

for ext in extensions:
    ext.language = "c++"
    ext.libraries = ['pacmensl', 'petsc']
    ext.extra_compile_args = extra_compile_args
    ext.include_dirs = [os.environ['PETSC_DIR'] + '/include', np.get_include(),
                        "src/pypacmensl/callbacks"]
    ext.library_dirs = [os.environ['PETSC_DIR'] + '/lib']

setup(
        packages=['pypacmensl', 'pypacmensl.fsp_solver', 'pypacmensl.state_set', 'pypacmensl.sensitivity',
                  'pypacmensl.stationary'],
        package_dir={'pypacmensl':             'src/pypacmensl',
                     'pypacmensl.fsp_solver':  'src/pypacmensl/fsp_solver',
                     'pypacmensl.state_set':   'src/pypacmensl/state_set',
                     'pypacmensl.sensitivity': 'src/pypacmensl/sensitivity',
                     'pypacmensl.stationary':  'src/pypacmensl/stationary'},
        ext_modules=extensions,
        **metadata
)
