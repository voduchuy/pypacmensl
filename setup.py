import os
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

extra_compile_args = ['-std=c++11', '-Wall', '-Wextra']
# , '-stdlib=libc++', '-mmacosx-version-min=10.9']
mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
extra_compile_args += mpi_compile_args
mpi_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')

print(extra_compile_args)

pypacmensl_srcs = glob('**/*.pyx', recursive=True)

pypacmensl_dirs = ['./src/pypacmensl']

extensions = cythonize('**/*.pyx', language_level=3, include_path=pypacmensl_dirs)

for ext in extensions:
    ext.language="c++"
    ext.libraries=['pacmensl', 'petsc']
    ext.extra_link_args=mpi_link_args
    ext.extra_compile_args = extra_compile_args
    ext.include_dirs = [os.environ['PETSC_DIR'] + '/include', np.get_include(),
                    "src/pypacmensl/include"]
    ext.library_dirs = [os.environ['PETSC_DIR'] + '/lib']


setup(
        packages=['pypacmensl', 'pypacmensl.fsp_solver', 'pypacmensl.state_set', 'pypacmensl.sensitivity',
                  'pypacmensl.stationary'],
        package_dir= {'pypacmensl': 'src/pypacmensl',
                      'pypacmensl.fsp_solver': 'src/pypacmensl/fsp_solver',
                      'pypacmensl.state_set' : 'src/pypacmensl/state_set',
                      'pypacmensl.sensitivity': 'src/pypacmensl/sensitivity',
                      'pypacmensl.stationary' : 'src/pypacmensl/stationary'},
        # package_data={'pypacmensl' : ['**/*.py']},
        # include_package_data=True,
        ext_modules=extensions,
        **metadata
)
