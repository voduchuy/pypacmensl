# pypacmensl

Python wrapper for the [PACMENSL](https://github.com/voduchuy/pacmensl) library.

## Dependencies

- Python 3.6+.
- mpi4py (https://mpi4py.readthedocs.io/en/stable/).
- numpy 1.18.5+.
- [Cython](https://cython.org/).
- Distutils.
- A C++ compiler.
- An MPI distribution such as [OpenMPI](https://www.open-mpi.org/).
- [PACMENSL](https://github.com/voduchuy/pacmensl) must be installed and added to your C++ compiler's search path.

## Installation

To install for all users:
`
python setup.py install
`

To install for local user:
`
python setup.py install --user
`
