from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport mpi4py.MPI as mpi
cimport numpy as cnp

cimport pypacmensl.libpacmensl._fsp as _fsp
cimport pypacmensl.libpacmensl._discrete_distribution as _dd
from pypacmensl.callbacks.pacmensl_callbacks cimport *

from pypacmensl.petsc.petsc_objects cimport PetscReal, PetscInt
cimport pypacmensl.arma.arma4cy as arma
cimport pypacmensl.fsp_solver.distribution as cdd

cdef class FspSolverMultiSinks:
    cdef:
        _fsp.FspSolverMultiSinks *this_
        bool set_up_
        object env_
