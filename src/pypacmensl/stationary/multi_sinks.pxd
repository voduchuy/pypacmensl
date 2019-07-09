from libcpp cimport bool
from mpi4py.libmpi cimport MPI_Comm

cimport mpi4py.MPI as mpi
from pypacmensl.callbacks.pacmensl_callbacks cimport PyConstrWrapper, PyPropWrapper, PyTFunWrapper
from pypacmensl.libpacmensl cimport _state_set as _ss
from pypacmensl.arma cimport arma4cy as arma
from pypacmensl.libpacmensl._fsp cimport Model, PartitioningType
from pypacmensl.petsc.petsc_objects cimport PetscInt, PetscReal
from pypacmensl.libpacmensl cimport _fsp
cimport pypacmensl.fsp_solver.distribution as cdd
cimport numpy as cnp


cdef class StationaryFspSolverMultiSinks:
      cdef:
            _fsp.StationaryFspSolverMultiSinks* this_
            bool set_up