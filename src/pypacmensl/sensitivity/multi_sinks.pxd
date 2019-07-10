from libcpp.vector cimport *
from libcpp cimport bool
cimport numpy as cnp
from mpi4py.libmpi cimport MPI_Comm
cimport mpi4py.MPI as mpi

from pypacmensl.arma cimport arma4cy as arma
from pypacmensl.libpacmensl cimport _sens_discrete_distribution as _sdd
from pypacmensl.libpacmensl cimport _fsp
from pypacmensl.sensitivity cimport distribution as sdd
from pypacmensl.callbacks.pacmensl_callbacks cimport *


cdef class SensFspSolverMultiSinks:
    cdef:
        _fsp.SensFspSolverMultiSinks* this_
        bool set_up_
        object env_