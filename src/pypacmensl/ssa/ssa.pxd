from mpi4py cimport MPI
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t

cdef class SSASolver:
    cdef:
        MPI.Comm comm_
        object prop_t_
        object prop_x_
        np.ndarray stoich_matrix_
        object bitGen_

cdef class SSATrajectory:
    cdef:
        object prop_t_
        object prop_x_
        np.ndarray stoich_matrix_

cdef class PmPdmsrSampler:
    cdef:
        MPI.Comm comm_
        object prop_t_
        object prop_x_
        np.ndarray stoich_matrix_
        object f_transcr_
        np.ndarray deg_rates_
        object bitGen_
