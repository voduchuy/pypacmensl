cdef extern from "petsc.h":
    ctypedef double PetscReal
    ctypedef int PetscInt

    ctypedef enum PetscBool:
        PETSC_FALSE
        PETSC_TRUE

    ctypedef struct Vec
    ctypedef struct Mat