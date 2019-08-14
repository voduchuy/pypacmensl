cimport numpy as cnp
from libcpp cimport bool
from libcpp.vector cimport vector

from pypacmensl.petsc.petsc_objects cimport PetscReal

from pypacmensl.libpacmensl cimport _smfish
cimport pypacmensl.arma.arma4cy as arma
from pypacmensl.fsp_solver.distribution cimport DiscreteDistribution
from pypacmensl.sensitivity.distribution cimport SensDiscreteDistribution

ctypedef fused FusedDistribution:
    DiscreteDistribution
    SensDiscreteDistribution

cdef class SmFishSnapshot:
    cdef:
        _smfish.SmFishSnapshot* this_
        object env_
