cimport numpy as cnp
cimport pypacmensl.libpacmensl._discrete_distribution as _dd
cimport pypacmensl.arma.arma4cy as arma


cdef class DiscreteDistribution:
    cdef:
        _dd.DiscreteDistribution* this_
        object env_
