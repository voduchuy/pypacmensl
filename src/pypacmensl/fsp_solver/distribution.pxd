cimport numpy as cnp
cimport pypacmensl.libpacmensl._discrete_distribution as _dd
cimport pypacmensl.arma.arma4cy as arma
from pypacmensl.callbacks.pacmensl_callbacks cimport call_py_weight_fun

from libcpp.vector cimport vector


cdef class DiscreteDistribution:
    cdef:
        _dd.DiscreteDistribution* this_
        object env_

