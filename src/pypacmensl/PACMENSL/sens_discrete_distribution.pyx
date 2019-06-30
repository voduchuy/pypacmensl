cimport sens_discrete_distribution as cfsp
import numpy as np
cimport numpy as cnp
cimport arma4cy as arma

cdef class SensDiscreteDistribution:
    cdef cfsp.SensDiscreteDistribution* this_

