import numpy as np
cimport numpy as cnp

from pypacmensl.fsp_solver.distribution cimport DiscreteDistribution
cimport pypacmensl.arma.arma4cy as arma
cimport pypacmensl.libpacmensl._sens_discrete_distribution as _sdd
from pypacmensl.callbacks.pacmensl_callbacks cimport call_py_weight_fun

cdef class SensDiscreteDistribution:
    cdef:
        _sdd.SensDiscreteDistribution* this_
        object env_

    cpdef GetStatesViewer(self)

    cpdef GetProbViewer(self)

    cpdef RestoreProbViewer(self, double[::1] viewer)

    cpdef GetSensViewer(self, int iS)

    cpdef RestoreSensViewer(self, int iS, cnp.ndarray sensview)

    cpdef GetNumParameters(self)
