import numpy as np
cimport numpy as cnp

cimport pypacmensl.arma.arma4cy as arma
cimport pypacmensl.libpacmensl._sens_discrete_distribution as _sdd

cdef class SensDiscreteDistribution:
    cdef:
        _sdd.SensDiscreteDistribution* this_

    cpdef GetStatesViewer(self)

    cpdef GetProbViewer(self)

    cpdef RestoreProbViewer(self, double[::1] viewer)

    cpdef GetSensViewer(self, int iS)

    cpdef RestoreSensViewer(self, int iS, cnp.ndarray sensview)
