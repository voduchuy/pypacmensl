# distutils: language = c++

cimport numpy as cnp
from libcpp cimport bool

from pypacmensl.libpacmensl cimport _smfish
cimport pypacmensl.arma.arma4cy as arma
from pypacmensl.fsp_solver.distribution cimport DiscreteDistribution

import numpy as np

cdef class SmFishSnapshot:
    cdef _smfish.SmFishSnapshot* this_

    def __cinit__(self, cnp.ndarray observations):
        if observations.dtype is not np.intc:
            observations = observations.astype(np.intc)
        if not observations.flags['C_CONTIGUOUS']:
            observations = np.ascontiguousarray(observations)

        cdef arma.Mat[int] observ = arma.MakeIntMat(observations)
        self.this_ = new _smfish.SmFishSnapshot(observ)

    def __dealloc__(self):
        if self.this_ is not NULL:
            del self.this_

    cpdef LogLikelihood(self, DiscreteDistribution dist, species=None, use_log_2 = False):
        cdef bool cuse_log_2
        cdef arma.Col[int] measured_species
        if use_log_2 is True:
            cuse_log_2 = True
        else:
            cuse_log_2 = False
        if species is not None:
            species = np.asarray(species).astype(np.intc)
            measured_species = arma.MakeIntCol(species)

        return _smfish.SmFishSnapshotLogLikelihood(self.this_[0],
                                   dist.this_[0],
                                   measured_species,
                                   cuse_log_2)

    def GetStates(self):
        cdef arma.Mat[int] observations = self.this_[0].GetObservations()
        cdef int [:,::1] view = <int[:observations.n_cols, :observations.n_rows]> observations.memptr()
        return np.copy(np.asarray(view))

    def GetFrequencies(self):
        cdef arma.Row[int] freq = self.this_[0].GetFrequencies()
        cdef int [:] view = <int[:freq.n_elem]> freq.memptr()
        return np.copy(np.asarray(view))




