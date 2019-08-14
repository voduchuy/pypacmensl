# distutils: language = c++


import pypacmensl.utils.environment as env
import numpy as np

cdef class SmFishSnapshot:
    def __cinit__(self, cnp.ndarray observations):
        if observations.dtype is not np.intc:
            observations = observations.astype(np.intc)
        if not observations.flags['C_CONTIGUOUS']:
            observations = np.ascontiguousarray(observations)

        cdef arma.Mat[int] observ = arma.MakeIntMat(observations)
        self.this_ = new _smfish.SmFishSnapshot(observ)
        self.env_ = []
        self.env_.append(env._PACMENSL_GLOBAL_ENV)

    def __dealloc__(self):
        if self.this_ is not NULL:
            del self.this_

    def LogLikelihood(self, FusedDistribution dist, species=None, use_log2 = False):
        cdef bool cuse_log_2
        cdef arma.Col[int] measured_species
        if use_log2 is True:
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

    def Gradient(self, SensDiscreteDistribution dist, species=None, use_log2 = False):
        cdef bool cuse_log_2
        cdef arma.Col[int] measured_species
        if use_log2 is True:
            cuse_log_2 = True
        else:
            cuse_log_2 = False
        if species is not None:
            species = np.asarray(species).astype(np.intc)
            measured_species = arma.MakeIntCol(species)

        cdef int ierr
        cdef vector[PetscReal] grad
        ierr = _smfish.SmFishSnapshotGradient(self.this_[0],
                                   dist.this_[0],
                                   grad,
                                   measured_species,
                                   cuse_log_2)
        assert(ierr==0)
        return np.copy(np.asarray(grad))

    def GetStates(self):
        cdef arma.Mat[int] observations = self.this_[0].GetObservations()
        cdef int [:,::1] view = <int[:observations.n_cols, :observations.n_rows]> observations.memptr()
        return np.copy(np.asarray(view))

    def GetFrequencies(self):
        cdef arma.Row[int] freq = self.this_[0].GetFrequencies()
        cdef int [:] view = <int[:freq.n_elem]> freq.memptr()
        return np.copy(np.asarray(view))
