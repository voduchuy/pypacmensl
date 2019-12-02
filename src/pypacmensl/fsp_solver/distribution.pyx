# distutils: language = c++

import pypacmensl.utils.environment as env
import numpy as np

cdef class DiscreteDistribution:
    def __cinit__(self):
        self.this_ = new _dd.DiscreteDistribution()
        self.env_ = [env._PACMENSL_GLOBAL_ENV]

    def __dealloc__(self):
        if (self.this_ != NULL):
            del self.this_

    def GetStatesViewer(self):
        """
        :return: 
        :rtype: 
        """
        cdef int* state_ptr
        cdef:
            int num_states
            int num_species
        self.this_[0].GetStateView(num_states, num_species, state_ptr)
        cdef int[:,::1] states_view = <int[:num_states, :num_species]> state_ptr
        return states_view

    def GetProbViewer(self):
        cdef:
            double* prob_ptr
            int num_states
        self.this_[0].GetProbView(num_states, prob_ptr)
        cdef double[::1] prob_view = <double[:num_states]> prob_ptr
        return np.asarray(prob_view)

    def RestoreProbViewer(self, cnp.ndarray prob):
        cdef double[::1] viewer = prob
        cdef double* prob_ptr = &viewer[0]
        self.this_[0].RestoreProbView(prob_ptr)

    def Marginal(self, int species):
        cdef arma.Col[double] marginal = _dd.Compute1DMarginal(self.this_[0], species)
        cdef double* marginal_ptr = marginal.memptr()
        cdef double[::1] marginal_view = <double[:marginal.n_elem]> marginal_ptr
        return np.copy(np.asarray(marginal_view))

    def WeightedAverage(self, nout, weightfun):
        fout = np.zeros((nout, ), dtype=np.double)
        fout = np.ascontiguousarray(fout)
        cdef double[::1] foutview = fout
        cdef int ierr = self.this_[0].WeightedAverage(nout, &foutview[0], call_py_weight_fun, <void*> weightfun)
        if ierr != 0:
            raise RuntimeError('An error was encountered during the call to the weight function.')
        return fout

