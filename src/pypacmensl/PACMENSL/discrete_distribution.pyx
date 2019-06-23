cimport discrete_distribution as dd
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
cimport state_set
cimport numpy as cnp
import numpy as np

cdef class DiscreteDistribution:
    cdef dd.DiscreteDistribution* this_;

    def __cinit__(self):
        self.this_ = new dd.DiscreteDistribution()

    def __dealloc__(self):
        if (self.this_ != NULL):
            del self.this_

    cpdef GetStatesViewer(self):
        cdef int* state_ptr
        cdef:
            int num_states
            int num_species
        self.this_[0].GetStateView(num_states, num_species, state_ptr)
        cdef int[:,::1] states_view = <int[:num_states, :num_species]> state_ptr
        return states_view

    cpdef GetProbViewer(self):
        cdef:
            double* prob_ptr
            int num_states
        self.this_[0].GetProbView(num_states, prob_ptr)
        cdef double[::1] prob_view = <double[:num_states]> prob_ptr
        return prob_view

    cpdef RestoreProbViewer(self, double[::1] viewer):
        cdef double* prob_ptr = &viewer[0]
        self.this_[0].RestoreProbView(prob_ptr)

    def Marginal(self, int species):
        cdef arma.Col[double] marginal = dd.Compute1DMarginal(self.this_[0], species)
        cdef double* marginal_ptr = marginal.memptr()
        cdef double[::1] marginal_view = <double[:marginal.n_elem]> marginal_ptr
        return np.copy(np.asarray(marginal_view))

