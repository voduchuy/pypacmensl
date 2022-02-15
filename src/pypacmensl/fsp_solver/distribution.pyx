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
        Get a view into the states.

        Returns
        -------
        states_view: 2-D Numpy array
            Numpy array wrapper for the local states. Each row is a CME state.

        """
        cdef int* state_ptr
        cdef:
            int num_states
            int num_species
        self.this_[0].GetStateView(num_states, num_species, state_ptr)
        cdef int[:,::1] states_view = <int[:num_states, :num_species]> state_ptr
        return states_view

    def GetProbViewer(self):
        """
        GetProbViewer()

        Get the view of the local section of the probability vector.

        Returns
        -------
        prob_view: 1-D Numpy array
            Numpy array that wraps around the internal array of the probability vector.

        Note
        ----
        This call must be matched by `RestoreProbViewer()`.

        """
        cdef:
            double* prob_ptr
            int num_states
        self.this_[0].GetProbView(num_states, prob_ptr)
        cdef double[::1] prob_view = <double[:num_states]> prob_ptr
        return np.asarray(prob_view)

    def RestoreProbViewer(self, cnp.ndarray prob):
        """
        RestoreProbViewer(prob)

        Restore the local section of the probability vector.

        Parameters
        ----------
        prob : 1-D Numpy array
            The local section of the probability vector.

        Returns
        -------

        """
        cdef double[::1] viewer = prob
        cdef double* prob_ptr = &viewer[0]
        self.this_[0].RestoreProbView(prob_ptr)

    def Marginal(self, int species):
        """
        Marginal(species)

        Compute the 1-D marginal distribution.

        Parameters
        ----------
        species : int
            Index of the species.

        Returns
        -------

        """
        cdef arma.Col[double] marginal = _dd.Compute1DMarginal(self.this_[0], species)
        cdef double* marginal_ptr = marginal.memptr()
        cdef double[::1] marginal_view = <double[:marginal.n_elem]> marginal_ptr
        return np.copy(np.asarray(marginal_view))

    def WeightedAverage(self, nout, weightfun):
        """
        WeightedAverage(nout, weightfun)

        Evaluate the expression E(f(X)) where X is the random variable distributed according to the probability distribution stored by this object. The function f can be vector-valued.

        Parameters
        ----------
        nout : int
            Dimensionality of the output function f. E.g., if f(x) = (f1(x), f2(x), f3(x)) then nout = 3.

        weightfun : Callable
            Python function to evaluate f with syntax `weightfun(x, out)` that write the evaluation of `f(x)` into the output array `out`.

        Returns
        -------
        fout: 1-D array
            Output vector of length `nout`, the resulting of averaging f over all states weighted by their probabilities.
        """
        fout = np.zeros((nout, ), dtype=np.double)
        fout = np.ascontiguousarray(fout)
        cdef double[::1] foutview = fout
        cdef int ierr = self.this_[0].WeightedAverage(nout, &foutview[0], call_py_weight_fun, <void*> weightfun)
        if ierr != 0:
            raise RuntimeError('An error was encountered during the call to the weight function.')
        return fout

