# distutils: language=c++

import pypacmensl.utils.environment as env
import numpy as np

cdef class SensDiscreteDistribution:
    def __cinit__(self):
        self.this_ = new _sdd.SensDiscreteDistribution()
        self.env_ = [env._PACMENSL_GLOBAL_ENV]

    def __dealloc__(self):
        if self.this_ is not NULL:
            del self.this_

    cpdef GetNumParameters(self):
        """
        Get number of sensitivity parameters.
        
        Returns
        -------
        num_parameters: int 
            Number of sensitivity parameters.
            
        """
        return self.this_[0].dp_.size()

    cpdef GetStatesViewer(self):
        cdef int* state_ptr
        cdef:
            int num_states
            int num_species
        self.this_[0].GetStateView(num_states, num_species, state_ptr)
        cdef int[:,::1] states_view = <int[:num_states, :num_species]> state_ptr
        return np.asarray(states_view)

    cpdef GetProbViewer(self):
        """
        SensDiscreteDistribution.GetProbViewer() -> numpy.ndarray
        
        Returns
        -------
        A numpy array wrapper for the on-processor section of the probability vector.

        Notes
        -----
        Row i of the output array of GetStatesViewer() maps to the i-th entry of the probability vector
        returned by this function.

        When done with modifying the array, RestoreProViewer() must be called to return ownership to the object.
        """
        cdef:
            double* prob_ptr
            int num_states
        self.this_[0].GetProbView(num_states, prob_ptr)
        cdef double[::1] prob_view = <double[:num_states]> prob_ptr
        return np.asarray(prob_view)

    cpdef RestoreProbViewer(self, double[::1] viewer):
        cdef double* prob_ptr = &viewer[0]
        self.this_[0].RestoreProbView(prob_ptr)
        return None

    cpdef GetSensViewer(self, int iS):
        cdef:
            double* s_ptr
            int num_states
        if iS > <int>self.this_[0].dp_.size():
            raise RuntimeError("Requested sensitivity vector does not exist.")
        self.this_[0].GetSensView(iS, num_states, s_ptr)
        cdef double[::1] sens_view = <double[:num_states]> s_ptr
        return np.asarray(sens_view)

    cpdef RestoreSensViewer(self, int iS, cnp.ndarray sensview):
        if iS > self.this_[0].dp_.size():
            raise RuntimeError("Requested sensitivity vector does not exist.")
        cdef _sdd.PetscReal[::1] sensmem = sensview
        cdef _sdd.PetscReal* ptr = &sensmem[0]
        self.this_[0].RestoreSensView(iS, ptr)

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
        cdef arma.Col[double] marginal = _sdd.Compute1DMarginal(self.this_[0], species)
        cdef double* marginal_ptr = marginal.memptr()
        cdef double[::1] marginal_view = <double[:marginal.n_elem]> marginal_ptr
        return np.copy(np.asarray(marginal_view))

    def SensMarginal(self, int iS, int species):
        """
        Compute the sensitivity of the marginal distribution.

        Parameters
        -----------

        iS: int
            Index of the sensitivity parameter.
        species: int
            Index of the species to compute the marginal distribution for.

        Returns
        --------
        svec: 1-D numpy array
            Sensitivity vector corresponding to the marginal distribution of the input species.

        """
        cdef arma.Col[_sdd.PetscReal] smarginal
        cdef int ierr = _sdd.Compute1DSensMarginal(self.this_[0],
                              iS, species, smarginal)
        if ierr is not 0:
            raise RuntimeError("Error encountered when calling smfish routine.")

        cdef _sdd.PetscReal[::1] smarginal_view = <_sdd.PetscReal[:smarginal.n_elem]> smarginal.memptr()
        return np.copy(np.asarray(smarginal_view))

    def ComputeFIM(self):
        cdef arma.Mat[_sdd.PetscReal] fim
        cdef int ierr = _sdd.ComputeFIM(self.this_[0], fim)
        if ierr is not 0:
            raise RuntimeError("Error encountered when calling smfish routine.")

        cdef _sdd.PetscReal[::, ::1] fim_view = <_sdd.PetscReal[:fim.n_cols, :fim.n_rows]> fim.memptr()
        return np.copy(np.asarray(fim_view))

    def WeightedAverage(self, iS, nout, weightfun):
        """
        WeightedAverage(iS, nout, weightfun)

        Evaluate the expression E(f(X)) or its partial derivatives where X is the random variable distributed according to the probability distribution stored by this object. The function f can be vector-valued.

        Parameters
        ----------
        nout : int
            Dimensionality of the output function f. E.g., if f(x) = (f1(x), f2(x), f3(x)) then nout = 3.

        iS : int
            Index of the sensitivity parameter. If `iS=-1`, the output will be E(f(x)), otherwise it will be the partial derivative of E(f(x)) with respect to the `iS`-th parameter.

        weightfun : Callable
            Python function to evaluate f with syntax `weightfun(x, out)` that write the evaluation of `f(x)` into the output array `out`.

        Returns
        -------
        fout: 1-D array
            Output vector of length `nout`.

        """
        fout = np.zeros((nout, ), dtype=np.double)
        fout = np.ascontiguousarray(fout)
        cdef double[::1] foutview = fout
        cdef int ierr = self.this_[0].WeightedAverage(iS, nout, &foutview[0], call_py_weight_fun, <void*> weightfun)
        if ierr != 0:
            raise RuntimeError('An error was encountered during the call to the weight function.')
        return fout
