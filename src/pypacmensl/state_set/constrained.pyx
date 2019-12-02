# distutils: language = c++

from libc.stdlib cimport malloc, free
cimport mpi4py.MPI as MPI
cimport numpy as cnp

from pypacmensl.callbacks.pacmensl_callbacks cimport *
from pypacmensl cimport libpacmensl
cimport pypacmensl.arma.arma4cy as arma
import pypacmensl.utils.environment as environment

import numpy as np
import mpi4py.MPI as mpi

cdef class StateSetConstrained:
    cdef:
        libpacmensl.StateSetConstrained* this_
        object env

    def __cinit__(self, MPI.Comm comm):
        cdef MPI.Comm comm_
        if comm is not None:
            comm_ = comm.Dup()
        else:
            comm_ = mpi.COMM_WORLD.Dup()

        self.this_ = new libpacmensl.StateSetConstrained(comm_.ob_mpi)
        self.env = []
        self.env.append(environment._PACMENSL_GLOBAL_ENV)

    def __dealloc__(self):
        if (self.this_ != NULL):
            del self.this_

    cpdef SetStoichiometry(self, cnp.ndarray sm):
        if sm.dtype is not np.intc:
            sm = sm.astype(np.intc)
        if not sm.flags['C_CONTIGUOUS']:
            sm = np.ascontiguousarray(sm)

        cdef arma.Mat[int] sm_arma = arma.Mat[int](<int*>sm.data, sm.shape[1], sm.shape[0])
        ierr = self.this_[0].SetStoichiometryMatrix(sm_arma)
        if ierr is not 0:
            raise RuntimeError("SetStoichiometryMatrix() returns error")


    def AddStates(self, cnp.ndarray X):
        if X.dtype is not np.intc:
            X = X.astype(np.intc)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        cdef arma.Mat[int] X_arma = arma.Mat[int](<int*> X.data, X.shape[1], X.shape[0])

        ierr = self.this_[0].AddStates(X_arma)
        if ierr is not 0:
            raise RuntimeError("AddStates() returns error")

    cpdef SetConstraint(self, constr_fun, init_bounds):
        assert isinstance(constr_fun, object)
        cdef int num_constraints = init_bounds.size
        cdef int* c_bounds = <int*> malloc(num_constraints*sizeof(int))
        for i in range(0, num_constraints):
            c_bounds[i] = init_bounds[i]
        if constr_fun is not None:
            ierr = self.this_[0].SetShape(num_constraints, call_py_constr_obj, c_bounds, <void*>constr_fun)
        else:
            ierr = self.this_[0].SetShapeBounds(num_constraints, c_bounds)
        free(c_bounds)
        if ierr is not 0:
            raise RuntimeError("SetConstraint() returns error")

    def GetNumSpecies(self):
        return self.this_[0].GetNumSpecies()

    def GetNumConstraints(self):
        cdef nc
        nc = self.this_[0].GetNumConstraints()
        return nc

    def SetUp(self):
        ierr = self.this_[0].SetUp()
        if ierr is not 0:
            raise RuntimeError("SetUp() returns error")
        return

    def Expand(self):
        ierr = self.this_[0].Expand()
        if ierr is not 0:
            raise RuntimeError("Expand() returns error")
        return

    def GetStates(self, return_fortran_array=False):
        cdef int num_states = self.this_[0].GetNumLocalStates()
        cdef int num_species = self.this_[0].GetNumSpecies()
        if num_states==0:
          return np.empty(shape=(0,num_species))

        states = np.ascontiguousarray(np.empty((num_states, num_species), dtype=np.intc, order='C'))

        cdef int[:,::1] states_view = states

        self.this_[0].CopyStatesOnProc(num_states, &states_view[0,0])
        if return_fortran_array:
            states = np.asfortranarray(states)
        return states

    def SetLBMethod(self, method="Graph"):
        if method is None:
            return
        method = method.lower()
        cdef libpacmensl.PartitioningType cmethod
        if (method == "graph"):
            cmethod = libpacmensl.GRAPH
        if (method == "block"):
            cmethod = libpacmensl.BLOCK
        if (method == "hypergraph"):
            cmethod = libpacmensl.HYPERGRAPH
        cdef int ierr = self.this_[0].SetLoadBalancingScheme(cmethod)
        assert(ierr == 0)

