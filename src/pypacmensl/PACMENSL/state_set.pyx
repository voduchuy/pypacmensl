from libc.stdlib cimport malloc, free
cimport mpi4py.MPI as MPI
cimport numpy as cnp
cimport state_set
import numpy as np
import ctypes
cimport arma4cy as arma
import mpi4py.MPI as mpi

cdef public int call_py_constr_obj(obj, int num_species, int num_constr, int n_states, int *states, int *outputs, void *args) except -1:
    cdef int[:,::1] states_view = <int[:n_states, :num_species]> states
    cdef int[::1] outputs_view = <int[:n_states * num_constr]> outputs
    states_np = np.asarray(states_view)
    outputs_np = np.asarray(outputs_view)
    try:
        obj(states_np, outputs_np)
    except:
        return -1
    return 0

cdef class StateSetConstrained:
    cdef state_set.StateSetConstrained* this_
    cdef state_set.PyConstrWrapper constr_

    def __cinit__(self, MPI.Comm comm):
        cdef MPI.Comm comm_
        if comm is not None:
            comm_ = comm.Dup()
        else:
            comm_ = mpi.COMM_WORLD.Dup()

        self.this_ = new state_set.StateSetConstrained(comm_.ob_mpi)

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
            self.constr_ = state_set.PyConstrWrapper(constr_fun)
            ierr = self.this_[0].SetShape(num_constraints, self.constr_, c_bounds, NULL)
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

        cdef int[:,::1] states_view = states.data

        # self.this_[0].CopyStatesOnProc(num_states, &states_view[0,0])
        # if return_fortran_array:
        #     states = np.asfortranarray(states)
        return states

    def SetLBMethod(self, method="Graph"):
        if method is None:
            return
        method = method.lower()
        cdef state_set.PartitioningType cmethod
        if (method == "graph"):
            cmethod = state_set.GRAPH
        if (method == "block"):
            cmethod = state_set.BLOCK
        if (method == "hypergraph"):
            cmethod = state_set.HYPERGRAPH
        cdef int ierr = self.this_[0].SetLoadBalancingScheme(cmethod)
        assert(ierr == 0)

