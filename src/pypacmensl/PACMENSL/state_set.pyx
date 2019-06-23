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
    cdef MPI.Comm comm_
    cdef state_set.StateSetConstrained* this_
    cdef state_set.PyConstrWrapper constr_
    cdef int num_species_

    def __cinit__(self, MPI.Comm comm, n_species, str part_type='graph', str repart_type='repartition'):

        if comm is not None:
            self.comm_ = comm.Dup()
        else:
            self.comm_ = mpi.COMM_SELF.Dup()
        part_type = part_type.lower()
        repart_type = repart_type.lower()

        cdef int num_species = n_species
        cdef state_set.PartitioningType part
        cdef state_set.PartitioningApproach repart

        if part_type == 'graph':
          part = state_set.GRAPH
        elif part_type == 'hypergraph':
          part = state_set.HYPERGRAPH
        elif part_type == 'block':
          part = state_set.BLOCK
        else:
          part = state_set.BLOCK

        if repart_type == 'repartition':
          repart = state_set.REPARTITION
        elif repart_type == 'from scratch':
          repart = state_set.FROMSCRATCH
        else:
          repart = state_set.REFINE

        self.this_ = new state_set.StateSetConstrained(self.comm_.ob_mpi, n_species, part, repart)
        self.num_species_ = n_species

    def __dealloc__(self):
        if (self.this_ != NULL):
            del self.this_

    cpdef SetStoichiometry(self, cnp.ndarray sm):
        if sm.dtype is not np.intc:
            sm = sm.astype(np.intc)
        cdef arma.Mat[int] sm_arma = arma.Mat[int](<int*>sm.data, sm.shape[1], sm.shape[0])
        self.this_[0].SetStoichiometryMatrix(sm_arma)

    def AddStates(self, cnp.ndarray X):
        if not X.dtype == np.intc:
            X = X.astype(np.intc)
        cdef arma.Mat[int] X_arma = arma.Mat[int](<int*> X.data, X.shape[1], X.shape[0])
        self.this_[0].AddStates(X_arma)

    cpdef SetConstraint(self, constr_fun, init_bounds):
        assert isinstance(constr_fun, object)
        cdef int num_constraints = init_bounds.shape[0]
        cdef int* c_bounds = <int*> malloc(num_constraints*sizeof(int))
        for i in range(0, num_constraints):
            c_bounds[i] = init_bounds[i]
        self.constr_ = state_set.PyConstrWrapper(constr_fun)
        self.this_[0].SetShape(num_constraints, self.constr_, c_bounds, NULL)
        free(c_bounds)

    def GetNumConstraints(self):
        cdef nc
        nc = self.this_[0].GetNumConstraints()
        return nc

    def Expand(self):
        self.this_[0].Expand()
        return

    def GetStates(self):
        cdef int num_states = self.this_[0].GetNumLocalStates()
        if num_states==0:
          return np.empty(shape=(0,self.num_species_))

        states = np.ascontiguousarray(np.empty((num_states, self.num_species_), dtype=np.intc))
        cdef int[:,::1] states_view = states.data
        self.this_[0].CopyStatesOnProc(num_states, &states_view[0,0])
        return states

    def GetComm(self):
        return MPI.Comm(self.comm_)

