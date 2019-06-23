from libcpp cimport bool
from libcpp.vector cimport *
cimport state_set
cimport fsp_solver
cimport arma4cy as arma
cimport mpi4py.MPI as mpi
cimport mpi4py.libmpi as libmpi
cimport numpy as cnp
import numpy as np
cimport discrete_distribution as cdd

cdef public int call_py_prop_obj(object, const int reaction, const int num_species, const int num_states, const int* states, double* outputs, void* args) except -1:
    cdef int[:,::1] state_view = <int[:num_states,:num_species]> states
    cdef double[::1] out_view = <double[:num_states]> outputs
    state_np = np.asarray(state_view)
    out_np = np.asarray(out_view)
    cdef double propensity
    try:
        object(reaction, state_np, out_np)
    except:
        return -1
    return 0

cdef public int call_py_t_fun_obj (object, double t, int num_coefs, double* outputs, void* args) except -1:
    cdef double[::1] out_view = <double[:num_coefs]> outputs
    try:
        object(t, out_view)
    except:
        return -1
    return 0

cdef class FspSolverMultiSinks:
    cdef mpi.Comm comm_
    cdef fsp_solver.FspSolverMultiSinks* this_;
    cdef fsp_solver.Model model_;

    def __cinit__(self, mpi.Comm comm = None):
        if comm is not None:
            self.comm_ = comm.Dup()
        else:
            self.comm_ = mpi.COMM_SELF.Dup()

        self.this_ = new fsp_solver.FspSolverMultiSinks(self.comm_.ob_mpi)

    def __dealloc_(self):
        if not self.this_ == NULL:
            del self.this_

    def SetModel(self, cnp.ndarray stoich_matrix, t_fun, propensity):
        cdef int ierr
        if stoich_matrix.dtype is not np.intc:
            stoich_matrix = stoich_matrix.astype(np.intc)
        cdef arma.Mat[int] stoich_matrix_arma = arma.Mat[int](<int*>stoich_matrix.data, stoich_matrix.shape[1], stoich_matrix.shape[0])
        self.model_.stoichiometry_matrix_ = stoich_matrix_arma
        self.model_.t_fun_ = fsp_solver.PyTFunWrapper(t_fun)
        self.model_.prop_ = fsp_solver.PyPropWrapper(propensity)
        ierr = self.this_[0].SetModel(self.model_)
        assert(ierr==0)

    def SetInitialDist(self, cnp.ndarray X0, cnp.ndarray p0):
        cdef int ierr = 0
        assert(X0.ndim == 2)
        assert(p0.ndim == 1)
        if X0.dtype is not np.intc:
            X0 = X0.astype(np.intc)
        if p0.dtype is not np.double:
            p0 = p0.astype(np.double)
        if not X0.flags['C_CONTIGUOUS']:
            X0 = np.ascontiguousarray(X0)
        if not p0.flags['C_CONTIGUOUS']:
            p0 = np.ascontiguousarray(p0)

        cdef arma.Mat[int] X0_arma = arma.Mat[int](<int*> X0.data, X0.shape[1], X0.shape[0], 0, 1)
        cdef arma.Col[fsp_solver.PetscReal] p0_arma = arma.Col[fsp_solver.PetscReal](<double*> p0.data, p0.size, 0, 1)
        ierr = self.this_[0].SetInitialDistribution(X0_arma, p0_arma)
        assert(ierr == 0)

    def SetFspShape(self, constr_fun, cnp.ndarray constr_bound, cnp.ndarray exp_factors = None):
        if constr_fun is not None:
            self.this_[0].SetConstraintFunctions(state_set.PyConstrWrapper(constr_fun))

        if constr_bound.dtype is not np.intc:
            constr_bound = constr_bound.astype(np.intc)
        if not constr_bound.flags['C_CONTIGUOUS']:
            constr_bound = np.ascontiguousarray(constr_bound)
        cdef arma.Row[int] bound_arma = arma.Row[int](<int*>constr_bound.data, constr_bound.size, 0, 1)
        self.this_[0].SetInitialBounds(bound_arma)

        if exp_factors is None:
            exp_factors = np.empty(constr_bound.size, dtype=np.double)
            exp_factors.fill(0.25)

        exp_factors = exp_factors.astype(np.double)
        exp_factors = np.ascontiguousarray(exp_factors)
        cdef arma.Row[fsp_solver.PetscReal] exp_factors_arma = arma.Row[fsp_solver.PetscReal](<double*>exp_factors.data, exp_factors.size, 0, 1)
        self.this_[0].SetExpansionFactors(exp_factors_arma)

    def SetVerbosity(self, int level):
        self.this_[0].SetVerbosity(level)

    def SetUp(self):
        self.this_[0].SetUp()

    def Solve(self, double t_final, double fsp_tol):
        solution = DiscreteDistribution()
        solution.this_[0] = self.this_[0].Solve(t_final, fsp_tol)
        return solution

    def SolveTspan(self, tspan, double fsp_tol):
        cdef int ntspan = tspan.size
        snapshots = []
        cdef vector[cdd.DiscreteDistribution] snapshots_c
        snapshots_c = self.this_[0].SolveTspan(tspan, fsp_tol)
        for i in range(0, ntspan):
            solution = DiscreteDistribution()
            solution.this_[0] = snapshots_c[i]
            snapshots.append(solution)
        return snapshots

    def ClearState(self):
        self.this_[0].DestroySolverState()


