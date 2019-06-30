from libcpp cimport bool
from libcpp.vector cimport *
from pacmensl_callbacks cimport *
cimport state_set
cimport fsp_solver as cfsp
cimport arma4cy as arma
cimport mpi4py.MPI as mpi
cimport mpi4py.libmpi as libmpi
cimport numpy as cnp
import numpy as np
cimport discrete_distribution as cdd

cdef class FspSolverMultiSinks:
    cdef cfsp.FspSolverMultiSinks* this_;

    def __cinit__(self, mpi.Comm comm = None):
        if comm is None:
            comm = mpi.COMM_WORLD.Dup()

        self.this_ = new cfsp.FspSolverMultiSinks(comm.ob_mpi)

    def __dealloc_(self):
        if  self.this_ is not NULL:
            del self.this_

    def SetModel(self, cnp.ndarray stoich_matrix, t_fun, propensity):
        cdef int ierr
        if stoich_matrix.dtype is not np.intc:
            stoich_matrix = stoich_matrix.astype(np.intc)
        if not stoich_matrix.flags['C_CONTIGUOUS']:
            stoich_matrix = np.ascontiguousarray(stoich_matrix)

        cdef arma.Mat[int] stoich_matrix_arma = arma.Mat[int](<int*>stoich_matrix.data, stoich_matrix.shape[1], stoich_matrix.shape[0], 0, 1)

        cdef cfsp.Model model_ = cfsp.Model(stoich_matrix_arma, PyTFunWrapper(t_fun), PyPropWrapper(propensity))

        ierr = self.this_[0].SetModel(model_)

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

        cdef arma.Mat[int] X0_arma = arma.Mat[int](<int*> X0.data, X0.shape[1], X0.shape[0], 1, 1)
        cdef arma.Col[cfsp.PetscReal] p0_arma = arma.Col[cfsp.PetscReal](<double*> p0.data, p0.size, 1, 1)
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
        cdef arma.Row[cfsp.PetscReal] exp_factors_arma = arma.Row[cfsp.PetscReal](<double*>exp_factors.data, exp_factors.size, 0, 1)
        self.this_[0].SetExpansionFactors(exp_factors_arma)

    def SetVerbosity(self, int level):
        self.this_[0].SetVerbosity(level)

    def SetLBMethod(self, method="Graph"):
        if method is None:
            return
        method = method.lower()
        cdef cfsp.PartitioningType cmethod
        if (method == "graph"):
            cmethod = cfsp.GRAPH
        if (method == "block"):
            cmethod = cfsp.BLOCK
        if (method == "hypergraph"):
            cmethod = cfsp.HYPERGRAPH
        cdef int ierr = self.this_[0].SetLoadBalancingMethod(cmethod)
        assert(ierr == 0)

    def SetUp(self):
        self.this_[0].SetUp()

    def Solve(self, double t_final, double fsp_tol):
        solution = DiscreteDistribution()
        try:
            solution.this_[0] = self.this_[0].Solve(t_final, fsp_tol)
        except RuntimeError:
            print("Runtime error!")
            return None
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
        self.this_[0].ClearState()


