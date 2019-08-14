# distutils: language = c++
import numpy as np
import pypacmensl.utils.environment as environment

cdef class StationaryFspSolverMultiSinks:
    def __cinit__(self, mpi.Comm comm = None):
        if comm is None:
            comm = mpi.Comm.COMM_WORLD.Dup()
        self.this_ = new _fsp.StationaryFspSolverMultiSinks(comm.ob_mpi)
        self.set_up = False
        self.env = []
        self.env.append(environment._PACMENSL_GLOBAL_ENV)

    def __dealloc__(self):
        if self.this_ is not NULL:
            del self.this_

    def SetModel(self, cnp.ndarray stoich_matrix, propensity_t, propensity_x):
        """
        def SetModel(self, stoich_matrix, t_fun, propensity)

        Set the stochastic chemical kinetics model to be solved.

        :param stoich_matrix: stoichiometry matrix stored in an array. Each row is a stoichiometry vector.

        :param propensity_t:
                callable object for computing the time-dependent coefficients. It must have signature
                        def propensity_t( t, out )
                where t is a scalar, and out is an array to be written to with the values of the coefficients at time t.

        :param propensity_x:
                callable object representing the time-independent factors of the propensities. It must have signature
                        def propensity_x( i_reaction, states, out )
                where i_reaction is the reaction index, states[:, :] an array where each row is an input state, and out[:] is
                the output array to be written to.
        """
        cdef int ierr
        if stoich_matrix.dtype is not np.intc:
            stoich_matrix = stoich_matrix.astype(np.intc)
        if not stoich_matrix.flags['C_CONTIGUOUS']:
            stoich_matrix = np.ascontiguousarray(stoich_matrix)

        cdef arma.Mat[int] stoich_matrix_arma = arma.Mat[int](<int*> stoich_matrix.data, stoich_matrix.shape[1],
                                                              stoich_matrix.shape[0], True, False)

        cdef _fsp.Model model_ = _fsp.Model(stoich_matrix_arma, call_py_t_fun_obj, call_py_prop_obj,
                                            <void*> propensity_t, <void*> propensity_x)

        ierr = self.this_[0].SetModel(model_)

        assert (ierr == 0)

    def SetInitialDist(self, cnp.ndarray X0, cnp.ndarray p0):
        """
        Set initial distribution. This must be called before .Solve() or .SolveTspan().

        :param X0: Initial states. Must be a 2-D array-like object, each row is a state.

        :param p0: Initial probabilities. Must be a 1-D array-like object, each entry maps to a row in X0.

        :return: None

        """
        cdef int ierr = 0

        assert (X0.ndim == 2)
        assert (p0.ndim == 1)

        if X0.dtype is not np.intc:
            X0 = X0.astype(np.intc)
        if p0.dtype is not np.double:
            p0 = p0.astype(np.double)
        if not X0.flags['C_CONTIGUOUS']:
            X0 = np.ascontiguousarray(X0)
        if not p0.flags['C_CONTIGUOUS']:
            p0 = np.ascontiguousarray(p0)

        cdef arma.Mat[int] X0_arma = arma.Mat[int](<int*> X0.data, X0.shape[1], X0.shape[0], True, False)
        cdef arma.Col[double] p0_arma = arma.Col[double](<double*> p0.data, p0.size, True, False)

        ierr = self.this_[0].SetInitialDistribution(X0_arma, p0_arma)
        assert (ierr == 0)

    def SetFspShape(self, constr_fun, cnp.ndarray constr_bound, cnp.ndarray exp_factors = None):
        """
        Set constraint functions and bounds that determine the shape of the truncated state space.

        :param constr_fun: Constraint function. Callable in the form
                    def constr_fun(states, out)
        which, upon returning, populate out[i, j] with the value of the j-th constraint function evaluated at states[i,:].

        :param constr_bound: array of bounds.

        :param exp_factors: array of expansion factors. s
        :type exp_factors:
        :return:
        :rtype:
        """
        if constr_fun is not None:
            self.this_[0].SetConstraintFunctions(call_py_constr_obj,<void*>constr_fun)

        if constr_bound.dtype is not np.intc:
            constr_bound = constr_bound.astype(np.intc)
        if not constr_bound.flags['C_CONTIGUOUS']:
            constr_bound = np.ascontiguousarray(constr_bound)
        cdef arma.Row[int] bound_arma = arma.Row[int](<int*> constr_bound.data, constr_bound.size, 0, 1)
        self.this_[0].SetInitialBounds(bound_arma)

        if exp_factors is None:
            exp_factors = np.empty(constr_bound.size, dtype=np.double)
            exp_factors.fill(0.25)

        exp_factors = exp_factors.astype(np.double)
        exp_factors = np.ascontiguousarray(exp_factors)
        cdef arma.Row[_fsp.PetscReal] exp_factors_arma = arma.Row[_fsp.PetscReal](
                <double*> exp_factors.data,
                exp_factors.size, 0, 1)
        self.this_[0].SetExpansionFactors(exp_factors_arma)

    def SetVerbosity(self, int level):
        self.this_[0].SetVerbosity(level)

    def SetLBMethod(self, method="Graph"):
        if method is None:
            return
        method = method.lower()
        cdef _fsp.PartitioningType cmethod
        if (method == "graph"):
            cmethod = _fsp.GRAPH
        if (method == "block"):
            cmethod = _fsp.BLOCK
        if (method == "hypergraph"):
            cmethod = _fsp.HYPERGRAPH
        cdef int ierr = self.this_[0].SetLoadBalancingMethod(cmethod)
        assert (ierr == 0)

    def SetUp(self):
        """
        Allocate resources for the FSP solver. This method is called implicitly within Solve() and SolveTspan() so normal
        users do not need to bother with it.
        :return:
        :rtype:
        """
        self.this_[0].SetUp()
        self.set_up = True

    def Solve(self, double fsp_tol):
        if self.set_up is not True:
            self.SetUp()

        cdef cdd.DiscreteDistribution solution = cdd.DiscreteDistribution()
        try:
            solution.this_[0] = self.this_[0].Solve(fsp_tol)
        except RuntimeError:
            print("Runtime error!")
            return None
        return solution

    def ClearState(self):
        self.this_[0].ClearState()
        self.set_up = False
