# distutils: language = c++
import numpy as np
import pypacmensl.utils.environment as env

cdef class SensFspSolverMultiSinks:
    def __cinit__(self, mpi.Comm comm = None):
        if comm is None:
            comm = mpi.Comm.COMM_WORLD.Dup()
        self.this_ = new _fsp.SensFspSolverMultiSinks(comm.ob_mpi)
        self.set_up_ = False
        self.env_ = [env._PACMENSL_GLOBAL_ENV]

    def __dealloc__(self):
        if self.this_ is not NULL:
            del self.this_

    def SetModel(self, cnp.ndarray stoich_matrix, propensity_t, propensity_x, d_propensity_t, d_propensity_x,
                 cnp.ndarray sparsity_patterns = None):
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

        :param d_propensity_t:
                list of callable objects for the time-dependent coefficients of the derivatives of propensity functions. This
                list has the same number of elements as sensitivity parameters.

        :param d_propensity_x:
                list of callable objects for the state-dependent factors of the derivatives of propensity functions. This list has the same
                number of elements as sensitivity parameters.

        :param sparsity_patterns:
                sparsity pattern of the propensity derivatives. If provided, must be an array of size num_parameters x num_reactions, with sparsity_patterns[i][j] = 1
                if the derivative of the j-th propensity wrt the i-th parameter is nonzero, and sparsity_pattersn[i][j] = 0 otherwise.
        """
        cdef int ierr

        if len(d_propensity_t) != len(d_propensity_x):
            raise RuntimeError("Input d_propensity_t and d_propensity_x must have the same number of elements")

        # convert stoich_matrix to armadillo array
        if stoich_matrix.dtype is not np.intc:
            stoich_matrix = stoich_matrix.astype(np.intc)
        if not stoich_matrix.flags['C_CONTIGUOUS']:
            stoich_matrix = np.ascontiguousarray(stoich_matrix)
        if sparsity_patterns.dtype is not np.intc:
            sparsity_patterns = sparsity_patterns.astype(np.intc)

        cdef arma.Mat[int] stoich_matrix_arma = arma.Mat[int](<int*> stoich_matrix.data, stoich_matrix.shape[1],
                                                              stoich_matrix.shape[0], 0, 1)

        cdef vector[int] tv_react
        prop_t_ptr = <void*> propensity_t
        for i in range(0, stoich_matrix.shape[0]):
            tv_react.push_back(i)



        cdef _fsp.SensModel model_
        model_.stoichiometry_matrix_ = stoich_matrix_arma
        model_.prop_t_ = call_py_t_fun_obj
        model_.prop_x_ = call_py_prop_obj
        model_.prop_t_args_ = <void*> propensity_t
        model_.prop_x_args_ = <void*> propensity_x

        model_.num_parameters_ = model_.dprop_x_.size()

        model_.dprop_t_.reserve(len(d_propensity_t))
        model_.dprop_t_args_.reserve(model_.num_parameters_)
        for f in d_propensity_t:
            model_.dprop_t_.push_back(call_py_t_fun_obj)
            model_.dprop_t_args_.push_back(<void*> f)

        model_.dprop_x_.reserve(model_.num_parameters_)
        model_.dprop_x_args_.reserve(model_.num_parameters_)
        for f in d_propensity_x:
            model_.dprop_x_.push_back(call_py_prop_obj)
            model_.dprop_x_args_.push_back(<void*> f)

        cdef vector[int] ic
        cdef vector[int] irow
        irow.push_back(0)
        if sparsity_patterns is not None:
            for i in range(0, sparsity_patterns.shape[0]):
                for j in range(0, sparsity_patterns.shape[1]):
                    if sparsity_patterns[i, j] == 1:
                        ic.push_back(j)
                irow.push_back(ic.size())
        model_.dpropensity_ic_ = ic
        model_.dpropensity_rowptr_ = irow
        model_.tv_reactions_ = tv_react

        ierr = self.this_[0].SetModel(model_)

        assert (ierr == 0)

    def SetInitialDist(self, cnp.ndarray X0, cnp.ndarray p0, s0):
        """
        Set initial distribution. This must be called before .Solve() or .SolveTspan().

        :param X0: Initial states. Must be a 2-D array-like object, each row is a state.

        :param p0: Initial probabilities. Must be a 1-D array-like object, each entry maps to a row in X0.

        :param s0: List of initial sensitivities. Each vector in this list must have the same layout as p0.

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

        cdef arma.Mat[int] X0_arma = arma.Mat[int](<int*> X0.data, X0.shape[1], X0.shape[0], 1, 1)
        cdef arma.Col[_fsp.PetscReal] p0_arma = arma.Col[_fsp.PetscReal](<double*> p0.data, p0.size, 1, 1)
        cdef vector[arma.Col[_fsp.PetscReal]] s0_vector
        cdef cnp.ndarray v
        for i in range(0, len(s0)):
            v = np.ascontiguousarray(s0[i]).astype(np.double)
            s0_vector.push_back(arma.Col[_fsp.PetscReal](<double*> v.data, v.size, 1, 1))
        ierr = self.this_[0].SetInitialDistribution(X0_arma, p0_arma, s0_vector)
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
            self.this_[0].SetConstraintFunctions(call_py_constr_obj, <void*> constr_fun)

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
        self.set_up_ = True

    def Solve(self, double t_final, double fsp_tol):
        if self.set_up_ is not True:
            self.SetUp()

        cdef sdd.SensDiscreteDistribution solution = sdd.SensDiscreteDistribution()
        try:
            solution.this_[0] = self.this_[0].Solve(t_final, fsp_tol)
        except RuntimeError:
            print("Runtime error!")
            return None
        return solution

    def SolveTspan(self, tspan, double fsp_tol):
        if self.set_up_ is not True:
            self.SetUp()
        cdef int ntspan = tspan.size
        snapshots = []
        cdef vector[_sdd.SensDiscreteDistribution] snapshots_c
        snapshots_c = self.this_[0].SolveTspan(tspan, fsp_tol)
        cdef sdd.SensDiscreteDistribution solution
        for i in range(0, ntspan):
            solution = sdd.SensDiscreteDistribution()
            solution.this_[0] = snapshots_c[i]
            snapshots.append(solution)
        return snapshots

    def ClearState(self):
        self.this_[0].ClearState()
        self.set_up_ = False
