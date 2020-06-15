# distutils: language = c++
import pypacmensl.utils.environment as env
import numpy as np
import mpi4py.MPI as mpi

cdef class FspSolverMultiSinks:
    def __cinit__(self, mpi.Comm comm = mpi.COMM_WORLD):
        self.this_ = new _fsp.FspSolverMultiSinks(comm.ob_mpi)
        self.env_ = [env._PACMENSL_GLOBAL_ENV]
        self.set_up_ = False

    def __dealloc__(self):
        if self.this_ is not NULL:
            del self.this_

    def SetModel(self, cnp.ndarray stoich_matrix, propensity_t, propensity_x, tv_reactions = None):
        """
        def SetModel(self, stoich_matrix, propensity_x, propensity_t, tv_reactions)

        Set the stochastic chemical kinetics model to be solved.

        :param stoich_matrix: stoichiometry matrix stored in an array. Each row is a stoichiometry vector.

        :param propensity_t:
                (Optional) callable object for computing the time-dependent coefficients. It must have signature
                        def propensity_t( t, out )
                where t is a scalar, and out is an array to be written to with the values of the coefficients at time t.
                This is only needed when solving a model with time-varying propensities, in which case you also need to
                input tv_reactions. If set to None, the model is assumed to have time-invariant propensities.

        :param propensity_x:
                callable object representing the time-independent factors of the propensities. It must have signature
                        def propensity_x( i_reaction, states, out )
                where i_reaction is the reaction index, states[:, :] an array where each row is an input state, and out[:] is
                the output array to be written to.

        :param tv_reactions:
                (Optional) array-like object that stores the indices of the reactions whose propensities are time-varying.
                If not specified or set to None, but with propensity_t specified, we assume that all reaction propensities are time-varying.
        """
        cdef int ierr
        if stoich_matrix.dtype is not np.intc:
            stoich_matrix = stoich_matrix.astype(np.intc)
        if not stoich_matrix.flags['C_CONTIGUOUS']:
            stoich_matrix = np.ascontiguousarray(stoich_matrix)

        cdef arma.Mat[int] stoich_matrix_arma = arma.Mat[int](<int*> stoich_matrix.data, stoich_matrix.shape[1],
                                                              stoich_matrix.shape[0], 0, 1)

        cdef void*prop_t_ptr
        cdef vector[int] tv_react

        if propensity_t is None:
            prop_t_ptr = NULL
        else:
            prop_t_ptr = <void*> propensity_t
            if tv_reactions is None:
                for i in range(0, stoich_matrix.shape[0]):
                    tv_react.push_back(i)
            else:
                for i in range(0, len(tv_reactions)):
                    tv_react.push_back(tv_reactions[i])

        cdef _fsp.Model model_ = _fsp.Model(stoich_matrix_arma, call_py_t_fun_obj,
                                            call_py_prop_obj, prop_t_ptr, <void*> propensity_x, tv_react)

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

        cdef arma.Mat[int] X0_arma = arma.Mat[int](<int*> X0.data, X0.shape[1], X0.shape[0], 1, 1)
        cdef arma.Col[_fsp.PetscReal] p0_arma = arma.Col[_fsp.PetscReal](<double*> p0.data, p0.size, 1, 1)
        ierr = self.this_[0].SetInitialDistribution(X0_arma, p0_arma)
        assert (ierr == 0)

    def SetInitialDist2(self, cdd.DiscreteDistribution dist):
        cdef int ierr = 0
        ierr = self.this_[0].SetInitialDistribution(dist.this_[0])
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
            exp_factors.fill(0.2)

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

    def SetOdeTolerances(self, double rel_tol = 1.0e-6, double abs_tol = 1.0e-14):
        """
        Set error tolerances for the ODE integrator.

        Parameters
        ==========

        rel_tol : relative tolerance

        abs_tol : absolute tolerance

        """
        self.this_[0].SetOdeTolerances(rel_tol, abs_tol)

    def SetUp(self):
        """
        Allocate resources for the FSP solver. This method is called implicitly within Solve() and SolveTspan() so normal
        users do not need to bother with it.
        :return:
        :rtype:
        """
        self.this_[0].SetUp()
        self.set_up_ = True

    def Solve(self, double t_final, double fsp_tol = -1.0, double t_init = 0.0, ):
        """
        Solve the CME up to a time point.
        :param t_final: final time.
        :type t_final: float.
        :param fsp_tol: FSP error tolerance. The solver will expand the state set to ensure the final truncation error is within tolerance.
        :type fsp_tol: float.
        If set to negative, the solver will not check the error.
        :param t_init: (optional) Initial time (default is 0).
        :return: solution at t_final.
        :rtype: DiscreteDistribution.
        """
        if self.set_up_ is not True:
            self.SetUp()

        cdef cdd.DiscreteDistribution solution = cdd.DiscreteDistribution()
        try:
            solution.this_[0] = self.this_[0].Solve(t_final, fsp_tol, t_init)
        except RuntimeError:
            print("Runtime error!")
            return None
        return solution

    def SolveTspan(self, tspan, double fsp_tol = -1.0, double t_init = 0.0):
        """
        Solve the chemical master equation over multiple timepoints.

        Parameters
        ----------

        tspan: np.ndarray
            one-dimensional array of timepoints to output the solutions. This array must be sorted in the ascending
            direction.

        fsp_tol: double
            FSP tolerance.

        t_init: double
            Starting time. Default: 0.

        Returns
        -------

        list of DiscreteDistribution. The i-th member corresponds to the approximate solution at time tspan[i].
        """
        if self.set_up_ is not True:
            self.SetUp()
        cdef int ntspan = tspan.size
        snapshots = []
        cdef vector[_dd.DiscreteDistribution] snapshots_c
        try:
            snapshots_c = self.this_[0].SolveTspan(tspan, fsp_tol, t_init)
        except RuntimeError:
            print("Runtime error!")
            return None
        cdef cdd.DiscreteDistribution solution
        for i in range(0, ntspan):
            solution = cdd.DiscreteDistribution()
            solution.this_[0] = snapshots_c[i]
            snapshots.append(solution)
        return snapshots

    def SetOdeSolver(self, solver="CVODE"):
        """
        Set the ODE solver for the truncated CME problem. Currently we support the Krylov integrator for time invariant propensities
        and Sundials' CVODES integrator for time-varying propensities.
        :param solver: a string in {"CVODE","KRYLOV","EPIC"}.
        :type solver: string.
        :return: None.
        :rtype: None.
        """
        solver = solver.lower()
        if solver == 'cvode':
            self.this_[0].SetOdesType(_fsp.CVODE)
        elif solver == 'epic':
            self.this_[0].SetOdesType(_fsp.EPIC)
        elif solver == 'petsc':
            self.this_[0].SetOdesType(_fsp.PETSC)
        else:
            self.this_[0].SetOdesType(_fsp.KRYLOV)

    def SetPetscTSType(self, type='rosw'):
        """
        Set the type of integrator when using PETSC's TS module.
        :param type: (string) name of the type, must be a name recognizable to PETSc.
        :type type:
        :return:
        :rtype:
        """
        type = type.lower()
        cdef string type_str = type.encode('ASCII')
        self.this_[0].SetOdesPetscType(type_str)

    def SetKrylovOrthLength(self, int q=-1):
        """
        SetKrylovOrthLength(self, int q=-1)
        Set the length of orthogonalization (if q=-1 use full orthogonalization).
        :param q:
        :type q:
        :return:
        :rtype:
        """
        self.this_[0].SetKrylovOrthLength(q)

    def SetKrylovDimRange(self, int m_min, int m_max):
        """
        Set the range for the dimension of the adaptive Krylov basis.
        """
        self.this_[0].SetKrylovDimRange(m_min, m_max)

    def ClearState(self):
        self.this_[0].ClearState()
        self.set_up_ = False
