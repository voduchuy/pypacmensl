# distutils: language = c++
from typing import List
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

    def SetModel(self,
                 int num_parameters,
                 cnp.ndarray stoich_matrix,
                 propensity_t,
                 propensity_x,
                 tv_reactions,
                 d_propensity_t=None,
                 d_propensity_t_sp=None,
                 d_propensity_x=None,
                 d_propensity_x_sp=None
                 ):
        '''
        SetModel(num_parameters, stoich_matrix, propensity_t, propensity_x, tv_reactions, d_propensity_t, d_propensity_t_sp, d_propenxity_x, d_propensity_x_sp)

        Set the information of the stochastic reaction network model to be solved by the forward sensitivity FSP.

        Parameters
        ----------

        num_parameters : int
            Number of model parameters.

        stoich_matrix : 2-d array
             Stoichiometry matrix. Reactions are arranged row-wise.

        propensity_t : Callable[[float, np.ndarray], None]
            Function to evaluate the time-varying coefficients of the propensities. This could be set to None if all
            reaction propensities are time-independent. The callable should fill only the entries corresponding to time-varying propensities
            in the output array.

        propensity_x : Callable[[int, np.ndarray, np.ndarray], None]
            Function to evaluate the state-dependent factors of the propensities. The first argument is the reaction index,
            the second argument is the input array of states (arranged row-wise), the third argument is the 1-d output array.

        tv_reactions : List[int]
            List of time-varying reactions. If empty, we assume all reactions are time-independent.

        d_propensity_t : Callable[[int, float, np.ndarray], None]
            Function to evaluate the derivatives of the time-varying propensity coefficients with respect to a specified parameter.
            The first argument is the parameter index, the second parameter is time, the third argument is the output 1-d array with
            the same length as the number of reactions. The callable should fill only the entries corresponding to time-varying propensities
            in the output array.

        d_propensity_t_sp : List[List[int]]
            Sparsity pattern for the d_propensity_t. Must be a list of array-like objects with `len(d_propensity_t_sp) == num_parameters`.
            For example, `d_propensity_t_sp[0] = [0, 1, 3]` means that the time-varying factors in the propensities of reactions 0, 1, and 3 have
            non-zero partial derivatives with respect to the 0th parameter.

        d_propensity_x : Callable[[int, int, np.ndarray, np.ndarray], None]
            Function to evaluate the derivatives of the time-independent propensity factors with respect to a specified parameter. The
            first argument is the parameter index, the second the reaction index, the third a 2-d array of input states arranged row-wise,
            and the final argument is the output 1-d array.

        d_propensity_x_sp : List[List[int]]
            Sparsity pattern for the partial derivatives of the time-independent propensity factors. Must be a list of
            array-like objects with `len(d_propensity_x_sp) == num_parameters`.
            For example, `d_propensity_x_sp[0] = [0, 1, 3]` means that the time-invariant factors in the propensities of reactions 0, 1, and 3 have
            non-zero partial derivatives with respect to the 0th parameter.

        Returns
        -------
        None
        '''
        cdef int ierr

        # convert stoich_matrix to armadillo array
        if stoich_matrix.dtype != np.intc:
            stoich_matrix = stoich_matrix.astype(np.intc)
        if not stoich_matrix.flags['C_CONTIGUOUS']:
            stoich_matrix = np.ascontiguousarray(stoich_matrix)

        cdef arma.Mat[int] stoich_matrix_arma = arma.Mat[int](<int*> stoich_matrix.data, stoich_matrix.shape[1],
                                                              stoich_matrix.shape[0], 0, 1)

        cdef _fsp.SensModel model_

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

        model_.num_parameters_ = num_parameters
        model_.stoichiometry_matrix_ = stoich_matrix_arma
        model_.tv_reactions_ = tv_react
        model_.prop_t_ = call_py_propt_obj
        model_.prop_x_ = call_py_propx_obj
        model_.dprop_t_ = call_py_dpropt_obj
        model_.dprop_x_ = call_py_dpropx_obj
        model_.prop_t_args_ = prop_t_ptr
        model_.prop_x_args_ = <void*> propensity_x

        model_.dprop_t_args_ = <void*> d_propensity_t if d_propensity_t is not None else NULL
        model_.dprop_x_args_ = <void*> d_propensity_x if d_propensity_x is not None else NULL

        model_.dprop_t_sp_.reserve(num_parameters)
        model_.dprop_x_sp_.reserve(num_parameters)

        if d_propensity_t_sp is not None:
            for i in range(num_parameters):
                model_.dprop_t_sp_.push_back(vector[int]())
                for r in d_propensity_t_sp[i]:
                    model_.dprop_t_sp_[i].push_back(r)
        if d_propensity_x_sp is not None:
            for i in range(num_parameters):
                model_.dprop_x_sp_.push_back(vector[int]())
                for r in d_propensity_x_sp[i]:
                    model_.dprop_x_sp_[i].push_back(r)

        ierr = self.this_[0].SetModel(model_)

        assert (ierr == 0)

    def SetInitialDist(self, cnp.ndarray X0, cnp.ndarray p0, s0):
        """
        SetInitialDist(x0, p0, s0)

        Set initial distribution. This must be called before .Solve() or .SolveTspan().

        Parameters
        ----------

        x0: 2-D numpy array
            Initial states. Each row is a state.

        p0: 1-D numpy array
            Initial probabilities. Each entry maps to a row in x0.

        s0: List of 1-D numpy arrays
            Initial sensitivities. Each vector in this list must have the same layout as p0.

        Returns
        -------
        None

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

    def SetInitialDist1(self, sdd.SensDiscreteDistribution s0):
        """
        SetInitialDist1()

        Set initial condition for the forward sensitivity system.

        Args:
            s0 (SensDiscretDistribution): Initial probability and sensitivities

        Returns:
            None

        """
        cdef int ierr = 0
        ierr = self.this_[0].SetInitialDistribution(s0.this_[0])
        assert (ierr == 0)

    def SetFspShape(self, constr_fun, cnp.ndarray constr_bound, cnp.ndarray exp_factors = None):
        """
        SetFspShape(constr_fun, constr_bound, exp_factors)

        Parameters
        ----------
        constr_fun : Callable
            Constraint function. Callable in the form `def constr_fun(states, out)` which, upon returning, populate out[i, j] with the value of the j-th constraint function evaluated at states[i,:].

        constr_bound : 1-D array
            Array of bounds.

        exp_factors : 1-D array
            Array of expansion factors.

        Returns
        -------
            None
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
        """
        SetVerbosity(level)

        Set the level of outputs to stdout.

        Parameters
        ----------
        level : int
            Output level. 0: no outputs. 1: outputs from the outer FSP loop (when e.g. expanding the state space). 2: like 1 but also include outputs from the inner ODE time-stepper.

        Returns
        -------
        None
        """
        self.this_[0].SetVerbosity(level)

    def SetLBMethod(self, method="Graph"):
        """
        SetLBMethod(method)

        Parameters
        ----------
        method : str, default="Graph"

        Returns
        -------
        None
        """
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

        Returns
        -------
        None

        """
        self.this_[0].SetUp()
        self.set_up_ = True

    def Solve(self, double t_final, double fsp_tol):
        """
        Solve(t_final, fsp_tol)

        Integrate the forward sensitivity CME and output the solution at the end time.

        Parameters
        ----------
        t_final : double
            Final time.

        fsp_tol : double
            FSP tolerance.

        Returns
        -------
        Final solution, an instance of SensDiscreteDistribution.

        """
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
        """
        SolveTspan(tspan, fsp_tol)

        Integrate the forward sensitivity CME and output solutions at intermediate times.

        Parameters
        ----------
        tspan : 1-D array
            Output times.

        fsp_tol : double
            Tolerance of the FSP.

        Returns
        -------
        List of intermediate solutions. Each element is an instance of the SensDiscreteDistribution class.

        """
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
        """
        Clear internal data structures.

        Returns
        -------
        None
        """
        self.this_[0].ClearState()
        self.set_up_ = False
