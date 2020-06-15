from libcpp.vector cimport *
from libcpp.string cimport *
from mpi4py.libmpi cimport MPI_Comm

from pypacmensl.arma cimport arma4cy as arma
from pypacmensl.libpacmensl._callbacks cimport *
from pypacmensl.petsc.petsc_objects cimport PetscBool, PetscReal, PetscInt
from pypacmensl.libpacmensl._discrete_distribution cimport DiscreteDistribution
from pypacmensl.libpacmensl._sens_discrete_distribution cimport SensDiscreteDistribution
from pypacmensl.libpacmensl._state_set cimport PartitioningType, GRAPH, HYPERGRAPH, BLOCK, StateSetBase

ctypedef void* void_ptr

cdef extern from "pacmensl.h" namespace "pacmensl":
    ctypedef enum ODESolverType:
        KRYLOV,
        CVODE,
        PETSC,
        EPIC

    cdef cppclass Model:
        arma.Mat[int] stoichiometry_matrix_
        PropTFun prop_t_
        PropXFun prop_x_
        void* prop_t_args_
        void* prop_x_args_
        vector[int] tv_reactions_

        Model(arma.Mat[int] stoichiometry_matrix, PropTFun, PropXFun) except +

        Model(arma.Mat[int] stoichiometry_matrix, PropTFun, PropXFun, void*, void*, vector[int]) except +

        Model();

        Model(const Model & model)

        Model& operator=(Model &model)
        Model& operator=(Model &&model)

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass FspSolverMultiSinks:
        FspSolverMultiSinks( MPI_Comm _comm)

        int SetConstraintFunctions( ConstrFun lhs_constr ) except +

        int SetConstraintFunctions( ConstrFun lhs_constr, void* constr_args ) except +

        int SetInitialBounds( arma.Row[int] _fsp_size ) except +

        int SetExpansionFactors( arma.Row[ PetscReal ] _expansion_factors ) except +

        int SetModel( Model model ) except +

        int SetVerbosity( int verbosity_level ) except +

        int SetInitialDistribution( arma.Mat[int] _init_states, arma.Col[PetscReal] _init_probs ) except +

        int SetInitialDistribution(DiscreteDistribution init_dist) except +

        int SetLogging( PetscBool logging ) except +

        int SetFromOptions( ) except +

        int SetLoadBalancingMethod( PartitioningType part_type ) except +

        int SetOdesType( ODESolverType odes_type ) except +

        int SetOdesPetscType(string ts_type) except +

        int SetKrylovOrthLength(int q) except +

        int SetKrylovDimRange(int m_min, int m_max) except +

        int SetOdeTolerances(PetscReal rel_tol, PetscReal abs_tol) except +

        int SetUp( ) except +

        const StateSetBase *GetStateSet( ) except +

        DiscreteDistribution Solve( PetscReal t_final,  PetscReal fsp_tol, PetscReal t_init ) except +

        vector[DiscreteDistribution] SolveTspan( vector[PetscReal] tspan, PetscReal fsp_tol, PetscReal t_init ) except +

        int ClearState( ) except +

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass SensModel(Model):
        int                   num_reactions_
        int                   num_parameters_
        arma.Mat[int]         stoichiometry_matrix_
        vector[int] tv_reactions_
        PropTFun         prop_t_
        PropXFun         prop_x_
        vector[PropXFun]  dprop_x_
        vector[PropTFun] dprop_t_
        void                  *prop_t_args_
        void                  *prop_x_args_
        vector[void_ptr]   dprop_x_args_
        vector[void_ptr]   dprop_t_args_
        vector[int] dpropensity_ic_
        vector[int] dpropensity_rowptr_

        SensModel()

        SensModel(arma.Mat[int] stoichiometry_matrix,
                  vector[int] tv_reactions,
                  PropTFun & prop_t,
                  PropXFun & prop_x,
                  vector[PropTFun] & dprop_t,
                  vector[PropXFun] & dprop_x)

        SensModel(arma.Mat[int] stoichiometry_matrix,
                  vector[int] tv_reactions,
                  PropTFun & prop_t,
                  PropXFun & prop_x,
                  vector[PropTFun] & dprop_t,
                  vector[PropXFun] & dprop_x,
                  vector[int]& dprop_ic ,
                  vector[int]& dprop_rowptr)

        # SensModel(SensModel & model)

        SensModel & operator=(SensModel & model)

        SensModel & operator=(SensModel & & model)

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass SensFspSolverMultiSinks:
          SensFspSolverMultiSinks(MPI_Comm _comm)
          int SetConstraintFunctions(ConstrFun lhs_constr, void* args)
          int SetInitialBounds(arma.Row[int] &_fsp_size)
          int SetExpansionFactors(arma.Row[PetscReal] &_expansion_factors)
          int SetModel(SensModel &model)
          int SetVerbosity(int verbosity_level)
          int SetInitialDistribution(arma.Mat[int] &_init_states,
                                                   arma.Col[PetscReal] &_init_probs,
                                                   vector[arma.Col[PetscReal]] &_init_sens)
          int SetLoadBalancingMethod(PartitioningType part_type)
          int SetUp()
          const StateSetBase *GetStateSet()
          SensDiscreteDistribution Solve(PetscReal t_final, PetscReal fsp_tol)
          vector[SensDiscreteDistribution] SolveTspan(vector[PetscReal] &tspan, PetscReal fsp_tol)
          int ClearState()

cdef extern from 'pacmensl.h' namespace 'pacmensl':
    cdef cppclass StationaryFspSolverMultiSinks:
        StationaryFspSolverMultiSinks(MPI_Comm comm)

        int SetConstraintFunctions(ConstrFun lhs_constr, void* args)

        int SetInitialBounds(arma.Row[int] &_fsp_size)

        int SetExpansionFactors(arma.Row[PetscReal] &_expansion_factors)

        int SetModel(Model &model)

        int SetVerbosity(int verbosity_level)

        int SetUp()

        int SetInitialDistribution(const arma.Mat[int] &_init_states, const arma.Col[double] &_init_probs)

        int SetLoadBalancingMethod(PartitioningType part_type)

        DiscreteDistribution Solve(PetscReal sfsp_tol)

        int ClearState()
