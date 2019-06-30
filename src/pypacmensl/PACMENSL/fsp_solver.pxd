cimport arma4cy as arma
from pacmensl_callbacks cimport *
from state_set cimport *
from libcpp.vector cimport vector
from discrete_distribution cimport *
from mpi4py.libmpi cimport *

cdef extern from "petsc.h":
    ctypedef double PetscReal
    ctypedef enum PetscBool:
        PETSC_FALSE
        PETSC_TRUE

cdef extern from "pacmensl.h" namespace "pacmensl":
    ctypedef enum ODESolverType: KRYLOV, CVODE_BDF

    cdef cppclass Model:
        arma.Mat[int] stoichiometry_matrix_
        PyTFunWrapper prop_t_
        PyPropWrapper prop_x_
        void* prop_t_args_
        void* prop_x_args_

        Model(arma.Mat[int] stoichiometry_matrix, PyTFunWrapper, PyPropWrapper) except +

        Model(arma.Mat[int] stoichiometry_matrix, PyTFunWrapper, PyPropWrapper, void*, void*) except +

        Model();

        Model(const Model & model)

        Model& operator=(Model &model)
        Model& operator=(Model &&model)

    cdef cppclass FspSolverMultiSinks:
        FspSolverMultiSinks( MPI_Comm _comm)

        int SetConstraintFunctions( PyConstrWrapper &lhs_constr ) except +

        int SetInitialBounds( arma.Row[int] &_fsp_size ) except +

        int SetExpansionFactors( arma.Row[ PetscReal ] &_expansion_factors ) except +

        int SetModel( Model &model ) except +

        int SetVerbosity( int verbosity_level ) except +

        int SetInitialDistribution( arma.Mat[int] &_init_states, arma.Col[PetscReal] &_init_probs ) except +

        int SetLogging( PetscBool logging ) except +

        int SetFromOptions( ) except +

        int SetLoadBalancingMethod( PartitioningType part_type ) except +

        int SetOdesType( ODESolverType odes_type ) except +

        int SetUp( ) except +

        const StateSetBase *GetStateSet( ) except +

        DiscreteDistribution Solve( PetscReal t_final, PetscReal fsp_tol ) except +

        vector[DiscreteDistribution] SolveTspan( vector[PetscReal] &tspan, PetscReal fsp_tol ) except +

        int ClearState( ) except +
