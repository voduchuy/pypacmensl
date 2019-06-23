cimport arma4cy as arma
from state_set cimport *
from libcpp.vector cimport vector
from discrete_distribution cimport *
from mpi4py.libmpi cimport *

cdef extern from "PyCallbacksWrapper.h":
    cdef cppclass PyPropWrapper:
        PyPropWrapper()
        PyPropWrapper(object)

    cdef cppclass PyTFunWrapper:
        PyTFunWrapper()
        PyTFunWrapper(object)

cdef extern from "petsc.h":
    ctypedef double PetscReal
    ctypedef enum PetscBool:
        PETSC_FALSE
        PETSC_TRUE

cdef extern from "pacmensl.h" namespace "pacmensl":
    ctypedef enum ODESolverType: KRYLOV, CVODE_BDF

    cdef cppclass Model:
        arma.Mat[int] stoichiometry_matrix_
        PyTFunWrapper t_fun_
        void* t_fun_args_
        PyPropWrapper prop_
        void* prop_args_

        Model();

        Model(arma.Mat[int] stoichiometry_matrix, PyTFunWrapper t_fun, void* t_fun_args, PyPropWrapper prop, void* prop_fun_args) except +

        Model(const Model & model)

    cdef cppclass FspSolverMultiSinks:
        FspSolverMultiSinks( MPI_Comm _comm)
        FspSolverMultiSinks( MPI_Comm _comm, PartitioningType _part_type)
        FspSolverMultiSinks( MPI_Comm _comm, PartitioningType _part_type,
                                      ODESolverType _solve_type)

        int SetConstraintFunctions( PyConstrWrapper &lhs_constr )

        int SetInitialBounds( arma.Row[int] &_fsp_size )

        int SetExpansionFactors( arma.Row[ PetscReal ] &_expansion_factors )

        int SetModel( Model &model )

        int SetVerbosity( int verbosity_level )

        int SetInitialDistribution( arma.Mat[int] &_init_states, arma.Col[PetscReal] &_init_probs )

        int SetLogging( PetscBool logging )

        int SetFromOptions( )

        int SetLoadBalancingMethod( PartitioningType part_type )

        int SetOdesType( ODESolverType odes_type )

        int SetUp( )

        const StateSetBase *GetStateSet( )

        DiscreteDistribution Solve( PetscReal t_final, PetscReal fsp_tol )

        vector[DiscreteDistribution] SolveTspan( vector[PetscReal] &tspan, PetscReal fsp_tol )

        int DestroySolverState( )
