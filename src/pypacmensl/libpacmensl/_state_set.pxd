from mpi4py cimport libmpi
cimport pypacmensl.arma.arma4cy as arma

from pypacmensl.libpacmensl._callbacks cimport *

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass PartitioningType:
        pass

    cdef cppclass PartitioningApproach:
        pass

cdef extern from "pacmensl.h" namespace "pacmensl::PartitioningType":
    cdef PartitioningType GRAPH
    cdef PartitioningType HYPERGRAPH
    cdef PartitioningType BLOCK

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass StateSetBase:
        StateSetBase(libmpi.MPI_Comm new_comm)

        void SetStoichiometryMatrix(int num_species, int num_reactions, const int*values)

        void SetInitialStates(int num_states, int*vals) except +

        void AddStates(arma.Mat[int] X) except +

        void State2Index(int num_states, int *state, int *indx)

        libmpi.MPI_Comm GetComm()

        int GetNumLocalStates()

        int GetNumGlobalStates()

        int GetNumSpecies()

        int GetNumReactions()

        void CopyStatesOnProc(int num_local_states, int*state_array) const

    ctypedef fsp_constr_multi_fn

    cdef cppclass StateSetConstrained:
        StateSetConstrained(libmpi.MPI_Comm comm) except +

        int SetStoichiometryMatrix(arma.Mat[int] SM) except +

        int AddStates(arma.Mat[int] X) except +

        void State2Index(int num_states, int *state, int *indx)

        libmpi.MPI_Comm GetComm()

        int GetNumLocalStates()

        int GetNumGlobalStates()

        int GetNumSpecies()

        int GetNumReactions()

        void CopyStatesOnProc(int num_local_states, int*state_array) const

        int CheckConstraints(int num_states, int *x, int *satisfied)

        int GetNumConstraints()

        int SetLoadBalancingScheme(PartitioningType type)

        int SetShape(int num_constraints, ConstrFun lhs_fun, int *bounds, void *args)

        int SetShapeBounds(int num_constraints, int *bounds)

        int SetUp()

        int Expand()

        int Clear()

