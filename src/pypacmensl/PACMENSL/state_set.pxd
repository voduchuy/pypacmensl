from mpi4py cimport libmpi
cimport arma4cy as arma
cimport mpi4py.MPI as MPI

cdef extern from "PyCallbacksWrapper.h":
    cdef cppclass PyConstrWrapper:
        PyConstrWrapper()
        PyConstrWrapper(object)

cdef extern from "pacmensl.h" namespace "pacmensl":
    ctypedef enum PartitioningType: BLOCK, GRAPH, HYPERGRAPH, HIERARCHICAL

    ctypedef enum PartitioningApproach: FROMSCRATCH, REPARTITION, REFINE

    cdef cppclass StateSetBase:
        StateSetBase(libmpi.MPI_Comm new_comm, int num_species, PartitioningType lb_type,
                     PartitioningApproach lb_approach)

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
        StateSetConstrained(libmpi.MPI_Comm comm, int num_species, PartitioningType part,
                            PartitioningApproach repart) except +

        void SetStoichiometryMatrix(arma.Mat[int] SM) except +

        void SetInitialStates(int num_states, int*vals) except +

        void AddStates(arma.Mat[int] X) except +

        void State2Index(int num_states, int *state, int *indx)

        libmpi.MPI_Comm GetComm()

        int GetNumLocalStates()

        int GetNumGlobalStates()

        int GetNumSpecies()

        int GetNumReactions()

        void CopyStatesOnProc(int num_local_states, int*state_array) const

        void CheckConstraints(int num_states, int *x, int *satisfied)

        int GetNumConstraints()

        void SetShape(int num_constraints, PyConstrWrapper lhs_fun, int *bounds, void *args)

        void SetShapeBounds(int num_constraints, int *bounds)

        void Expand()

