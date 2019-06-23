cimport mpi4py.libmpi as mpi
cimport state_set
cimport arma4cy as arma

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass DiscreteDistribution:
        DiscreteDistribution()

        DiscreteDistribution(const
        DiscreteDistribution & dist)

        DiscreteDistribution(DiscreteDistribution &&dist)

        DiscreteDistribution & operator = (const DiscreteDistribution &)
        DiscreteDistribution & operator = (const DiscreteDistribution &&)

        void GetStateView( int &num_states, int &num_species, int *&states )
        void GetProbView(int &num_states, double *&p)
        void RestoreProbView( double *&p )

    cdef arma.Col[double] Compute1DMarginal(const DiscreteDistribution dist, int species)