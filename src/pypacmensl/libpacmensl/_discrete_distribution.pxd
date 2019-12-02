cimport pypacmensl.arma.arma4cy as arma
from libpacmensl._callbacks cimport WeightFun

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass DiscreteDistribution:
        DiscreteDistribution()

        DiscreteDistribution(const
        DiscreteDistribution & dist)

        DiscreteDistribution(DiscreteDistribution &&dist)

        DiscreteDistribution & operator = (const DiscreteDistribution &)
        DiscreteDistribution & operator = (const DiscreteDistribution &&)

        void GetStateView( int num_states, int &num_species, int *states )
        void GetProbView(int num_states, double *p)
        void RestoreProbView( double *p )
        int WeightedAverage(int nout, double* fout, WeightFun w, void *wf_args)

    cdef arma.Col[double] Compute1DMarginal(const DiscreteDistribution dist, int species)
