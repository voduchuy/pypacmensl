from libcpp.vector cimport vector
from mpi4py.libmpi cimport MPI_Comm
from pypacmensl.libpacmensl._callbacks cimport WeightFun
from pypacmensl.libpacmensl._discrete_distribution cimport DiscreteDistribution
from pypacmensl.libpacmensl._state_set cimport StateSetBase
cimport pypacmensl.arma.arma4cy as arma

cdef extern from "petsc.h":
    ctypedef double PetscReal
    cdef struct Vec

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass SensDiscreteDistribution(DiscreteDistribution):
        vector[Vec] dp_

        SensDiscreteDistribution()

        SensDiscreteDistribution(MPI_Comm comm, double t, StateSetBase *state_set, Vec & p, vector[Vec] & dp)

        SensDiscreteDistribution(const SensDiscreteDistribution & dist)

        SensDiscreteDistribution(SensDiscreteDistribution & & dist)

        SensDiscreteDistribution & operator=(const SensDiscreteDistribution & dist)

        SensDiscreteDistribution & operator=(SensDiscreteDistribution & & dist)

        # void GetStateView(int & num_states, int & num_species, int *& states)
        #
        # void GetProbView(int & num_states, double *& p)
        #
        # void RestoreProbView(PetscReal* p)

        int GetSensView(int, int num_states, double*p)

        int RestoreSensView(int, PetscReal*p)

        int WeightedAverage(int iS, int nout,
                            PetscReal *fout,
                            WeightFun w,
                            void *wf_args)

    arma.Col[PetscReal] Compute1DMarginal(const SensDiscreteDistribution dist, int species)

    int Compute1DSensMarginal(SensDiscreteDistribution& dist,
                              int,
                              int,
                              arma.Col[PetscReal]& out)

    int ComputeFIM(SensDiscreteDistribution& dist, arma.Mat[PetscReal] fim)
