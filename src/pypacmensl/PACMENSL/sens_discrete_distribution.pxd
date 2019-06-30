cimport arma4cy as arma
from libcpp.vector cimport *
from mpi4py.libmpi cimport *
from state_set cimport *

from sens_discrete_distribution cimport *

cdef extern from "petsc.h":
    ctypedef double PetscReal
    ctypedef Vec

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass SensDiscreteDistribution:
        vector[Vec] dp_

        SensDiscreteDistribution()

        SensDiscreteDistribution(MPI_Comm comm, double t, StateSetBase *state_set, Vec &p, vector[Vec] &dp)

        SensDiscreteDistribution(const SensDiscreteDistribution &dist)

        SensDiscreteDistribution(SensDiscreteDistribution &&dist)

        SensDiscreteDistribution &operator=(const SensDiscreteDistribution &dist)

        SensDiscreteDistribution &operator=(SensDiscreteDistribution &&dist)

        int GetSensView(int , int &num_states, double* &p)

        int RestoreSensView(int , double* &p)

    int Compute1DSensMarginal(SensDiscreteDistribution &dist,
                                        int,
                                        int,
                                        arma.Col[PetscReal] &out)

    int ComputeFIM(SensDiscreteDistribution &dist, arma.Mat[PetscReal] fim)