cimport pypacmensl.arma.arma4cy as arma
from pypacmensl.libpacmensl._discrete_distribution cimport DiscreteDistribution
from pypacmensl.libpacmensl._sens_discrete_distribution cimport SensDiscreteDistribution
from pypacmensl.petsc.petsc_objects cimport PetscReal
from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass SmFishSnapshot:
        SmFishSnapshot()
        SmFishSnapshot(arma.Mat[int] & observations)
        SmFishSnapshot& operator=(SmFishSnapshot& & src)
        int GetObservationIndex(arma.Col[int] & x)
        int GetNumObservations()
        arma.Mat[int] & GetObservations()
        arma.Row[int] & GetFrequencies()
        void Clear()

    double SmFishSnapshotLogLikelihood(SmFishSnapshot & data,
                                       const DiscreteDistribution & distribution,
                                       arma.Col[int] measured_species,
                                       bool use_base_2)

    int SmFishSnapshotGradient(SmFishSnapshot & data,
                               SensDiscreteDistribution & distribution,
                               vector[PetscReal]& gradient,
                               arma.Col[int] measured_species,
                               bool use_base_2)
