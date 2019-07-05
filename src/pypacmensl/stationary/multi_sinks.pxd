from mpi4py.libmpi cimport MPI_Comm

from pypacmensl.libpacmensl cimport _state_set as _ss
from pypacmensl.arma cimport arma4cy as arma
from pypacmensl.libpacmensl cimport Model, PartitioningType
from pypacmensl.petsc.petsc_objects cimport PetscInt, PetscReal
cimport pypacmensl.fsp_solver.distribution as cdd


cdef extern from "pacmensl.h" namespace "pacmensl":
    cdef cppclass StationaryFspSolverMultiSinks:
          StationaryFspSolverMultiSinks(MPI_Comm comm)

          int SetConstraintFunctions(_ss.PyConstrWrapper lhs_constr)
        
          int SetInitialBounds(arma.Row[int] fsp_size)
        
          int SetExpansionFactors(arma.Row[PetscReal] expansion_factors)
        
          int SetModel(Model model)
        
          int SetVerbosity(int verbosity_level)
        
          int SetUp()
        
          int SetInitialDistribution(const arma.Mat[PetscInt] _init_states, const arma.Col[PetscReal] _init_probs)
        
          int SetLoadBalancingMethod(PartitioningType part_type)
        
          cdd.DiscreteDistribution Solve(PetscReal sfsp_tol)
        
          int ClearState()
