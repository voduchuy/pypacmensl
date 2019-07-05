import pypacmensl.PACMENSL as pacmensl
import numpy as np
import mpi4py.MPI as mpi



class tnfa_moodel:
    def __init__(self, comm):
        self.parameters_ = np.empty((9), dtype=np.double)
        self.solver_ = pacmensl.FspSolverMultiSinks(comm)
