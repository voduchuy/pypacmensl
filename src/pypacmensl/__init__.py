# distutils : language = c++

# this must be done first because my mpi_init code is fragile on some system
import mpi4py.MPI as MPI

import pypacmensl.utils.environment as environment
from pypacmensl.callbacks import _pacmensl_callbacks

__all__ = ["fsp_solver", "sensitivity", "stationary", "smfish", "state_set"]

my_env = environment._Environment()