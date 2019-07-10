import mpi4py.MPI as mpi
from pypacmensl.stationary import StationaryFspSolverMultiSinks
import numpy as np

comm = mpi.COMM_WORLD

stoich = np.array([[1], [-1]], dtype=np.intc)

def t_fun(t, out):
    out[0] = 20.0
    out[1] = 20.0

def prop(reaction, X, out):
    if reaction == 0:
        out[:] = 1.0
    if reaction == 1:
        out[:] = np.double(X[:,0])
    return


x0 = np.array([[0]])
p0 = np.array([1.0])
init_bounds = np.array([2])

solver = StationaryFspSolverMultiSinks(comm)
solver.SetModel(stoich, t_fun, prop)
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()
solution0 = solver.Solve(1.0e-10)
solver.ClearState()

p = solution0.Marginal(0)
print(p)

