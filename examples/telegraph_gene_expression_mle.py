import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pypacmensl.ssa.ssa import SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot
from pypacmensl.fsp_solver import FspSolverMultiSinks

stoich_mat = np.array([ [-1, 1, 0],
                        [1, -1, 0],
                        [0, 0, 1],
                        [0, 0, -1]])
x0 = np.array([[1, 0, 0]])
p0 = np.array([1.0])
constr_init = np.array([1, 1, 100])

def propensity(reaction, x, out):
    if reaction is 0:
        out[:] = x[:,0]
    if reaction is 1:
        out[:] = x[:,1]
    if reaction is 2:
        out[:] = x[:,1]
    if reaction is 3:
        out[:] = x[:,2]

def t_fun(t, out):
    out[0] = 0.01
    out[1] = 0.001
    out[2] = 1
    out[3] = 0.1

n_t = 4
tspan = np.linspace(0, 100, n_t)

ssa = SSASolver()
ssa.SetModel(stoich_mat, t_fun, propensity)
observations = ssa.Solve(100, x0, 10000)

data = SmFishSnapshot(observations[:,2])

xdat = data.GetStates()
pdat = data.GetFrequencies()/10000

indx = np.argsort(xdat, axis=0)
xdat = xdat[indx[:], 0]
pdat = pdat[indx[:]]



solver = FspSolverMultiSinks(mpi.COMM_WORLD)
solver.SetModel(stoich_mat, t_fun, propensity)
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()
solution = solver.Solve(100.0, 1.0e-4)


marginals = []
for i in range(0, 3):
    marginals.append(solution.Marginal(i))

plt.plot(xdat, pdat)
plt.plot(marginals[2])
plt.show()