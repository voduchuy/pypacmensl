from pypacmensl.fsp_solver import FspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import exp

stoich_mat = np.array([ [-1, 1, 0, 0, 0, 0],
                        [1, -1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, -1, 0, 0, 0],
                        [0, 0, 0, -1, 1, 0],
                        [0, 0, 0, 1, -1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, -1]
                        ])
x0 = np.array([[1, 0, 100, 1, 0, 10]])
p0 = np.array([1.0])
constr_init = np.array([1, 1, 10, 1, 1, 10])

def propensity(reaction, x, out):
    if reaction is 0:
        out[:] = x[:,0]
    if reaction is 1:
        out[:] = x[:,1]
    if reaction is 2:
        out[:] = x[:,1]
    if reaction is 3:
        out[:] = x[:,2]
    if reaction is 4:
        out[:] = x[:,3]
    if reaction is 5:
        out[:] = x[:,4]
    if reaction is 6:
        out[:] = x[:,4]
    if reaction is 7:
        out[:] = x[:,5]

def t_fun(t, out):
    signal = exp(-0.01*t)*(1 - exp(-0.005*t))
    out[0] = 0.01
    out[1] = 0.1
    out[2] = 10
    out[3] = 0.1
    out[4] = 0.01
    out[5] = 0.1
    out[6] = 1
    out[7] = 0.1

n_t = 4
tspan = np.linspace(0, 10, n_t)

solver = FspSolverMultiSinks(mpi.COMM_SELF)
solver.SetModel(stoich_mat, t_fun, propensity)
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()
solutions = solver.SolveTspan(tspan, 1.0e-4)

X = np.asarray(solutions[n_t - 1].GetStatesViewer())
P = np.asarray(solutions[n_t - 1].GetProbViewer())
n_state = P.shape[0]
print(X.shape)
print(n_state)
# Compute E(XY)
exy = np.sum(X[:, 2]*X[:, 5]*P[:])
print(exy)

ex = np.sum(X[:, 2]*P[:])
ey = np.sum(X[:, 5]*P[:])
print(ex*ey)

sigmax = np.sqrt(np.sum(X[:,2]*X[:,2]*P[:]) - ex*ex)
sigmay = np.sqrt(np.sum(X[:,5]*X[:,5]*P[:]) - ey*ey)

corr = (exy - ex*ey)/(sigmax*sigmay)
print(corr)