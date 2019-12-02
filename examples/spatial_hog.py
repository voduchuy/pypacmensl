import mpi4py.MPI as MPI
import pypacmensl.fsp_solver.multi_sinks as pacmensl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import exp

Ahog = 9.3E09
num_genes = 1
Mhog = 2.2E-2
etahog = 5.9
r1 = 6.1E-3
r2 = 6.9E-3


def hog1p(t):
    """Hog1p signal in arbitrary unit (AU)"""
    return (1.0 - exp(-r1 * t)) * exp(-r2 * t)


def hog1pstar(t):
    """Saturated hog1p signal"""
    return ((hog1p(t) / (1.0 + hog1p(t) / Mhog)) ** etahog) * Ahog


t0 = 2.6E-1
k01 = 2.6E-3
k10a = 1.0E1
k10b = 3.2E4
k12 = 7.6E-3
k21 = 1.2E-2
k23 = 4E-3
k32 = 3.1E-3
kr0 = 5.9E-4
kr1 = 1.7E-1
kr2 = 1.0
kr3 = 3.2E-2
gamma_nuc = 2.2E-6
k_transport = 2.6E-1
gamma_cyt = 19.3E-3

x0 = np.array([[num_genes, 0, 0, 0, 0, 0]])
p0 = np.array([1.0])
sm = np.array([[-1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, 0, 0],
               [0, -1, 1, 0, 0, 0],
               [0, 1, -1, 0, 0, 0],
               [0, 0, -1, 1, 0, 0],
               [0, 0, 1, -1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, -1, 0],
               [0, 0, 0, 0, -1, 1],
               [0, 0, 0, 0, 0, -1]
               ])


# // TODO: automatic
# generation
# of
# mass
# action
# propensity


def propensity(reaction, X, out):
    if reaction > 12:
        raise RuntimeError("The model has only 13 reactions!")
    if reaction is 0:
        out[:] = k01*X[:, 0]
        return
    if reaction is 1:
        out[:] = X[:, 1]
        return
    if reaction is 2:
        out[:] = k12*X[:, 1]
        return
    if reaction is 3:
        out[:] = k21*X[:, 2]
        return
    if reaction is 4:
        out[:] = k23*X[:, 2]
        return
    if reaction is 5:
        out[:] = k32*X[:, 3]
        return
    if reaction is 6:
        out[:] = kr0*X[:, 0]
        return
    if reaction is 7:
        out[:] = kr1*X[:, 1]
        return
    if reaction is 8:
        out[:] = kr2*X[:, 2]
        return
    if reaction is 9:
        out[:] = kr3*X[:, 3]
        return
    if reaction is 10:
        out[:] = gamma_nuc*X[:, 4]
        return
    if reaction is 11:
        out[:] = k_transport*X[:, 4]
        return
    if reaction is 12:
        out[:] = gamma_cyt*X[:, 5]
        return


def t_fun(time, out):
    if time > t0:
        out[1] = np.double(max(0.0, k10a - k10b * hog1pstar(time - t0)))
    else:
        out[1] = 0.0


init_bounds = np.array([1, 1, 1, 1, 50, 100])

# Create parallel solver object
solver = pacmensl.FspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, propensity, t_fun, [1])
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetOdeSolver("petsc")
solver.SetUp()

tspan = np.linspace(0, 60 * 15, 5)
solution1 = solver.SolveTspan(tspan, 1.0e-4)
solver.ClearState()


# Create parallel solver object
solver = pacmensl.FspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, propensity, t_fun, [1])
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetOdeSolver("cvode")
solver.SetUp()

tspan = np.linspace(0, 60 * 15, 5)
solution2 = solver.SolveTspan(tspan, 1.0e-4)
solver.ClearState()

for i in range(len(tspan)):
    prna1 = solution1[i].Marginal(4)
    prna2 = solution2[i].Marginal(4)
    print(np.linalg.norm(prna1 - prna2, ord=2))