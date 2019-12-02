import mpi4py.MPI as MPI
from pypacmensl.fsp_solver import FspSolverMultiSinks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

k1 = 100.0
ka = 20.0
ket = 6.0
kg = 1.0


def propensity(reaction, X, out):
    if reaction is 0:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 1] ** ket)
        return 0
    if reaction is 1:
        out[:] = kg*X[:, 0]
        return 0
    if reaction is 2:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 2] ** ket)
        return 0
    if reaction is 3:
        out[:] = kg*X[:, 1]
        return 0
    if reaction is 4:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 0] ** ket)
        return 0
    if reaction is 5:
        out[:] = kg*X[:, 2]
        return 0

def rep_constr(X, out):
    n_constr = 6
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]
    out[:, 2] = X[:, 2]
    out[:, 3] = np.multiply(X[:, 0], X[:, 1])
    out[:, 4] = np.multiply(X[:, 2], X[:, 1])
    out[:, 5] = np.multiply(X[:, 0], X[:, 2])


init_bounds = np.array([22, 2, 2, 44, 4, 44])
exp_factors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

rank = MPI.COMM_WORLD.rank

x0 = np.array([[21, 0, 0]])
p0 = np.array([1.0])
sm = np.array([[1, 0, 0],
               [-1, 0, 0],
               [0, 1, 0],
               [0, -1, 0],
               [0, 0, 1],
               [0, 0, -1]])
tspan = np.linspace(0, 1, 2)

# Create FSP solver object
solver = FspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, propensity)
solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)

t1 = MPI.Wtime()
solution_cvode = solver.SolveTspan(tspan, 1.0e-4)
if (rank == 0):
    print("Solve with CVODE takes " + str(MPI.Wtime() - t1))

solver.ClearState()
solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
solver.SetOdeSolver("KRYLOV")
solver.SetVerbosity(2)
t1 = MPI.Wtime()
solution_krylov = solver.SolveTspan(tspan, 1.0e-4)
if (rank == 0):
    print("Solve with Krylov takes " + str(MPI.Wtime() - t1))

if rank == 0:
    fig, axes = plt.subplots(len(tspan),3)

for it in range(0, len(tspan)):
    for ix in range(0,3):
        pmar1 = solution_cvode[it].Marginal(ix)
        pmar2 = solution_krylov[it].Marginal(ix)

        if rank == 0:
            axes[it, ix].plot(pmar1, label='CVODE', color='r')
            axes[it, ix].plot(pmar2, label='Krylov', color='b')

if rank == 0:
    plt.show()