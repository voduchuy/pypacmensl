import mpi4py.MPI as MPI
from pypacmensl.stationary import StationaryFspSolverMultiSinks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

k1 = 100.0
ka = 20.0
ket = 6.0
kg = 1.0

def propensity(reaction, X, out):
    """
    Propensity function for the repressilator.

    Arguments
    =========

    reaction:   int
        Reaction index.

    X:  2-D array
        Array of CME states to evaluate propensity function at. Each row is a CME state.

    out:    1-D array, output
        Array to be filled with propensity function evaluations.
    """
    if reaction == 0:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 1] ** ket)
    if reaction == 1:
        out[:] = kg*X[:, 0]
    if reaction == 2:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 2] ** ket)
    if reaction == 3:
        out[:] = kg*X[:, 1]
    if reaction == 4:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 0] ** ket)
    if reaction == 5:
        out[:] = kg*X[:, 2]
    return 0


def rep_constr(X, out):
    """
    Constraint function for the FSP.

    Arguments
    =========

    X:  2-D array, input
        Array of CME state vectors. Each row is a state.

    out: 2-D array, output
        Array of output for the constraint functions. Each column corresponds to a constraint, and each row a state.
    """
    n_constr = 6
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]
    out[:, 2] = X[:, 2]
    out[:, 3] = np.multiply(X[:, 0], X[:, 1])
    out[:, 4] = np.multiply(X[:, 2], X[:, 1])
    out[:, 5] = np.multiply(X[:, 0], X[:, 2])

init_bounds = np.array([10, 10, 10, 44, 4, 44])
exp_factors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

rank = MPI.COMM_WORLD.rank

x0 = np.array([[0, 0, 1]])
p0 = np.array([1.0])
sm = np.array([[1, 0, 0],
               [-1, 0, 0],
               [0, 1, 0],
               [0, -1, 0],
               [0, 0, 1],
               [0, 0, -1]])

# Create FSP solver object
solver = StationaryFspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, None, propensity)
solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solution = solver.Solve(1e-4)


if rank == 0:
    fig, axes = plt.subplots(1,3)

for ix in range(0,3):
        p_marginal = solution.Marginal(ix)

        if rank == 0:
            axes[ix].bar(range(0, len(p_marginal)), p_marginal,  color='r')

if rank == 0:
    plt.show()