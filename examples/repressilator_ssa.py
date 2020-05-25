import mpi4py.MPI as MPI
import numpy as np
import matplotlib.pyplot as plt
from pypacmensl.ssa.ssa import SSASolver
from matplotlib.ticker import FormatStrFormatter

rank = MPI.COMM_WORLD.rank


k1 = 10.0
ka = 20.0
ket = 6.0
kg = 1.0


def propensity(reaction, X, out):
    if reaction == 0:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 1] ** ket)
        return 0
    if reaction == 1:
        out[:] = kg*X[:, 0]
        return 0
    if reaction == 2:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 2] ** ket)
        return 0
    if reaction == 3:
        out[:] = kg*X[:, 1]
        return 0
    if reaction == 4:
        out[:] = k1*np.reciprocal(1.0 + ka * X[:, 0] ** ket)
        return 0
    if reaction == 5:
        out[:] = kg*X[:, 2]
        return 0

def tfun(t, out):
    out[:] = 1.0

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



x0 = np.array([[20, 0, 0]])
p0 = np.array([1.0])
sm = np.array([[1, 0, 0],
               [-1, 0, 0],
               [0, 1, 0],
               [0, -1, 0],
               [0, 0, 1],
               [0, 0, -1]])
t_final = 20

ssa = SSASolver(MPI.COMM_WORLD)
ssa.SetModel(sm, tfun, propensity)
X = ssa.Solve(t_final, x0, 1000, True)

if MPI.COMM_WORLD.Get_rank() == 0:
    plt.scatter(X[:,0], X[:, 1])
    plt.show()


