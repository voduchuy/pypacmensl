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
        out[:] = np.reciprocal(1.0 + ka * X[:, 1] ** ket)
        return 0
    if reaction is 1:
        out[:] = X[:, 0]
        return 0
    if reaction is 2:
        out[:] = np.reciprocal(1.0 + ka * X[:, 2] ** ket)
        return 0
    if reaction is 3:
        out[:] = X[:, 1]
        return 0
    if reaction is 4:
        out[:] = np.reciprocal(1.0 + ka * X[:, 0] ** ket)
        return 0
    if reaction is 5:
        out[:] = X[:, 2]
        return 0


def t_fun(time, out):
    out[0] = np.double(k1)
    out[1] = np.double(kg)
    out[2] = np.double(k1)
    out[3] = np.double(kg)
    out[4] = np.double(k1)
    out[5] = np.double(kg)


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
tspan = np.linspace(0, 5, 5)

# Create FSP solver object
solver = FspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, t_fun, propensity)
solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)

t1 = MPI.Wtime()
solution = solver.SolveTspan(tspan, 1.0e-4)
if (rank == 0):
    print("Solve with CVODE takes " + str(MPI.Wtime() - t1))

ntspan = tspan.size
marginals = []
for i in range(0, 3):
    for j in range(0, ntspan):
        marginals.append(solution[j].Marginal(i))

species = ['S1','S2','S3']
fig = plt.figure()
fig.set_size_inches(10, 10)
if rank is 0:
    for i in range(0, 3):
        for j in range(0, ntspan):
            # marginals.append(solution.Marginal(i))
            ax = fig.add_subplot(3, ntspan, i * ntspan + 1 + j)
            ax.plot(marginals[i * ntspan  + j])
            ax.set_xlim(0,120)
            ax.set_ylim(0, 0.1)
            ax.grid(b=1)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            plt.setp(ax.get_xticklabels(), fontsize=10)
            plt.setp(ax.get_yticklabels(), fontsize=10)

            if j is 0:
                ax.set_ylabel('Probability')
            else:
                ax.set_yticklabels([])

            if j is ntspan:
                ax.set_title(species[i], loc='right')

            if i is 0:
                ax.set_title('t = ' + str(tspan[j]) + ' min')

            if i is 2:
                ax.set_xlabel('Molecule count')
            else:
                ax.set_xticklabels([])

    fig.savefig('rep_snapshots_cvode.eps', format='eps')

solver.ClearState()
solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
solver.SetOdeSolver("KRYLOV")
t1 = MPI.Wtime()
solution = solver.SolveTspan(tspan, 1.0e-4)
if (rank == 0):
    print("Solve with Krylov takes " + str(MPI.Wtime() - t1))


ntspan = tspan.size
marginals = []
for i in range(0, 3):
    for j in range(0, ntspan):
        marginals.append(solution[j].Marginal(i))

species = ['S1','S2','S3']
fig = plt.figure()
fig.set_size_inches(10, 10)
if rank is 0:
    for i in range(0, 3):
        for j in range(0, ntspan):
            # marginals.append(solution.Marginal(i))
            ax = fig.add_subplot(3, ntspan, i * ntspan + 1 + j)
            ax.plot(marginals[i * ntspan  + j])
            ax.set_xlim(0,120)
            ax.set_ylim(0, 0.1)
            ax.grid(b=1)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            plt.setp(ax.get_xticklabels(), fontsize=10)
            plt.setp(ax.get_yticklabels(), fontsize=10)

            if j is 0:
                ax.set_ylabel('Probability')
            else:
                ax.set_yticklabels([])

            if j is ntspan:
                ax.set_title(species[i], loc='right')

            if i is 0:
                ax.set_title('t = ' + str(tspan[j]) + ' min')

            if i is 2:
                ax.set_xlabel('Molecule count')
            else:
                ax.set_xticklabels([])

    fig.savefig('rep_snapshots_krylov.eps', format='eps')
    plt.show()
