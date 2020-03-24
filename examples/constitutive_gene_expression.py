from pypacmensl.fsp_solver import FspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

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

n_t = 100
tspan = np.linspace(0, 1000, n_t)

solver = FspSolverMultiSinks(mpi.COMM_WORLD)
solver.SetModel(stoich_mat, t_fun, propensity)
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()
solutions = solver.SolveTspan(tspan, 1.0e-4)


marginals = []
for i in range(0, 3):
    for j in range(0, n_t):
        marginals.append(solutions[j].Marginal(i))


def weightfun(x, fout):
    fout[0] = 1.0*x[2]
    fout[1] = x[2]*x[2]


meanvar = np.zeros((n_t, 2), dtype=np.double)

for j in range(0, n_t):
    meanvar[j, :] = solutions[j].WeightedAverage(2, weightfun)

rank = mpi.COMM_WORLD.rank
if rank == 0:
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    sns.set(style='darkgrid')
    std = np.sqrt(meanvar[:, 1] - meanvar[:, 0]*meanvar[:,0])
    ax = fig.add_subplot(1,1,1)
    ax.plot(tspan, meanvar[:, 0])
    ax.fill_between(tspan, meanvar[:, 0] + std, meanvar[:, 0] - std, alpha=0.5)
    fig.savefig('const_ge_rna_mean_var.pdf')

# if rank is 0:
#     for i in range(0, 3):
#         for j in range(0, n_t):
#             # marginals.append(solution.Marginal(i))
#             ax = fig.add_subplot(3, n_t, i * n_t + 1 + j)
#             ax.plot(marginals[i * n_t + j])
#             ax.set_ylim(0, 1)
#             ax.grid(b=1)
#
#             ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
#             plt.setp(ax.get_xticklabels(), fontsize=10)
#             plt.setp(ax.get_yticklabels(), fontsize=10)
#
#             if j is 0:
#                 ax.set_ylabel('Probability')
#             # else:
#             #     ax.set_yticklabels([])
#
#             if i is 0:
#                 ax.set_title('t = ' + str(tspan[j]) + ' min')
#
#             if i is 2:
#                 ax.set_xlabel('Molecule count')
#             # else:
#             #     ax.set_xticklabels([])
#
#     fig.savefig('const_ge_snapshots.eps', format='eps')
#     plt.show()