from pypacmensl.stationary import StationaryFspSolverMultiSinks
from pypacmensl.fsp_solver import FspSolverMultiSinks
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

stoich_mat = np.array([ [-1, 1, 0, 0],
                        [1, -1, 0, 0],
                        [0, -1, 1, 0],
                        [0, 1, -1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, -1]])
x0 = np.array([[1, 0, 0, 0]])
p0 = np.array([1.0])
constr_init = np.array([1, 1, 1, 2])

k01 = 3.89e-2
k10 = 7.62e-3
k12 = 3.93e-5
k21 = 8.37e-3
a0 = 1.09e-4
a1 = 1.64e-5
a2 = 9.99e-1
deg = 5.67e-5
def propensity(reaction, x, out):
    if reaction == 0:
        out[:] = k01*x[:,0]
    if reaction == 1:
        out[:] = k10*x[:,1]
    if reaction == 2:
        out[:] = k12*x[:,1]
    if reaction == 3:
        out[:] = k21*x[:,2]
    if reaction == 4:
        out[:] = a0*x[:, 0] + a1*x[:, 1] + a2*x[:, 2]
    if reaction == 5:
        out[:] = deg*x[:, 3]

solver = StationaryFspSolverMultiSinks(mpi.COMM_WORLD)
solver.SetModel(stoich_mat, None, propensity)
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()

t1 = mpi.Wtime()
solution = solver.Solve(1e-4)
t2 = mpi.Wtime()


marginals = []
for i in range(0, 4):
    marginals.append(solution.Marginal(i))

rank = mpi.COMM_WORLD.rank

if rank == 0:
    print(f"Wall clock time: {t2-t1: .2e}")
    
    fig = plt.figure(tight_layout=True)
    for i in range(0, 4):                    
        ax = fig.add_subplot(4, 1, i+1)
        ax.bar(np.arange(0, len(marginals[i])), marginals[i])             
        ax.grid(visible=True)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        
        ax.set_ylabel('Probability')
        ax.set_xlabel('Molecule count')
        # ax.set_xticks(range(0, len(marginals[i])))

        print(np.sum(np.arange(0, len(marginals[i]))*marginals[i]))

    fig.suptitle("Stationary solution prior to LPS")

    fig.savefig('const_ge_stationary.pdf', format='pdf')
    plt.show()