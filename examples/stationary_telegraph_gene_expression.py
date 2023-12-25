from pypacmensl.stationary import StationaryFspSolverMultiSinks
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
constr_init = np.array([1, 1, 2])

def propensity(reaction, x, out):
    if reaction == 0:
        out[:] = x[:,0]
    if reaction == 1:
        out[:] = x[:,1]
    if reaction == 2:
        out[:] = x[:,1]
    if reaction == 3:
        out[:] = x[:,2]

def t_fun(t, out):
    out[0] = 0.05
    out[1] = 0.015
    out[2] = 5
    out[3] = 0.1

solver = StationaryFspSolverMultiSinks(mpi.COMM_WORLD)
solver.SetModel(stoich_mat, t_fun, propensity, [0, 1, 2, 3])
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()
solution = solver.Solve(1e-6)


marginals = []
for i in range(0, 3):
    marginals.append(solution.Marginal(i))

rank = mpi.COMM_WORLD.rank

if rank == 0:
    fig = plt.figure()
    for i in range(0, 3):                    
        ax = fig.add_subplot(3, 1, i+1)
        ax.bar(np.arange(0, len(marginals[i])), marginals[i])             
        ax.grid(visible=True)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        plt.setp(ax.get_xticklabels(), fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=10)
        
        ax.set_ylabel('Probability')
        ax.set_xlabel('Molecule count')

    fig.savefig('const_ge_stationary.pdf', format='pdf')
    plt.show()