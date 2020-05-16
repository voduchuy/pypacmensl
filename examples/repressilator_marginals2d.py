import mpi4py.MPI as MPI
from pypacmensl.fsp_solver import FspSolverMultiSinks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

rank = MPI.COMM_WORLD.rank


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



x0 = np.array([[20, 0, 0]])
p0 = np.array([1.0])
sm = np.array([[1, 0, 0],
               [-1, 0, 0],
               [0, 1, 0],
               [0, -1, 0],
               [0, 0, 1],
               [0, 0, -1]])

t_exports = np.linspace(0, 10, 20, dtype=float) #np.array([0.1, 1, 5, 10], dtype=float)
#
# # Create FSP solver object
# solver = FspSolverMultiSinks(MPI.COMM_WORLD)
# solver.SetModel(sm, None, propensity)
# solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
# solver.SetInitialDist(x0, p0)
# solver.SetOdeSolver("CVODE")
# solver.SetVerbosity(2)
#
# t1 = MPI.Wtime()
# solutions = solver.SolveTspan(t_exports, 1.0e-1)
# t_cvode = MPI.Wtime() - t1
#
# #%%
# def compute_2d_marginals(distribution, species, nmax):
#     nout = (nmax+1)*(nmax+1)
#     s0 = species[0]
#     s1 = species[1]
#     def marginalize2d(X, fout):
#         '''
#         fout[i + j*(nmax+1)] = 1 if X = [i,j]
#         '''
#         fout[:] = 0
#         fout[X[s0] + X[s1]*(nmax+1)] = 1
#     return distribution.WeightedAverage(nout, marginalize2d)
# #%%
# spec_pairs = [[0,1], [1,2], [2,0]]
# twospec_marginals = []
# NMAX = 150
# for i in range(0, len(t_exports)):
#     twospec_marginals.append([])
#     for sp in spec_pairs:
#         twospec_marginals[i].append(np.reshape(compute_2d_marginals(solutions[i], sp, NMAX), (NMAX+1, NMAX+1)))
#
# #%%
#
# if rank == 0:
#     np.savez('repressilator_2dmarginals.npz', allow_pickle=True, marginals2=twospec_marginals)
#%%
if rank == 0:
    with np.load('repressilator_2dmarginals.npz', allow_pickle=True) as file:
        twospec_marginals = file['marginals2']

    import matplotlib.colors as colors

    spec_pairs = [[0,1], [1,2], [2,0]]
    spec_names = ['TetR', 'LacI', '$\lambda$cI']
    export_idx = [1, 8, 10, 19]

    fig, axs = plt.subplots(len(spec_pairs), len(export_idx))
    fig.set_size_inches(6, 8)
    fig.set_tight_layout(True)
    for i in range(0, len(spec_pairs)):
        for j in range(0, len(export_idx)):
            p1 = axs[i,j].contourf(np.maximum(twospec_marginals[export_idx[j]][i], 1.0e-5),
                            norm=colors.LogNorm(
                    vmin=1.0E-5, vmax=1.0E0))
            axs[i,j].grid()
            axs[i,j].set_xlabel(spec_names[spec_pairs[i][0]])
            axs[i,j].set_ylabel(spec_names[spec_pairs[i][1]])
            # if i == 0:
            axs[i,j].set_title(f't={t_exports[export_idx[j]]:.2f}')
        fig.colorbar(p1, ax=axs[i,-1], orientation='vertical', extend='min', label="Probability", drawedges=True)

    plt.show()