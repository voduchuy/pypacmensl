import pypacmensl.sensitivity.multi_sinks as sensfsp
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg as LA

# %%
stoich_matrix = np.array(
    [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ]
)


def tcoeff(t, out):
    out[0] = 35.0
    out[1] = 0.20
    out[2] = 50.0
    out[3] = 1.0


def propensity(reaction, states, outs):
    if reaction is 0:
        outs[:] = np.reciprocal(1 + states[:, 1] ** 1.5)
        return
    if reaction is 1:
        outs[:] = states[:, 0]
        return
    if reaction is 2:
        outs[:] = np.reciprocal(1 + states[:, 0] ** 2.5)
        return
    if reaction is 3:
        outs[:] = states[:, 1]


def simple_constr(X, out):
    # The spear of Adun
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]
    out[:, 2] = X[:, 0] + X[:, 1]


init_bounds = np.array([10, 10, 10])


def d_tcoeff_factory(i):
    def d_tcoeff(t, out):
        out[i] = 1.0

    return d_tcoeff


dtcoeff = []
for i in range(0, 4):
    dtcoeff.append(d_tcoeff_factory(i))
dpropensity = [propensity] * 4

dprop_sparsity = np.zeros((4, 4), dtype=np.intc)
for i in range(0, 4):
    dprop_sparsity[i][i] = 1

X0 = np.array([[0, 0]])
p0 = np.array([1.0])
s0 = np.array([0.0])

n_cells = 1000

t_meas = np.linspace(0, 10)
# %%

comm = mpi.COMM_WORLD
my_rank = comm.rank
# %%
my_solver = sensfsp.SensFspSolverMultiSinks(comm)
my_solver.SetModel(stoich_matrix, tcoeff, propensity,
                dtcoeff, dpropensity, dprop_sparsity)
my_solver.SetFspShape(constr_fun=simple_constr, constr_bound=init_bounds)
my_solver.SetInitialDist(X0, p0, [s0] * 4)
my_solver.SetVerbosity(2)
solutions = my_solver.SolveTspan(t_meas, 1.0e-6)
# %%
# FIMs for observing both species
FIMatrices = []
for v in solutions:
    FIMatrices.append(v.ComputeFIM())

# %%
# FIMs for observing species 0
FIMatrices0 = []
for v in solutions:
    I = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            si = v.SensMarginal(i, 0)
            sj = v.SensMarginal(j, 0)
            p = v.Marginal(0)
            I[i, j] = np.dot(si, np.divide(sj, p))
    FIMatrices0.append(I)
# %%
# FIMs for observing species 1
FIMatrices1 = []
for v in solutions:
    I = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            si = v.SensMarginal(i, 1)
            sj = v.SensMarginal(j, 1)
            p = v.Marginal(1)
            I[i, j] = np.dot(si, np.divide(sj, p))
    FIMatrices1.append(I)
# %%
DetFIMs = np.zeros(len(FIMatrices))
for i in range(0, len(DetFIMs)):
    DetFIMs[i] = np.linalg.det(n_cells * FIMatrices[i])
# %%
DetFIMs0 = np.zeros(len(FIMatrices0))
for i in range(0, len(DetFIMs0)):
    DetFIMs0[i] = np.linalg.det(n_cells * FIMatrices0[i])

DetFIMs1 = np.zeros(len(FIMatrices1))
for i in range(0, len(DetFIMs1)):
    DetFIMs1[i] = np.linalg.det(n_cells * FIMatrices1[i])
# %%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.rc('text', usetex=True)
plt.plot(t_meas, np.log10(DetFIMs))
plt.plot(t_meas, np.log10(DetFIMs0))
plt.plot(t_meas, np.log10(DetFIMs1))
plt.xlabel('Time (sec)')
plt.ylabel(r'$\log_{10}(|FIM|)$')
plt.show()

# %%
s = solutions[20]
X = s.GetStatesViewer()
p = s.GetProbViewer()

plt.scatter(X[:, 0], X[:, 1], c=np.log10(p))
plt.colorbar()
