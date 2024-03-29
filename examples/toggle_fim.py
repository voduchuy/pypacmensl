import pypacmensl.sensitivity.multi_sinks as sensfsp
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt

# %%
stoich_matrix = np.array([[1, 0],
                          [-1, 0],
                          [0, 1],
                          [0, -1]])


def propensity(reaction, states, outs):
    if reaction == 0:
        outs[:] = 35.0*np.reciprocal(1 + states[:, 1] ** 1.5)
        return
    if reaction == 1:
        outs[:] = 0.20*states[:, 0]
        return
    if reaction == 2:
        outs[:] = 50.0*np.reciprocal(1 + states[:, 0] ** 2.5)
        return
    if reaction == 3:
        outs[:] = 1.0*states[:, 1]

def dpropensity(parameter, reaction, states, outs):
    if parameter == 0:
        if reaction == 0:
            outs[:] = np.reciprocal(1 + states[:, 1] ** 1.5)
            return
    if parameter == 1:
        if reaction == 1:
            outs[:] = states[:, 0]
            return
    if parameter == 2:
        if reaction == 2:
            outs[:] = np.reciprocal(1 + states[:, 0] ** 2.5)
            return
    if parameter == 3:
        if reaction == 3:
            outs[:] = states[:, 1]


def simple_constr(X, out):
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]
    out[:, 2] = X[:, 0] + X[:, 1]

init_bounds = np.array([10, 10, 10])

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
my_solver.SetModel(
    4,
    stoich_matrix=stoich_matrix,
    propensity_t=None,
    propensity_x=propensity,
    tv_reactions=[],
    d_propensity_t=None,
    d_propensity_t_sp=None,
    d_propensity_x=dpropensity,
    d_propensity_x_sp=[[i] for i in range(4)]
)
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
if comm.Get_rank() == 0:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.rc("text", usetex=True)
    plt.plot(t_meas, np.log10(DetFIMs), label="Observing both species")
    plt.plot(t_meas, np.log10(DetFIMs0), label="Observing species 0")
    plt.plot(t_meas, np.log10(DetFIMs1), label="Observing species 1")
    plt.title(
        "Log10-determinant of the Fisher Information Matrix for different combinations of observables"
    )
    plt.xlabel("Time (sec)")
    plt.ylabel(r"$\log_{10}(|FIM|)$")
    plt.show()
