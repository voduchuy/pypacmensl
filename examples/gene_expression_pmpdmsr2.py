from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.ssa.ssa import PmPdmsrSampler, SSASolver
from pypacmensl.smfish.snapshot import SmFishSnapshot
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

#%% Rate constants
K01 = 0.01
K10 = 0.001
BETA1 = 1.0
BETA2 = 0.5
DEGRADATION = 0.1
#%% This block of code defines the telegraph model to be solved with FSP
stoich_mat = np.array(
    [
        [-1, 1, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
    ]
)
x0 = np.array([[1, 0, 0, 0]])
p0 = np.array([1.0])
constr_init = np.array([1, 1, 100, 100])

def propensity_t(t, out):
    out[:] = 1.0
    # out[1] = 1.0+np.cos(2*t)

def propensity(reaction, x, out):
    if reaction == 0:
        out[:] = K01 * x[:, 0]
    if reaction == 1:
        out[:] = K10 * x[:, 1]
    if reaction == 2:
        out[:] = BETA1 * x[:, 1]
    if reaction == 3:
        out[:] = BETA2 * x[:, 1]
    if reaction == 4:
        out[:] = DEGRADATION * x[:, 2]
    if reaction == 5:
        out[:] = DEGRADATION * x[:, 3]


n_t = 5
tspan = np.linspace(0, 100, n_t)
#%%
def transcription_rate(gene_state):
    return np.array([(gene_state[1] == 1) * BETA1, (gene_state[1] == 1) * BETA2])


gene_transition_stoich = np.array([[-1, 1], [1, -1]])


def gene_transition_propensity(reaction, x, out):
    if reaction == 0:
        out[:] = K01 * x[:, 0]
    if reaction == 1:
        out[:] = K10 * x[:, 1]


#%% Find the marginal mRNA distributions with FSP
solver = FspSolverMultiSinks(mpi.COMM_WORLD)
solver.SetModel(stoich_mat, propensity_t, propensity)
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetOdeSolver("PETSC")
solver.SetUp()
solutions = solver.SolveTspan(tspan, 1.0e-4)

marginals_fsp = []
for j in range(0, n_t):
    marginals_fsp.append([solutions[j].Marginal(2), solutions[j].Marginal(3)])
#%% Monte Carlo approximation with PMPDMSR

def joint_dist_from_poisson_parameters(
    comm: mpi.Comm, poisson_parameters: np.ndarray, nmax: int
):
    nsamples_local = poisson_parameters.shape[0]
    nsamples = comm.allreduce(nsamples_local, mpi.SUM)
    nmax = comm.allreduce(nmax, mpi.MAX)

    p_local = np.zeros((nmax + 1, nmax + 1), dtype=float)

    x_eval = np.arange(0, nmax + 1, dtype=int)
    p_marginals = np.zeros((2, nmax + 1), dtype=float)

    for j in range(0, nsamples_local):
        for ispecies in range(0, 2):
            p_marginals[ispecies, :] = poisson.pmf(
                x_eval, mu=poisson_parameters[j, ispecies]
            )
        p_local += np.kron(p_marginals[0, :], p_marginals[1, :]).reshape(
            (nmax + 1, nmax + 1)
        )
    p_global = comm.allreduce(p_local, mpi.SUM)
    p_global = (1.0 / nsamples) * p_global
    return p_global


sampler = PmPdmsrSampler(mpi.COMM_WORLD)
sampler.SetModel(
    gene_transition_stoich,
    propensity_t,
    gene_transition_propensity,
    transcription_rate,
    np.array([DEGRADATION]*2),
)
poisson_samples = []
pjoint_pmpdmsr = []
marginals_pmpdmsr = []
for j in range(0, n_t):
    _, p_samples = sampler.SolveTv(
        tspan[j], np.array([[1, 0]], dtype=int), 10000, update_rates=1.0E-7, send_to_root=False
    )
    poisson_samples.append(p_samples)
    pjoint = joint_dist_from_poisson_parameters(mpi.COMM_WORLD, p_samples, 100)
    marginals_pmpdmsr.append(
        [
            np.sum(pjoint, axis=1), np.sum(pjoint, axis=0)
        ]
    )
    pjoint_pmpdmsr.append(pjoint)
#
# if mpi.COMM_WORLD.Get_rank() == 0:
#     import matplotlib.colors as colors
#     fig, axs = plt.subplots(2, n_t)
#     for j in range(0, n_t):
#         for i in range(0, 2):
#             axs[i, j].plot(np.arange(0, 101), marginals_pmpdmsr[j][i], color="r", label="PMPDMSR")
#             axs[i, j].plot(marginals_fsp[j][i], color="b", label="FSP", linestyle=":")
#             axs[i, j].legend()
#     plt.show()
#
#     fig, axs = plt.subplots(1, n_t)
#     for j in range(0, n_t):
#         axs[j].pcolorfast(pjoint_pmpdmsr[j], norm=colors.LogNorm(vmin=1.0E-5, vmax=1.0))
#     plt.show()

#%% Likelihood computation

ssa = SSASolver(mpi.COMM_WORLD)
ssa.SetModel(stoich_mat, propensity_t, propensity)
data = []
raw_obs = []
for i in range(0, n_t):
    samples = ssa.SolveTv(tspan[j], np.array([[1, 0, 0, 0]], dtype=int), 1000, update_rates=1.0E-7, send_to_root=True)
    samples = mpi.COMM_WORLD.bcast(samples)
    data.append(SmFishSnapshot(samples[:, 2:]))
    raw_obs.append(samples[:,2:])

for i in range(0, n_t):
    ll_fsp = data[i].LogLikelihood(solutions[i], np.array([2,3]))
    ll_pm = sampler.compute_loglike(poisson_samples[i], raw_obs[i])
    if mpi.COMM_WORLD.Get_rank() == 0:
        print([ll_fsp, ll_pm, ll_fsp - ll_pm])

