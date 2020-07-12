from pypacmensl.fsp_solver import FspSolverMultiSinks
from pypacmensl.ssa.ssa import PmPdmsrSampler
import mpi4py.MPI as mpi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

#%% Rate constants
K01 = 0.01
K10 = 0.001
BETA1 = 1.0
DEGRADATION = 0.1
#%% This block of code defines the telegraph model to be solved with FSP
stoich_mat = np.array([ [-1, 1, 0],
                        [1, -1, 0],
                        [0, 0, 1],
                        [0, 0, -1]])
x0 = np.array([[1, 0, 0]])
p0 = np.array([1.0])
constr_init = np.array([1, 1, 100])

def propensity(reaction, x, out):
    if reaction == 0:
        out[:] = K01*x[:,0]
    if reaction == 1:
        out[:] = K10*x[:,1]
    if reaction == 2:
        out[:] = BETA1*x[:,1]
    if reaction == 3:
        out[:] = DEGRADATION*x[:,2]

n_t = 5
tspan = np.linspace(0, 100, n_t)
#%%
def transcription_rate(gene_state):
    return np.array([(gene_state[1] == 1)*BETA1])

gene_transition_stoich = np.array([ [-1, 1],
                                    [1, -1]])

def gene_transition_propensity(reaction, x, out):
    if reaction == 0:
        out[:] = K01*x[:,0]
    if reaction == 1:
        out[:] = K10*x[:,1]
#%% Find the marginal mRNA distributions with FSP
solver = FspSolverMultiSinks(mpi.COMM_WORLD)
solver.SetModel(stoich_mat, None, propensity)
solver.SetFspShape(None, constr_init)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetOdeSolver("KRYLOV")
solver.SetUp()
solutions = solver.SolveTspan(tspan, 1.0e-4)

marginals_fsp = []
for j in range(0, n_t):
    marginals_fsp.append(solutions[j].Marginal(2))
#%% Monte Carlo approximation with PMPDMSR
def rna_dist_from_samples(comm: mpi.Comm, poisson_samples: np.ndarray, nmax: int) -> np.ndarray:

    nsamples_local = len(poisson_samples)
    nsamples = comm.allreduce(nsamples_local, mpi.SUM)
    nmax = comm.allreduce(nmax, mpi.MAX)

    p_local = np.zeros((nmax+1,), dtype=float)
    x_eval = np.arange(0, nmax+1, dtype=int)
    for j in range(0, len(poisson_samples)):
        p_local += poisson.pmf(x_eval, mu=poisson_samples[j])
    p_local = comm.allreduce(p_local, mpi.SUM)
    p_local = (1.0/nsamples)*p_local
    return p_local


sampler = PmPdmsrSampler(mpi.COMM_WORLD)
sampler.SetModel(gene_transition_stoich, None, gene_transition_propensity, transcription_rate, np.array([DEGRADATION]))
poisson_samples = []
marginals_pmpdmsr = []
for j in range(0, n_t):
    _, p_samples = sampler.SolveTv(tspan[j], np.array([[1, 0]], dtype=int), 10000, send_to_root=False)
    poisson_samples.append(p_samples)
    marginals_pmpdmsr.append(rna_dist_from_samples(mpi.COMM_WORLD, p_samples, 100))

if mpi.COMM_WORLD.Get_rank() == 0:
    fig, axs = plt.subplots(1, n_t)
    for j in range(0, n_t):
        axs[j].plot(np.arange(0, 101), marginals_pmpdmsr[j], color="r", label="PMPDMSR")
        axs[j].plot(marginals_fsp[j], color="b", label="FSP")
        axs[j].legend()
    plt.show()
