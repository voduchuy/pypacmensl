import mpi4py.MPI as MPI
import pypacmensl.PACMENSL as pacmensl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import exp


Ahog = 9.3E09
num_genes = 1
Mhog = 2.2E-2
etahog = 5.9
r1 = 6.1E-3
r2 = 6.9E-3

def hog1p(t):
    """Hog1p signal in arbitrary unit (AU)"""
    return (1.0 - exp(-r1 * t) )* exp(-r2 * t)


def hog1pstar(t):
    """Saturated hog1p signal"""
    return ((hog1p(t) / (1.0 + hog1p(t)/Mhog)) ** etahog)*Ahog


t0 = 2.6E-1
k01 = 2.6E-3
k10a = 1.0E1
k10b = 3.2E4
k12 = 7.6E-3
k21 = 1.2E-2
k23 = 4E-3
k32 = 3.1E-3
kr0 = 5.9E-4
kr1 = 1.7E-1
kr2 = 1.0
kr3 = 3.2E-2
gamma_nuc = 2.2E-6
k_transport = 2.6E-1
gamma_cyt = 19.3E-3


x0 = np.array([[num_genes, 0, 0, 0, 0, 0]])
p0 = np.array([1.0])
sm = np.array([[-1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, 0, 0],
               [0, -1, 1, 0, 0, 0],
               [0, 1, -1, 0, 0, 0],
               [0, 0, -1, 1, 0, 0],
               [0, 0, 1, -1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, -1, 0],
               [0, 0, 0, 0, -1, 1],
               [0, 0, 0, 0, 0, -1]
               ])


# // TODO: automatic
# generation
# of
# mass
# action
# propensity


def propensity(reaction, X, out):
    if reaction > 12:
        raise RuntimeError("The model has only 13 reactions!")
    if reaction is 0:
        out[:] = X[:, 0]
        return
    if reaction is 1:
        out[:] = X[:, 1]
        return
    if reaction is 2:
        out[:] = X[:, 1]
        return
    if reaction is 3:
        out[:] = X[:, 2]
        return
    if reaction is 4:
        out[:] = X[:, 2]
        return
    if reaction is 5:
        out[:] = X[:, 3]
        return
    if reaction is 6:
        out[:] = X[:, 0]
        return
    if reaction is 7:
        out[:] = X[:, 1]
        return
    if reaction is 8:
        out[:] = X[:, 2]
        return
    if reaction is 9:
        out[:] = X[:, 3]
        return
    if reaction is 10:
        out[:] = X[:, 4]
        return
    if reaction is 11:
        out[:] = X[:, 4]
        return
    if reaction is 12:
        out[:] = X[:, 5]
        return


def t_fun(time, out):
    out[0] = np.double(k01)
    if time > t0:
        out[1] = np.double(max( 0.0, k10a - k10b*hog1pstar(time - t0)))
    else:
        out[1] = 0.0
    out[2] = np.double(k12)
    out[3] = np.double(k21)
    out[4] = np.double(k23)
    out[5] = np.double(k32)
    out[6] = np.double(kr0)
    out[7] = np.double(kr1)
    out[8] = np.double(kr2)
    out[9] = np.double(kr3)
    out[10] = np.double(gamma_nuc)
    out[11] = np.double(k_transport)
    out[12] = np.double(gamma_cyt)


init_bounds = np.array([1, 1, 1, 1, 10, 10])

# Create parallel solver object
solver = pacmensl.FspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, t_fun, propensity)
solver.SetFspShape(constr_fun=None, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()

tspan = np.linspace(0, 60*15, 5)
solution = solver.SolveTspan(tspan, 1.0e-2)





ntspan = tspan.size
marginals = []
for i in range(0, 6):
    for j in range(0, ntspan):
        marginals.append(solution[j].Marginal(i))

rank = MPI.COMM_WORLD.rank
fig = plt.figure()
fig.set_size_inches(10, 10)
if rank is 0:
    for i in range(4, 6):
        for j in range(0, ntspan):
            # marginals.append(solution.Marginal(i))
            ax = fig.add_subplot(2, ntspan, (i-4) * ntspan + 1 + j)
            ax.plot(marginals[i * ntspan  + j])
            ax.fill_between(range(0,marginals[i*ntspan+j].size), 0, marginals[i * ntspan  + j])
            ax.set_xlim(left=0, auto=True)
            ax.set_ylim(0, 1)
            ax.grid(b=1)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            plt.setp(ax.get_xticklabels(), fontsize=10)
            plt.setp(ax.get_yticklabels(), fontsize=10)

            if j is 0:
                ax.set_ylabel('Probability')
            else:
                ax.set_yticklabels([])

            if i is 4:
                ax.set_title('t = ' + str(tspan[j]) + ' min')

            if i is 5:
                ax.set_xlabel('Molecule count')

    fig.savefig('hog_snapshots.eps', format='eps')
    plt.show()
