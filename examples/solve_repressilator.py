import mpi4py.MPI as MPI
import pypacmensl.PACMENSL as pacmensl
import numpy as np
import matplotlib.pyplot as plt


def propensity(reaction, X, out):
    if reaction is 0:
        out[:] = np.conjugate(1 + X[:,2])
        return 0
    if reaction is 1:
        out[:] = X[:,0]
        return 0
    if reaction is 2:
        out[:] = np.conjugate(1 + X[:,0])
        return 0
    if reaction is 3:
        out[:] = X[:,1]
        return 0
    if reaction is 4:
        out[:] = np.conjugate(1 + X[:,1])
        return 0
    if reaction is 5:
        out[:] = X[:,2]
        return 0


def t_fun(time, out):
    out[:] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.double)


def rep_constr(X, out):
    n_constr = 6
    out[::n_constr] = X[:, 0]
    out[1::n_constr] = X[:, 1]
    out[2::n_constr] = X[:, 2]
    out[3::n_constr] = np.multiply(X[:,0],X[:,1])
    out[4::n_constr] = np.multiply(X[:, 2], X[:, 1])
    out[5::n_constr] = np.multiply(X[:, 0], X[:, 2])


init_bounds = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
exp_factors = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

x0 = np.array([[0,0,0]])
p0 = np.array([1.0])
sm = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
tspan = np.array([0.0, 1.0, 2.5, 5, 10.0])
# Create sequential solver object
solver = pacmensl.FspSolverMultiSinks(MPI.COMM_WORLD)
solver.SetModel(sm, t_fun, propensity)
solver.SetFspShape(constr_fun=rep_constr, constr_bound=init_bounds)
solver.SetInitialDist(x0, p0)
solver.SetVerbosity(2)
solver.SetUp()
solution = solver.SolveTspan(tspan, 1.0e-4)

ntspan  =tspan.size
fig = plt.figure()
for i in range(0,3):
    for j in range(0,ntspan):
        # marginals.append(solution.Marginal(i))
        ax = fig.add_subplot(3,ntspan,i*ntspan + 1 + j)
        ax.plot(solution[j].Marginal(i))
plt.show()


