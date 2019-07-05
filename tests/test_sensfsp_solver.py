import unittest
import mpi4py.MPI as mpi
import numpy as np
import pypacmensl.sensitivity.multi_sinks as sensfsp


def tcoeff(t, out):
    out[0] = 1
    out[1] = 1
    out[2] = 1
    out[3] = 1


def dtcoeff_factory(i):
    def dtcoeff(t, out):
        out[i] = 1

    return dtcoeff


def propensity(reaction, states, outs):
    if reaction is 0:
        outs[:] = np.reciprocal(1 + states[:, 1])
        return
    if reaction is 1:
        outs[:] = states[:, 0]
        return
    if reaction is 2:
        outs[:] = np.reciprocal(1 + states[:, 0])
        return
    if reaction is 3:
        outs[:] = states[:, 1]


def simple_constr(X, out):
    # The spear of Adun
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]


init_bounds = np.array([10, 10, 10])

dtcoeff = []
for i in range(0, 4):
    dtcoeff.append(dtcoeff_factory(i))
dpropensity = [propensity] * 4

dprop_sparsity = np.zeros((4, 4), dtype=np.intc)
for i in range(0, 4):
    dprop_sparsity[i][i] = 1


class TestFspSolver(unittest.TestCase):
    def setUp(self):
        self.stoich_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    def test_serial_constructor(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_SELF)

    def test_set_model(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity, dtcoeff, dpropensity, dprop_sparsity)

    def test_set_initial_distribution(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity, dtcoeff, dpropensity, dprop_sparsity)
        X0 = np.array([[0, 0]])
        p0 = np.array([1.0])
        s0 = np.array([0.0])
        solver.SetInitialDist(X0, p0, [s0] * 4)

    def test_set_shape(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity, dtcoeff, dpropensity)
        solver.SetFspShape(simple_constr, init_bounds)

    def test_solve_serial(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_SELF)
        solver.SetModel(self.stoich_mat, tcoeff, propensity, dtcoeff, dpropensity, dprop_sparsity)
        solver.SetFspShape(simple_constr, init_bounds)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        s0 = np.array([0.0])
        solver.SetInitialDist(X0, p0, [s0]*4)
        solution = solver.Solve(10.0, 1.0e-4)
        prob = np.asarray(solution.GetProbViewer())
        self.assertAlmostEqual(prob.sum(), 1.0, 4)
        for i in range(0,4):
            svec = np.asarray(solution.GetSensViewer(i))
            self.assertAlmostEqual(sum(svec), 0.0, 2)
            solution.RestoreSensViewer(i, svec)


if __name__ == '__main__':
    unittest.main()
