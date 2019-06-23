import unittest
import pypacmensl.PACMENSL as pac
import mpi4py.MPI as mpi
import numpy as np


def tcoeff(t, out):
    out[0] = 1
    out[1] = 1
    out[2] = 1
    out[3] = 1

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
    n_constr = 3
    assert(n_constr*X.shape[0] == out.size)
    out[0::n_constr] = X[:,0]
    out[1::n_constr] = X[:,1]
    out[2::n_constr] = X[:,0] + X[:,1]

init_bounds=np.array([10, 10, 10])

class TestFspSolver(unittest.TestCase):
    def setUp(self):
        self.stoich_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    def test_serial_constructor(self):
        solver = pac.FspSolverMultiSinks()

    def test_set_model(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)

    def test_set_initial_distribution(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        solver.SetInitialDist(X0, p0)

    def test_set_shape(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)
        solver.SetFspShape(simple_constr, init_bounds)

    def test_solve_serial(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_SELF)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)
        solver.SetFspShape(simple_constr, init_bounds)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        solver.SetInitialDist(X0, p0)
        solver.SetUp()
        solution = solver.Solve(10.0, 1.0e-4)
        prob = np.asarray(solution.GetProbViewer())
        self.assertAlmostEqual(prob.sum(), 1.0, 4)

if __name__ == '__main__':
    unittest.main()
