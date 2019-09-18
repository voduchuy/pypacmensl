import unittest
import pypacmensl.fsp_solver.multi_sinks as pac
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
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]

init_bounds=np.array([10, 10])

class TestFspSolver(unittest.TestCase):
    def setUp(self):
        self.stoich_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    def test_serial_constructor(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_SELF)

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
        solution = solver.Solve(10.0, 1.0E-6)
        prob = np.asarray(solution.GetProbViewer())
        self.assertAlmostEqual(prob.sum(), 1.0, 4)

    def test_solve_parallel(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)
        solver.SetFspShape(simple_constr, init_bounds)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        solver.SetInitialDist(X0, p0)
        solution = solver.Solve(10.0, 1.0E-6)
        prob = np.asarray(solution.GetProbViewer())
        psum1 = prob.sum()
        psum = mpi.COMM_WORLD.allreduce(psum1)
        self.assertAlmostEqual(psum, 1.0, 4)

    def test_solve_parallel_krylov(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)
        solver.SetFspShape(simple_constr, init_bounds)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        solver.SetInitialDist(X0, p0)
        solver.SetOdeSolver("Krylov")
        solution = solver.Solve(10.0, 1.0e-6)
        prob = np.asarray(solution.GetProbViewer())
        psum1 = prob.sum()
        psum = mpi.COMM_WORLD.allreduce(psum1)
        self.assertAlmostEqual(psum, 1.0, 4)

    def test_solve_parallel_twice(self):
        solver = pac.FspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(self.stoich_mat, tcoeff, propensity)
        solver.SetFspShape(simple_constr, init_bounds)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        solver.SetInitialDist(X0, p0)
        solver.SetOdeSolver("Krylov")
        solution = solver.Solve(10.0, 1.0e-6)
        solver.ClearState()
        solver.SetInitialDist(X0, p0)
        solution = solver.Solve(10.0, 1.0e-4)
        prob = np.asarray(solution.GetProbViewer())
        psum1 = prob.sum()
        psum = mpi.COMM_WORLD.allreduce(psum1)
        self.assertAlmostEqual(psum, 1.0, 4)

if __name__ == '__main__':
    unittest.main()
