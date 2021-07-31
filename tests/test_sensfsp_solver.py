import unittest
import mpi4py.MPI as mpi
import numpy as np
import pypacmensl.sensitivity.multi_sinks as sensfsp


def tcoeff(t, out):
    out[0] = 1
    out[1] = 1
    out[2] = 1
    out[3] = 1

def dtcoeff(parameter, t, out):
    if parameter == 0:
        out[0] = 1.0
    elif parameter == 1:
        out[1] = 1.0
    elif parameter == 2:
        out[2] = 1.0
    elif parameter == 3:
        out[3] = 1.0

def propensity(reaction, states, outs):
    if reaction == 0:
        outs[:] = np.reciprocal(1 + states[:, 1])
        return
    if reaction == 1:
        outs[:] = states[:, 0]
        return
    if reaction == 2:
        outs[:] = np.reciprocal(1 + states[:, 0])
        return
    if reaction == 3:
        outs[:] = states[:, 1]


def simple_constr(X, out):
    out[:, 0] = X[:, 0]
    out[:, 1] = X[:, 1]

init_bounds = np.array([10, 10])


class TestFspSolver(unittest.TestCase):
    def setUp(self):
        self.stoich_mat = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    def test_serial_constructor(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_SELF)

    def test_set_model(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(num_parameters=4,
                        stoich_matrix=self.stoich_mat,
                        propensity_t=tcoeff,
                        propensity_x=propensity,
                        tv_reactions=list(range(4)),
                        d_propensity_t=dtcoeff,
                        d_propensity_t_sp=[[i] for i in range(4)],
                        d_propensity_x=None
                        )

    def test_set_initial_distribution(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(num_parameters=4,
                        stoich_matrix=self.stoich_mat,
                        propensity_t=tcoeff,
                        propensity_x=propensity,
                        tv_reactions=list(range(4)),
                        d_propensity_t=dtcoeff,
                        d_propensity_t_sp=[[i] for i in range(4)],
                        d_propensity_x=None
                        )
        X0 = np.array([[0, 0]])
        p0 = np.array([1.0])
        s0 = np.array([0.0])
        solver.SetInitialDist(X0, p0, [s0] * 4)

    def test_set_shape(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_WORLD)
        solver.SetModel(num_parameters=4,
                        stoich_matrix=self.stoich_mat,
                        propensity_t=tcoeff,
                        propensity_x=propensity,
                        tv_reactions=list(range(4)),
                        d_propensity_t=dtcoeff,
                        d_propensity_t_sp=[[i] for i in range(4)],
                        d_propensity_x=None
                        )
        solver.SetFspShape(simple_constr, init_bounds)

    def test_solve_serial(self):
        solver = sensfsp.SensFspSolverMultiSinks(mpi.COMM_SELF)
        solver.SetModel(num_parameters=4,
                        stoich_matrix=self.stoich_mat,
                        propensity_t=tcoeff,
                        propensity_x=propensity,
                        tv_reactions=list(range(4)),
                        d_propensity_t=dtcoeff,
                        d_propensity_t_sp=[[i] for i in range(4)],
                        d_propensity_x=None
                        )
        solver.SetFspShape(simple_constr, init_bounds)
        X0 = np.array([[0,0]])
        p0 = np.array([1.0])
        s0 = np.array([0.0])
        solver.SetInitialDist(X0, p0, [s0]*4)
        solution = solver.Solve(10.0, 1.0E-4)
        prob = np.asarray(solution.GetProbViewer())
        self.assertAlmostEqual(prob.sum(), 1.0, 4)
        for i in range(0,4):
            svec = np.asarray(solution.GetSensViewer(i))
            self.assertAlmostEqual(sum(svec), 0.0, 2)
            solution.RestoreSensViewer(i, svec)


if __name__ == '__main__':
    unittest.main()
