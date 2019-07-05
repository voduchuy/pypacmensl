import unittest
import mpi4py.MPI as mpi
import numpy as np
import pypacmensl.state_set.constrained as pac


class TestStateSet(unittest.TestCase):

    def test_handling_wrong_state_input(self):
        omega = pac.StateSetConstrained(comm=mpi.COMM_SELF)
        x0 = np.array([[0,1],])
        omega.SetStoichiometry(np.array([1]))
        self.assertRaises(RuntimeError, omega.AddStates, x0)

    def test_expand(self):
        x0 = np.array([[0, 0], ])
        sm = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.intc)

        def constr(X, out):
            n_constr = 2
            out[:, 0] = X[:, 0]
            out[:, 1] = X[:, 1]
        bounds = np.array([1, 9])

        omega = pac.StateSetConstrained(comm=mpi.COMM_SELF)
        omega.SetStoichiometry(sm)
        omega.SetConstraint(constr, bounds)
        omega.SetUp()
        omega.AddStates(x0)
        print(omega.GetNumSpecies())
        omega.Expand()
        x1 = omega.GetStates()
        self.assertEqual(x1.shape, (20, 2))


if __name__ == '__main__':
    unittest.main()