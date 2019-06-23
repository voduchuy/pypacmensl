import unittest
import pypacmensl.PACMENSL as pac
import mpi4py.MPI as mpi
import numpy as np


class TestStateSet(unittest.TestCase):

    def test_handling_wrong_stoichiometry_input(self):
        state_set = pac.StateSetConstrained(comm=mpi.COMM_SELF, n_species = 1)
        sm = np.array([[0,1],[1,-1]], dtype=np.intc)
        self.assertRaises(RuntimeError, state_set.SetStoichiometry, sm)

    def test_handling_wrong_state_input(self):
        state_set = pac.StateSetConstrained(comm=mpi.COMM_SELF, n_species=1)
        x0 = np.array([[0,1],])
        self.assertRaises(RuntimeError, state_set.AddStates, x0)

    def test_new_states_add(self):
        state_set = pac.StateSetConstrained(comm=mpi.COMM_SELF, n_species=2)
        x0 = np.array([[0, 1], ])
        state_set.AddStates(x0)
        x1 = state_set.GetStates()
        self.assertEqual(x0.shape, x1.shape)
        self.assertEqual(x0[0,0], x1[0,0])

    def test_expand(self):
        state_set = pac.StateSetConstrained(comm=mpi.COMM_SELF, n_species=2)
        x0 = np.array([[0, 0], ])
        state_set.AddStates(x0)
        sm = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.intc)
        state_set.SetStoichiometry(sm)

        def constr(X, out):
            n_constr = 2
            out[0::n_constr] = X[:,0]
            out[1::n_constr] = X[:,1]

        bounds = np.array([1, 9])
        state_set.SetConstraint(constr, bounds)
        state_set.Expand()
        x1 = state_set.GetStates()
        self.assertEqual(x1.shape, (20, 2))


if __name__ == '__main__':
    unittest.main()