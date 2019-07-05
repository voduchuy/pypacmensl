import unittest
import pypacmensl.fsp_solver.distribution as pac
import mpi4py.MPI as mpi
import numpy as np


class TestDiscreteDistribution(unittest.TestCase):
    def test_empty_constructor(self):
        dist = pac.DiscreteDistribution()

#
# def test_serial_constructor(self):
#     success = 1
#     try:
#         dist = pac.DiscreteDistribution(mpi.COMM_SELF)
#     except:
#         success = 0
#     self.assertEqual(success, 1)
#
# def test_parallel_constructor(self):
#     success = 1
#     try:
#         dist = pac.DiscreteDistribution(mpi.COMM_WORLD)
#     except:
#         success = 0
#     success = mpi.COMM_WORLD.allreduce(success, op=mpi.MAX)
#     self.assertEqual(success, 1, "An exception was thrown during the construction of object on COMM_WORLD.")
#
# def test_parallel_constructor_with_vals(self):
#     states = np.array([[0,0], [0,1], [1,2], [3,4]], dtype=int)
#     probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
#     success = 1
#     try:
#         dist = pac.DiscreteDistribution(mpi.COMM_SELF, states, probs)
#     except:
#         success = 0
#     self.assertEqual(success, 1, "An exception was raised during the construction of object on COMM_WORLD with values.")


if __name__ == '__main__':
    unittest.main()
