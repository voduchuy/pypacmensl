# distutils: language = c++
import numpy as np
from numpy.random import random_sample
from math import log, exp
import mpi4py.MPI as MPI

cdef class SSASolver:
    def __cinit__(self, comm = None):
        self.prop_t_ = None
        self.prop_x_ = None
        self.stoich_matrix_ = np.empty((0, 0))
        if comm is None:
            self.comm_ = MPI.Comm.COMM_SELF.Dup()
        else:
            self.comm_ = comm.Dup()

    def SetModel(self, np.ndarray stoich_matrix, propensity_t, propensity_x):
        self.prop_t_ = propensity_t
        self.prop_x_ = propensity_x
        self.stoich_matrix_ = stoich_matrix

    def Solve(self, double t_final, np.ndarray x0, int num_samples = 1, send_to_root=False):
        """
        X = Solve(self, double t_final, np.ndarray x0, int num_samples = 1, send_to_root=False)
        :return X : numpy array of sampled states at time t_final, each row is a state.
        """
        cdef:
            # int rank, num_procs, num_samples_local
            int num_species, num_reactions
            int reaction
            double t_now, r1, r2, tau
            double asum, tmp
            np.ndarray xtmp
            np.ndarray props, tcoef

        rank = self.comm_.Get_rank()
        num_procs = self.comm_.Get_size()

        num_samples_local = (num_samples // num_procs) + (rank < num_samples % num_procs)

        if x0.ndim == 1:
            num_species = len(x0)
        elif x0.ndim == 2:
            num_species = x0.shape[1]
        num_reactions = self.stoich_matrix_.shape[0]

        xtmp = np.zeros((1, num_species), dtype=np.intc)
        tcoef = np.zeros((num_reactions,))
        props = np.zeros((num_reactions,))
        tmp_array = np.zeros((1,))

        outputs_local = np.zeros((num_samples_local, num_species), dtype=np.intc)

        for i in range(0, num_samples_local):
            xtmp[0, :] = x0
            t_now = 0.0
            while t_now < t_final:
                r1 = random_sample()
                r2 = random_sample()

                # Compute propensities
                reaction = 0
                asum = 0.0
                self.prop_t_(t_now, tcoef)
                for r in range(0, num_reactions):
                    self.prop_x_(r, xtmp, tmp_array)
                    props[r] = tcoef[r] * tmp_array[0]
                    asum = asum + props[r]

                if asum == 0.0:
                    break
                # Decide time step
                tau = log(1.0 / r1) / asum

                if tau < 0.0:
                    break

                if t_now + tau > t_final:
                    break

                # Determine the reaction that will happen
                tmp = 0.0
                for reaction in range(0, num_reactions):
                    tmp = tmp + props[reaction]
                    if tmp >= r2 * asum:
                        break

                xtmp[0, :] = xtmp[0, :] + self.stoich_matrix_[reaction, :]
                t_now = t_now + tau
            outputs_local[i, :] = xtmp[0, :]
        if send_to_root:
            send_count = np.zeros((num_procs, 1), dtype=np.intc)
            ns_send = np.array([num_samples_local], dtype=np.intc)
            sc = np.ones((num_procs,), dtype=np.intc)
            dis = np.linspace(0, num_procs - 1, num_procs, dtype=np.intc)
            self.comm_.Gatherv(ns_send, [send_count,
                                         tuple(sc),
                                         tuple(dis), MPI.INT], root=0)

            send_count = send_count*num_species
            displacements = np.zeros((num_procs,), dtype=np.intc)

            if num_procs > 1:
                displacements[1:] = np.cumsum(send_count[0:num_procs - 1])

            if rank == 0:
                outputs = np.zeros((num_samples, num_species), dtype=np.intc)
            else:
                outputs = np.empty((0,))

            self.comm_.Gatherv(outputs_local, [outputs, tuple(send_count), tuple(displacements), MPI.INT], root=0)
        else:
            outputs = outputs_local
        return outputs

cdef class SSATrajectory:
    def __cinit__(self):
        self.prop_t_ = None
        self.prop_x_ = None
        self.stoich_matrix_ = np.empty((0, 0))

    def SetModel(self, np.ndarray stoich_matrix, propensity_t, propensity_x):
            self.prop_t_ = propensity_t
            self.prop_x_ = propensity_x
            self.stoich_matrix_ = stoich_matrix

    def Solve(self, np.ndarray tspan, np.ndarray x0):
        """
        [X, T] = Solve(self, double t_final, np.ndarray x0, int num_samples = 1, send_to_root=False)
        :return X : numpy array of sampled states at time t_final, each row is a state.
        """
        cdef:
            # int rank, num_procs, num_samples_local
            int num_species, num_reactions, num_steps
            int reaction
            double t_now, r1, r2, tau, t_final
            double asum, tmp
            np.ndarray xtmp
            np.ndarray props, tcoef

        if x0.ndim == 1:
            num_species = len(x0)
        elif x0.ndim == 2:
            num_species = x0.shape[1]
        num_reactions = self.stoich_matrix_.shape[0]

        t_final = tspan[-1]
        xtmp = np.zeros((1, num_species), dtype=np.intc)
        tcoef = np.zeros((num_reactions,))
        props = np.zeros((num_reactions,))
        tmp_array = np.zeros((1,))
        num_steps = len(tspan)

        X = np.zeros((num_steps, num_species), dtype=np.intc)

        X[0, :] = x0
        t_now = 0.0
        xtmp = np.copy(x0)
        while t_now < t_final:
            r1 = random_sample()
            r2 = random_sample()

            # Compute propensities
            reaction = 0
            asum = 0.0
            self.prop_t_(t_now, tcoef)
            for r in range(0, num_reactions):
                self.prop_x_(r, xtmp, tmp_array)
                props[r] = tcoef[r] * tmp_array[0]
                asum = asum + props[r]

            # Decide time step
            tau = log(1.0 / r1) / asum

            if tau < 0.0:
                break

            if t_now + tau > t_final:
                break

            # Determine the reaction that will happen
            tmp = 0.0
            for reaction in range(0, num_reactions):
                tmp = tmp + props[reaction]
                if tmp >= r2 * asum:
                    break

            X[0, :] = xtmp[:] + self.stoich_matrix_[reaction, :]
            t_now = t_now + tau
        return X
