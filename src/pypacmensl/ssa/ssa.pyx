# distutils: language = c++
import numpy as np
import mpi4py.MPI as MPI
from numpy.random import random_sample, PCG64, SeedSequence
from math import log
from scipy.stats import poisson

cdef class SSASolver:
    def __cinit__(self, comm = None, seed_seq = None):
        self.prop_t_ = None
        self.prop_x_ = None
        self.stoich_matrix_ = np.empty((0, 0))
        if comm is None:
            self.comm_ = MPI.Comm.COMM_SELF
        else:
            self.comm_ = comm
        if seed_seq is None:
            seq = SeedSequence()
            if comm.Get_rank() == 0:
                ss = [np.random.PCG64(s) for s in seq.spawn(comm.Get_size())]
            else:
                ss = None
            self.bitGen_ = comm.scatter(ss, root=0)
        else:
            self.bitGen_ = PCG64(seed_seq)


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

        cdef:
            double[:] tcoefview = tcoef
            double[:] propsview = props
            int[:,:] xtmpview = xtmp
            double[:] tmpview = tmp_array
            bitgen_t* rng;

        capsule = self.bitGen_.capsule
        rng = <bitgen_t*> PyCapsule_GetPointer(capsule, "BitGenerator")

        outputs_local = np.zeros((num_samples_local, num_species), dtype=np.intc)

        for i in range(0, num_samples_local):
            xtmp[0, :] = x0
            t_now = 0.0
            while t_now < t_final:
                r1 = rng.next_double(rng.state)
                r2 = rng.next_double(rng.state)

                # Compute propensities
                reaction = 0
                asum = 0.0
                self.prop_t_(t_now, tcoef)
                for r in range(0, num_reactions):
                    self.prop_x_(r, xtmp, tmp_array)
                    propsview[r] = tcoefview[r] * tmpview[0]
                    asum = asum + propsview[r]

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
                    tmp = tmp + propsview[reaction]
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

    def SolveTv(self, double t_final, np.ndarray x0, int num_samples = 1, double update_rates = 1.0E-4,
    send_to_root=False):
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
            bitgen_t* rng;

        capsule = self.bitGen_.capsule
        rng = <bitgen_t*> PyCapsule_GetPointer(capsule, "BitGenerator")

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
        props = np.zeros((num_reactions+1,))
        props[-1] = update_rates
        tmp_array = np.zeros((1,))

        cdef:
            double[:] tcoefview = tcoef
            double[:] propsview = props
            int[:,:] xtmpview = xtmp
            double[:] tmpview = tmp_array

        outputs_local = np.zeros((num_samples_local, num_species), dtype=np.intc)

        for i in range(0, num_samples_local):
            xtmp[0, :] = x0
            t_now = 0.0
            while t_now < t_final:
                r1 = rng.next_double(rng.state)
                r2 = rng.next_double(rng.state)

                # Compute propensities
                reaction = 0
                asum = 0.0
                self.prop_t_(t_now, tcoef)
                for r in range(0, num_reactions):
                    self.prop_x_(r, xtmp, tmp_array)
                    propsview[r] = tcoefview[r] * tmpview[0]
                    asum = asum + propsview[r]
                asum += update_rates

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
                    tmp = tmp + propsview[reaction]
                    if tmp >= r2 * asum:
                        break

                if reaction < num_reactions:
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

cdef class PmPdmsrSampler:
    """
    Implement the PM-PDMSR method for sampling the Poisson mean parameter in the Poisson mixture representation for
    the solution of gene expression model.

    Reference:
    Y. T. Lin, N. E. Buchler. "Exact and efficient hybrid Monte Carlo algorithm for accelerated Bayesian inference of
    gene expression models from snapshots of single-cell transcripts". J. Chem. Phys. 151, 024106 (2019).
    """
    def __cinit__(self, comm = None, seed_seq = None):
        self.prop_t_ = None
        self.prop_x_ = None
        self.stoich_matrix_ = np.empty((0,0), dtype=np.intc)
        self.f_transcr_ = None
        self.deg_rates_ = None

        if comm is None:
            self.comm_ = MPI.Comm.COMM_SELF
        else:
            self.comm_ = comm
        if seed_seq is None:
            seq = SeedSequence()
            if comm.Get_rank() == 0:
                ss = [np.random.PCG64(s) for s in seq.spawn(comm.Get_size())]
            else:
                ss = None
            self.bitGen_ = comm.scatter(ss, root=0)
        else:
            self.bitGen_ = PCG64(seed_seq)

    def SetModel(self, np.ndarray stoich_matrix, propensity_t, propensity_x, f_transcr, np.ndarray deg_rates):
        self.prop_t_ = propensity_t
        self.prop_x_ = propensity_x
        self.stoich_matrix_ = np.copy(stoich_matrix)
        self.f_transcr_ = f_transcr
        self.deg_rates_ = np.copy(deg_rates)

    def Solve(self, double t_final, np.ndarray x0, int num_samples = 1, send_to_root=False)->[np.ndarray, np.ndarray]:
        """
        Sample the gene state and Poisson parameters at a user-specified time given the initial gene state.

        Parameters
        ----------
        t_final : double
            Timepoint at which to sample.

        x0 : numpy array
            Initial gene state.

        num_samples : int
            Number of samples.

        send_to_root : bool
            Whether to send all samples to process 0. Default is false.

        Returns
        -------

        gene_states: 2-d numpy array
            Gene state samples, in which gene_states[i, :] is the final state reached by the i-th SSA sample path.

        poisson_states: 1-d numpy array
            Poisson state samples, in which poisson_states[i] is the final solution of the Poisson state ODE
            determined by the i-th SSA sample path.
        """
        cdef:
            # int rank, num_procs, num_samples_local
            int num_gene_states, num_transitions, num_rna_species
            int reaction
            double t_now, r1, r2, tau
            double asum, tmp
            np.ndarray xtmp
            np.ndarray props, tcoef

        rank = self.comm_.Get_rank()
        num_procs = self.comm_.Get_size()

        num_samples_local = (num_samples // num_procs) + (rank < num_samples % num_procs)

        if x0.ndim == 1:
            num_gene_states = len(x0)
        elif x0.ndim == 2:
            num_gene_states = x0.shape[1]
        num_transitions = <int> self.stoich_matrix_.shape[0]
        num_rna_species = len(self.deg_rates_)

        xtmp = np.zeros((1, num_gene_states), dtype=np.intc)
        tcoef = np.ones((num_transitions,))
        props = np.zeros((num_transitions,))
        tmp_array = np.zeros((1,))

        cdef:
            double[:] tcoefview = tcoef
            double[:] propsview = props
            int[:,:] xtmpview = xtmp
            double[:] tmpview = tmp_array
            bitgen_t* rng;

        capsule = self.bitGen_.capsule
        rng = <bitgen_t*> PyCapsule_GetPointer(capsule, "BitGenerator")

        outputs_local = np.zeros((num_samples_local, num_gene_states), dtype=np.intc)
        poisson_states_local = np.zeros((num_samples_local, num_rna_species), dtype=np.double)

        for i in range(0, num_samples_local):
            xtmp[0, :] = x0
            poisson_state_tmp = np.zeros((num_rna_species,), dtype=np.double)
            t_now = 0.0
            while t_now < t_final:
                r1 = rng.next_double(rng.state)
                r2 = rng.next_double(rng.state)

                # Compute propensities
                reaction = 0
                asum = 0.0
                if self.prop_t_ is not None:
                    self.prop_t_(t_now, tcoef)
                for r in range(0, num_transitions):
                    self.prop_x_(r, xtmp, tmp_array)
                    propsview[r] = tcoefview[r] * tmpview[0]
                    asum = asum + propsview[r]

                if asum == 0.0:
                    break
                # Decide time step
                tau = log(1.0 / r1) / asum

                if tau < 0.0:
                    break

                if t_now + tau > t_final:
                    tau = t_final - t_now
                    poisson_state_tmp[:] = self.f_transcr_(xtmp[0, :])/self.deg_rates_ + \
                    (poisson_state_tmp[:] - self.f_transcr_(xtmp[0, :])/self.deg_rates_)*np.exp(-self.deg_rates_*tau)
                    break

                # Determine the reaction that will happen
                tmp = 0.0
                for reaction in range(0, num_transitions):
                    tmp = tmp + propsview[reaction]
                    if tmp >= r2 * asum:
                        break

                # Update Poisson state and gene state (in that order)
                poisson_state_tmp[:] = self.f_transcr_(xtmp[0, :])/self.deg_rates_ + \
                    (poisson_state_tmp[:] - self.f_transcr_(xtmp[0, :])/self.deg_rates_)*np.exp(-self.deg_rates_*tau)
                xtmp[0, :] = xtmp[0, :] + self.stoich_matrix_[reaction, :]

                t_now = t_now + tau

            poisson_states_local[i, :] = poisson_state_tmp[:]
            outputs_local[i, :] = xtmp[0, :]
        if send_to_root:
            send_count = np.zeros((num_procs, 1), dtype=np.intc)
            ns_send = np.array([num_samples_local], dtype=np.intc)
            sc = np.ones((num_procs,), dtype=np.intc)
            dis = np.linspace(0, num_procs - 1, num_procs, dtype=np.intc)
            self.comm_.Gatherv(ns_send, [send_count,
                                         tuple(sc),
                                         tuple(dis), MPI.INT], root=0)

            send_count = send_count * num_gene_states
            displacements = np.zeros((num_procs,), dtype=np.intc)

            if num_procs > 1:
                displacements[1:] = np.cumsum(send_count[0:num_procs - 1])

            if rank == 0:
                poisson_states = np.zeros((num_samples,), dtype=np.double)
                gene_states = np.zeros((num_samples, num_gene_states), dtype=np.intc)
            else:
                poisson_states = np.empty((0,))
                gene_states = np.empty((0,))

            self.comm_.Gatherv(outputs_local, [gene_states, tuple(send_count), tuple(displacements), MPI.INT], root=0)
            self.comm_.Gatherv(poisson_states_local, [poisson_states, tuple(num_rna_species*send_count //
                                                                            num_gene_states),
                                                      tuple(num_rna_species*displacements // num_gene_states), MPI.DOUBLE], root=0)
        else:
            gene_states = outputs_local
            poisson_states = poisson_states_local
        return gene_states, poisson_states

    def SolveTv(self, double t_final, np.ndarray x0, int num_samples = 1, double update_rates = 1.0E-4,
    send_to_root=False):
        """
        Sample the gene state and Poisson parameters at a user-specified time given the initial gene state. This
        method is essentially the same as Solve, but we have a safeguard against timepoints where all the
        time-varying propensities
        are zero (at which SSA may mistakenly assume that no reactions will ever happen for the whole timespan).

        Parameters
        ----------
        t_final : double
            Timepoint at which to sample.

        x0 : numpy array
            Initial gene state.

        num_samples : int
            Number of samples.

        send_to_root : bool
            Whether to send all samples to process 0. Default is false.

        update_rates: double
            A small number to safeguard against timepoints where all propensities are zero. In particular,
            we add a "null" reaction that does nothing, but has a small rate, so that when all the usual propensities are zero,
            the SSA will only jump a small step ahead rather than to the final endpoint.

        Returns
        -------

        gene_states: 2-d numpy array
            Gene state samples, in which gene_states[i, :] is the final state reached by the i-th SSA sample path.

        poisson_states: 1-d numpy array
            Poisson state samples, in which poisson_states[i] is the final solution of the Poisson state ODE
            determined by the i-th SSA sample path.
        """

        cdef:
            # int rank, num_procs, num_samples_local
            int num_species, num_reactions, num_rna_species
            int reaction
            double t_now, r1, r2, tau
            double asum, tmp
            np.ndarray xtmp
            np.ndarray props, tcoef
            bitgen_t* rng;

        capsule = self.bitGen_.capsule
        rng = <bitgen_t*> PyCapsule_GetPointer(capsule, "BitGenerator")

        rank = self.comm_.Get_rank()
        num_procs = self.comm_.Get_size()

        num_samples_local = (num_samples // num_procs) + (rank < num_samples % num_procs)

        if x0.ndim == 1:
            num_species = len(x0)
        elif x0.ndim == 2:
            num_species = x0.shape[1]
        num_reactions = self.stoich_matrix_.shape[0]
        num_rna_species = len(self.deg_rates_)

        xtmp = np.zeros((1, num_species), dtype=np.intc)
        tcoef = np.ones((num_reactions,))
        props = np.zeros((num_reactions+1,))
        props[-1] = update_rates
        tmp_array = np.zeros((1,))

        cdef:
            double[:] tcoefview = tcoef
            double[:] propsview = props
            int[:,:] xtmpview = xtmp
            double[:] tmpview = tmp_array

        outputs_local = np.zeros((num_samples_local, num_species), dtype=np.intc)
        poisson_states_local = np.zeros((num_samples_local, num_rna_species), dtype=np.double)

        for i in range(0, num_samples_local):
            poisson_state_tmp = np.zeros((num_rna_species,), dtype=np.double)
            xtmp[0, :] = x0
            t_now = 0.0
            while t_now < t_final:
                r1 = rng.next_double(rng.state)
                r2 = rng.next_double(rng.state)

                # Compute propensities
                reaction = 0
                asum = 0.0
                if self.prop_t_ is not None:
                    self.prop_t_(t_now, tcoef)
                for r in range(0, num_reactions):
                    self.prop_x_(r, xtmp, tmp_array)
                    propsview[r] = tcoefview[r] * tmpview[0]
                    asum = asum + propsview[r]
                asum += update_rates

                # Decide time step
                tau = log(1.0 / r1) / asum

                if tau < 0.0:
                    break

                if t_now + tau > t_final:
                    tau = t_final - t_now
                    poisson_state_tmp[:] = self.f_transcr_(xtmp[0, :])/self.deg_rates_ + \
                                           (poisson_state_tmp[:] - self.f_transcr_(xtmp[0, :])/self.deg_rates_)*np.exp(-self.deg_rates_*tau)
                    break

                # Determine the reaction that will happen
                tmp = 0.0
                for reaction in range(0, num_reactions+1):
                    tmp = tmp + propsview[reaction]
                    if tmp >= r2 * asum:
                        break

                poisson_state_tmp[:] = self.f_transcr_(xtmp[0, :])/self.deg_rates_ + \
                                       (poisson_state_tmp[:] - self.f_transcr_(xtmp[0, :])/self.deg_rates_)*np.exp(-self.deg_rates_*tau)
                if reaction < num_reactions:
                    xtmp[0, :] = xtmp[0, :] + self.stoich_matrix_[reaction, :]
                t_now = t_now + tau

            outputs_local[i, :] = xtmp[0, :]
            poisson_states_local[i, :] = poisson_state_tmp[:]
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
                poisson_states = np.zeros((num_samples,), dtype=np.double)
                gene_states = np.zeros((num_samples, num_species), dtype=np.intc)
            else:
                poisson_states = np.empty((0,))
                gene_states = np.empty((0,))

            self.comm_.Gatherv(outputs_local, [gene_states, tuple(send_count), tuple(displacements), MPI.INT], root=0)
            self.comm_.Gatherv(poisson_states_local, [poisson_states, tuple(num_rna_species*send_count//num_species),
                                                      tuple(num_rna_species*displacements//num_species), MPI.DOUBLE],
                               root=0)
        else:
            gene_states = outputs_local
            poisson_states = poisson_states_local
        return gene_states, poisson_states

    def compute_loglike(self, poisson_parameters: np.ndarray, observations: np.ndarray)->float:
        """
        Compute the loglikelihood of observing a dataset of mRNA molecular counts from the PMPDMSR samples.

        Parameters
        ----------
        poisson_parameters : numpy array

        observations : numpy array
            Single-cell measurements. Each row is one measurement. The number of columns must equal the number of RNA species measured.
            All processes must pass the same set of observations.

        Returns
        -------

        Log-likelihood of the dataset.

        Notes
        -----

        All processes must pass the same set of observations.
        """
        comm = self.comm_

        nsamples_local = poisson_parameters.shape[0]
        nsamples = comm.allreduce(nsamples_local, MPI.SUM)

        num_observations = observations.shape[0]
        num_rnas = observations.shape[1]

        x_eval_points = []
        p_marginal_evals = []
        for i in range(0, num_rnas):
            unique_obs, unique_inverse = np.unique(observations[:, i], return_inverse=True)
            x_eval_points.append({
                    "eval_points": unique_obs,
                    "inverse": unique_inverse
            })
            p_marginal_evals.append(np.zeros((unique_obs.shape[0]), dtype=float))


        p_local = np.zeros((num_observations,), dtype=float)
        p_loc_term = np.ones((num_observations,))
        for j in range(0, nsamples_local):
            p_loc_term[:] = 1.0
            for ispecies in range(0, num_rnas):
                p_marginal_evals[ispecies] = poisson.pmf(x_eval_points[ispecies]["eval_points"],
                                                         mu=poisson_parameters[j, ispecies])
                p_loc_term *= p_marginal_evals[ispecies][x_eval_points[ispecies]["inverse"]]
            p_local += p_loc_term

        p_global = comm.allreduce(p_local, MPI.SUM)
        p_global = (1.0 / nsamples) * p_global
        p_global[p_global < 1E-16] = 1E-16

        ll = float(np.sum(np.log(p_global)))
        return ll




