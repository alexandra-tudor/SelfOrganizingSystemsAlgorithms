from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import logging
import numpy as np
import math
from scipy.spatial import cKDTree
from past.builtins import xrange

# Import from package
from .base import DiscreteSwarmBase


def cli_print(message, verbosity, threshold, logger):
    """Helper function to print console output
    Parameters
    ----------
    message : str
        the message to be printed into the console
    verbosity : int
        verbosity setting of the user
    threshold : int
        threshold for printing
    logger : logging.getLogger
        logger instance
    """
    if verbosity >= threshold:
        logger.info(message)
    else:
        pass


def end_report(cost, pos, verbosity, logger):
    template = ('================================\n'
                'Optimization finished!\n'
                'Final cost: {:06.4f}\n'
                'Best value: {}\n').format(cost, pos)
    if verbosity >= 1:
        logger.info(template)


class BinaryMutatedPSO(DiscreteSwarmBase):

    def assertions(self):
        # Check clamp settings
        if self.velocity_clamp is not None:
            if not isinstance(self.velocity_clamp, tuple):
                raise TypeError('Parameter `velocity_clamp` must be a tuple')
            if not len(self.velocity_clamp) == 2:
                raise IndexError('Parameter `velocity_clamp` must be of '
                                 'size 2')
            if not self.velocity_clamp[0] < self.velocity_clamp[1]:
                raise ValueError('Make sure that velocity_clamp is in the '
                                 'form (v_min, v_max)')

        # Required keys in options argument
        if not all(key in self.options for key in ('c1', 'c2', 'w')):
            raise KeyError('Missing either c1, c2, or w in options')

        if not all(key in self.options for key in ('k', 'p')):
            raise KeyError('Missing either k or p in options')
        if not 0 <= self.k <= self.n_particles:
            raise ValueError('No. of neighbors must be between 0 and no. of'
                             'particles.')
        if self.p not in [1, 2]:
            raise ValueError('p-value should either be 1 (for L1/Minkowski)'
                             'or 2 (for L2/Euclidean).')

    def __init__(self, n_particles, dimensions, options, velocity_clamp=None):
        """Initializes the swarm.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        options : dict with keys :code:`{'c1', 'c2', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        binary = True
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options['k'], options['p']
        # Initialize parent class
        super(BinaryMutatedPSO, self).__init__(n_particles, dimensions, binary,
                                        options, velocity_clamp)
        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()

    def optimize(self, objective_func, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        for i in xrange(iters):
            # Compute cost for current position and personal best
            current_cost = objective_func(self.pos)
            pbest_cost = objective_func(self.personal_best_pos)

            # Update personal bests if the current position is better
            # Create a 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)

            # Create a 2-D mask to update positions
            _m = np.repeat(m[:, np.newaxis], self.dimensions, axis=1)
            self.personal_best_pos = np.where(~_m, self.personal_best_pos,
                                              self.pos)

            # Obtain the indices of the best position for each
            # neighbour-space, and get the local best cost and
            # local best positions from it.
            nmin_idx = self._get_neighbors(pbest_cost)
            self.best_cost = pbest_cost[nmin_idx]
            self.best_pos = self.personal_best_pos[nmin_idx]

            # print ("nmin_idx {0}".format(nmin_idx))
            # print("pbest_cost {0}".format(pbest_cost))
            # print("best_cost {0}".format(self.best_cost))

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                          (i+1, iters, np.min(self.best_cost)), verbose, 2,
                          logger=self.logger)

            # Save to history
            hist = self.ToHistory(
                best_cost=np.min(self.best_cost),
                mean_pbest_cost=np.mean(pbest_cost),
                mean_neighbor_cost=np.mean(self.best_cost),
                position=self.pos,
                velocity=self.velocity
            )
            self._populate_history(hist)

            # Perform position velocity update
            self._update_velocity()
            self._update_position()
            self._adaptive_uniform_mutation(i, iters)


        # Obtain the final best_cost and the final best_position
        final_best_cost_arg = np.argmin(self.best_cost)
        final_best_cost = np.min(self.best_cost)
        final_best_pos = self.best_pos[final_best_cost_arg]

        end_report(final_best_cost, final_best_pos, verbose,
                   logger=self.logger)
        return final_best_cost, final_best_pos

    def _get_neighbors(self, pbest_cost):
        # Use cKDTree to get the indices of the nearest neighbors
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, p=self.p, k=self.k)

        # print ("idx {0}".format(idx))

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if self.k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = pbest_cost[idx][:, np.newaxis].argmin(axis=1)
        else:
            idx_min = pbest_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]

        return best_neighbor

    def _update_velocity(self):
        # Define the hyper parameters from options dictionary
        c1, c2, w = self.options['c1'], self.options['c2'], self.options['w']

        # Compute for cognitive and social terms
        r1 = np.random.uniform(0, 1, self.swarm_size)
        r2 = np.random.uniform(0, 1, self.swarm_size)
        cognitive = (c1 * r1 * (self.personal_best_pos - self.pos))
        social = (c2 * r2 * (self.best_pos - self.pos))
        temp_velocity = (w * self.velocity) + cognitive + social

        # Create a mask to clamp the velocities
        if self.velocity_clamp is not None:
            # Create a mask depending on the set boundaries
            min_velocity, max_velocity = self.velocity_clamp[0], self.velocity_clamp[1]
            _b = np.logical_and(temp_velocity >= min_velocity, temp_velocity <= max_velocity)
            # Use the mask to finally clamp the velocities
            self.velocity = np.where(~_b, self.velocity, temp_velocity)
        else:
            self.velocity = temp_velocity

    def _update_position(self):
        self.pos = (np.random.random_sample(size=self.swarm_size) < self._sigmoid(self.velocity)) * 1

    def _adaptive_uniform_mutation(self, crtiter, alliters):
        pm = 0.5 * math.pow(math.e, -10 * crtiter / alliters) + 0.01
        for i in range(self.n_particles):
            if pm > np.random.uniform(0,1):
                K = int(max(1, math.ceil(self.dimensions * pm)))
                # print ("K {0}".format(K))
                S = np.random.choice(range(self.dimensions), K)
                # print("S {0}".format(S))

                for k in range(K):
                    self.pos[i, S[k]] = np.random.randint(2, size=1)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(x))