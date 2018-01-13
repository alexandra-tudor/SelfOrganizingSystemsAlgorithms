import random

import math


class Particle:
    """
    space_dimensions - the number of features (integer)
    space_bounds - the ranges for all dimensions (vector of integer)
    """
    def __init__(self, space_dimensions, space_bounds):
        self.pos = []
        self.pos_best = []
        self.pos_best_local = []

        self.obj = -1
        self.obj_best = -1
        self.obj_best_local = -1

        self.neighbours = []
        self.velocity = []
        self.num_dimensions = space_dimensions
        self.space_bounds = space_bounds

        for i in range(0, self.num_dimensions):
            self.velocity.append(random.uniform(-1, 1))
            self.pos.append(random.uniform(self.space_bounds[0][0], self.space_bounds[0][1]))

    def evaluate_and_update(self, cost_func):
        self.obj = cost_func(self.num_dimensions, self.pos)

        if self.obj < self.obj_best or self.obj_best == -1:
            self.pos_best = self.pos
            self.obj_best = self.obj

    def update_velocity(self, influence_model, inertia_weight=1, constriction_factor=1):
        fi_1 = influence_model[0]
        fi_2 = influence_model[1]

        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = fi_1 * r1 * (self.pos_best[i] - self.pos[i])
            social_velocity = fi_2 * r2 * (self.pos_best_local[i] - self.pos[i])

            self.velocity[i] = constriction_factor * (inertia_weight * self.velocity[i] + cognitive_velocity + social_velocity)

    def update_position(self):
        for i in range(0, self.num_dimensions):
            self.pos[i] = self.pos[i] + self.velocity[i]

            # adjust maximum position if necessary
            if self.pos[i] > self.space_bounds[i][1]:
                self.pos[i] = self.space_bounds[i][1]

            # adjust minimum position if necessary
            if self.pos[i] < self.space_bounds[i][0]:
                self.pos[i] = self.space_bounds[i][0]

    def add_neighbour(self, neighbour):
        self.neighbours += [neighbour]

    def adaptive_uniform_mutation(self, iteration):
        pm = 0.5 * math.pow(math.e, -10*iteration/self.num_dimensions) + 0.01

