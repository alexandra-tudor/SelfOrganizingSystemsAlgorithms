from particle import Particle
import math


class PSO:
    def __init__(self,
                 max_iter,
                 simulation_runs,
                 num_particles,
                 num_output_features,
                 num_dimensions,
                 inertia_weight,
                 constriction_factor):

        self.max_iter = max_iter
        self.simulation_runs = simulation_runs
        self.num_particles = num_particles
        self.num_output_features = num_output_features
        self.num_dimensions = num_dimensions
        self.inertia_weight = inertia_weight
        self.constriction_factor = constriction_factor

    def create_swarm(self, topology, space_bounds):
        swarm = []

        if topology == 'full-graph':
            for p in range(self.num_particles):
                particle = Particle(self.num_dimensions, space_bounds)
                for p in range(self.num_particles):
                    particle.add_neighbour(p)
                swarm += [particle]
            # print(list(map(lambda x: x.neighbours, swarm)))

        elif topology == 'ring':
            particle = Particle(self.num_dimensions,space_bounds)
            particle.add_neighbour(1)
            particle.add_neighbour(self.num_particles-1)
            swarm += [particle]

            for p in range(1, self.num_particles-1):
                particle = Particle(self.num_dimensions, space_bounds)
                particle.add_neighbour(p-1)
                particle.add_neighbour(p+1)
                swarm += [particle]

            particle = Particle(self.num_dimensions, space_bounds)
            particle.add_neighbour(0)
            particle.add_neighbour(self.num_particles-2)
            swarm += [particle]
            # print(list(map(lambda x: x.neighbours, swarm)))

        elif topology == '4-neighbours':
            matrix = []
            matrix_size = int(math.sqrt(self.num_particles))
            p = 0
            for idx1 in range(matrix_size):
                neighbours = []
                for idx2 in range(matrix_size):
                    neighbours += [p]
                    p += 1
                matrix += [neighbours]

            for idx1 in range(matrix_size):
                for idx2 in range(matrix_size):
                    particle = Particle(self.num_dimensions, space_bounds)

                    if idx2 > 0:
                        particle.add_neighbour(matrix[idx1][idx2-1])
                    else:
                        particle.add_neighbour(-1)

                    if idx1 > 0:
                        particle.add_neighbour(matrix[idx1-1][idx2])
                    else:
                        particle.add_neighbour(-1)

                    if idx2 < matrix_size-1:
                        particle.add_neighbour(matrix[idx1][idx2+1])
                    else:
                        particle.add_neighbour(-1)

                    if idx1 < matrix_size-1:
                        particle.add_neighbour(matrix[idx1+1][idx2])
                    else:
                        particle.add_neighbour(-1)

                    swarm += [particle]
            # print(list(map(lambda x: x.neighbours, swarm)))

        return swarm

    def print_swarm(self, swarm):
        p = swarm[10]
        # for p in swarm:
        # print("{0} -> {1}".format(p.pos, p.obj))
        # print("{0} -> {1}".format(p.pos_best, p.obj_best))
        print("{0} -> {1}".format(p.pos_best_local, p.obj_best_local))
        # print(p.velocity)

        print()

    def run(self, topology, pso_variant, influence_model, objective_function):
        best_solution = (0, 0)
        best_objective_function_value = (9999, 9999)
        best_error_value = 9999

        for i in range(self.simulation_runs):
            # print("== [{0}] == Running experiment: {1}, {2}, {3}, {4}".format(i, topology, pso_variant, influence_model, objective_function))

            solution = ""
            objective_function_value = ""
            error_value = ""

            # PARTICLE INITIALIZATION
            swarm = self.create_swarm(topology, objective_function[2])

            i = 0
            while i < self.max_iter:

                # print("########    {0}    #######".format(i))
                # self.print_swarm(swarm)

                # EVALUATE EACH PARTICLE ACCORDING TO THE OBJECTIVE FUNCTION
                # IF THE PARTICLE's BEST POSITION IS BETTER THAN ITS PREVIOUS BEST POSITION, UPDATE IT
                for j in range(self.num_particles):
                    swarm[j].evaluate_and_update(objective_function[0])

                # DETERMINE THE BEST PARTICLE ACCORDING TO THE GIVEN TOPOLOGY
                # EACH PARTICLE UPDATES ITS KNOWN LOCAL BEST QUERYING ITS NEIGHBOURS
                for j in range(self.num_particles):
                    for np in swarm[j].neighbours:
                        if np > -1:
                            if swarm[np].obj_best < swarm[j].obj_best_local or swarm[j].obj_best_local == -1:
                                swarm[j].obj_best_local = swarm[np].obj_best
                                swarm[j].pos_best_local = swarm[np].pos_best

                    # WHEN THE CURRENT PARTICLE's OBJ IS THE BEST LOCAL
                    if swarm[j].obj_best < swarm[j].obj_best_local:
                        swarm[j].obj_best_local = swarm[j].obj_best
                        swarm[j].pos_best_local = swarm[j].pos_best

                # UPDATE EACH PARTICLE's VELOCITY ACCORDING TO THE PSO-VARIANT's FORMULA
                for j in range(self.num_particles):
                    weight = 1
                    factor = 1

                    if pso_variant == 'main':
                        weight = 1
                        factor = 1

                    elif pso_variant == 'inertia-weight':
                        weight = self.inertia_weight
                        factor = 1

                    elif pso_variant == 'constriction-factor':
                        weight = 1
                        factor = self.constriction_factor

                    swarm[j].update_velocity(influence_model, inertia_weight=weight, constriction_factor=factor)

                # MOVE PARTICLES TO THEIR NEW POSITIONS
                for j in range(self.num_particles):
                    swarm[j].update_position()

                i += 1

                # get the final best solution
                solution = swarm[0].pos_best
                objective_function_value = swarm[0].obj_best

                for j in range(1, self.num_particles):
                    if swarm[j].obj_best < objective_function_value:
                        solution = swarm[j].pos_best
                        objective_function_value = swarm[j].obj_best
                error_value = abs(objective_function[1] - objective_function_value)

                self.inertia_weight -= 0.01
            # print("Results: {0}, {1}, {2}\n\n".format(str(solution), str(objective_function_value), str(error_value)))

            if error_value < best_error_value:
                best_error_value = error_value
                best_solution = solution
                best_objective_function_value = objective_function_value

        return str(best_solution), str(best_objective_function_value), str(best_error_value)
