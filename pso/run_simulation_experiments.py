import math
import numpy as np
from pso import PSO
from plot import plot2d


def sphere(p):
    return p[0]**2 + p[1]**2


def rosenbrock(p):
    return (1 - p[0])**2 + 100*(p[1] - p[0]**2)**2


def rastrigin(p):
    num_dimensions = 2
    A = 10

    sum = 0
    for i in range(num_dimensions):
        sum += p[i]**2 - A*np.cos(2*math.pi*p[i])

    return A*num_dimensions + sum


def griewank(p):
    num_dimensions = 2
    A = 4000

    sum = 0
    for i in range(num_dimensions):
        sum += p[i] ** 2

    prod = 1
    for i in range(num_dimensions):
        prod *= np.cos(p[i]/math.sqrt(i+1))

    return 1 + 1/A*sum - prod


def print_table_data(table_data):
    for experiment in table_data:
        print("\t\t|\t\t".join(experiment))


def main():

    iterations = 100
    simulation_runs = 5

    num_particles = 36
    inertia_weight = 1.5
    constriction_factor = 0.5

    pso = PSO(max_iter=iterations,
              simulation_runs=simulation_runs,
              num_particles=num_particles,
              inertia_weight=inertia_weight,
              constriction_factor=constriction_factor)

    topology_list = ['full-graph',
                     'ring',
                     '4-neighbours']
    pso_variant_list = ['main',
                        'inertia-weight',
                        'constriction-factor']
    influence_model_list = [(1.03, 2.07),
                            (2.10, 2.20),
                            (1.0, 2.0)]
    objective_function_list = [(sphere, 0, (-100, 100), (0, 0)),
                               (rosenbrock, 0, (-100, 100), (1, 1)),
                               (rastrigin, 0, (-5.12, 5.12), (0, 0)),
                               (griewank, 0, (-100, 100), (0, 0))]

    table_data = []

    for objective_function in objective_function_list:
        for topology in topology_list:
            for pso_variant in pso_variant_list:
                for influence_model in influence_model_list:

                        solution, objective_function_value, error_value = pso.run(topology,
                                                                                  pso_variant,
                                                                                  influence_model,
                                                                                  objective_function)
                        table_data += [[objective_function[0].__name__,
                                        pso_variant,
                                        topology,
                                        str(influence_model),
                                        str(solution),
                                        objective_function_value,
                                        error_value]]

    print_table_data(table_data)


if __name__ == "__main__":
    # main()
    plot2d(rosenbrock, [-0.7, 0.8])
    plot2d(sphere, [-5, 5])
    plot2d(rastrigin, [-5, 5])
    plot2d(griewank, [-10, 10])
