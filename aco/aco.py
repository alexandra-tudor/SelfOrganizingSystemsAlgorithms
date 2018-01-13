from copy import deepcopy
from random import randint

from ant import Ant
from utils import create_empty_graph
import matplotlib.pyplot as plt


class ACO:
    def __init__(self,
                 simulation_runs,
                 num_ants,
                 pheromone_quantity,
                 num_colonies):

        self.simulation_runs = simulation_runs
        self.num_ants = num_ants
        self.pheromone_quantity = pheromone_quantity,
        self.num_colonies = num_colonies

    def print_graph(self, graph):
        print(graph)

    def run(self, cost_graph, alpha, beta, ro):
        fig = plt.figure()
        plt.clf()
        plt.title(str(alpha) + "_" + str(beta) + "_" + str(ro))
        best_solution = None
        best_cost = float('inf')

        for simulation in range(self.simulation_runs):
            pheromone_graph = create_empty_graph(len(cost_graph), 0.1)
            solution = []
            best_cost = float('inf')
            cost_list = []
            for nc in range(self.num_colonies):
                # Place each ant in a randomly chosen city
                cost = float('inf')
                ants = []
                for i in range(self.num_ants):
                    choose = randint(0, len(cost_graph)-1)
                    ants += [Ant(int(choose), cost_graph, pheromone_graph, alpha, beta, ro, self.pheromone_quantity)]

                # While there are unvisited cities, choose NextCity (For Each Ant)
                for i in range(len(cost_graph)-1):
                    for ant in ants:
                        ant.choose_next_city()

                # Update pheromone level using the tour cost for each ant
                for ant in ants:
                    ant.update_pheromone_level()

                # memorize the best cost and route
                for ant in ants:
                    if ant.tour_length < cost:
                        cost = ant.tour_length
                        solution = deepcopy(ant.visited_cities)

                if cost < best_cost:
                    best_cost = cost
                    best_solution = deepcopy(solution)

                # Return to the initial cities
                for ant in ants:
                    ant.return_to_initial_city()

                # for i in range(len(pheromone_graph)):
                #     print("   ".join(list(map(lambda x: "{0:.2f}".format(round(x, 2)), pheromone_graph[i]))))
                print("\tTour: {0} Cost: {1} Solution: {2}".format(nc, cost, solution))
                cost_list += [(nc, cost)]

            plt.plot(*zip(*cost_list), label=str(simulation))
            print("Simulation: {0} Cost: {1} Solution: {2}\n".format(simulation, best_cost, best_solution))

        plt.legend(loc='best')
        plt.savefig("solution_" + str(len(cost_graph)) + "_" + str(alpha) + "_" + str(beta) + "_" + str(ro) + ".png")

        return best_solution, best_cost
