from copy import deepcopy


class TSP:
    def __init__(self):
        self.best_solution = None
        self.best_cost = float('inf')

    def run(self, graph):
        self.best_cost = float('inf')
        def compute_cost(path):
            total_cost = 0
            start_node = path[0]
            for node in path[1:]:
                total_cost += graph[start_node][node]
                start_node = node
            return total_cost

        def add_nodes(solution):
            if len(solution) == len(graph):
                cost = compute_cost(solution)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = deepcopy(solution)
            else:
                for n in range(len(graph)):
                    if n not in solution:
                        add_nodes(deepcopy(solution) + [n])

            solution = solution[:-1]

        solution = []
        add_nodes(solution)

        print("TSP Cost: {0} Solution: {1}".format(self.best_cost, self.best_solution))

        return self.best_cost, self.best_solution

