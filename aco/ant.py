import random


class Ant:
    def __init__(self, node, cost_graph, pheromone_graph, alpha, beta, ro, pheromone_quantity):
        self.cost_graph = cost_graph
        self.alpha = alpha
        self.beta = beta
        self.ro = ro
        self.pheromone_quantity = pheromone_quantity
        self.pheromone_graph = pheromone_graph

        self.current_city = node
        self.visited_cities = [node]
        self.tour_length = 0

    def choose_next_city(self):
        values = []
        sum = 0
        max = [0, 0]
        # compute formula's value for each next city
        for next_city in range(len(self.cost_graph)):
            if self.cost_graph[self.current_city][next_city] == float('inf'):
                continue
            if self.current_city == next_city:
                value = 0
            else:
                value = (self.pheromone_graph[self.current_city][next_city] ** self.alpha) * ((1./self.cost_graph[self.current_city][next_city]) ** self.beta)
            values += [[value, next_city]]
            sum += value

        # compute formula's average value for each next city and the maximum value
        for i in range(len(values)):
            if values[i][1] in self.visited_cities:
                continue
            if sum != 0:
                values[i][0] /= sum
            if max[0] <= values[i][0]:
                max[0] = values[i][0]
                max[1] = values[i][1]

        self.visited_cities += [max[1]]
        self.current_city = max[1]
        self.tour_length += self.cost_graph[self.visited_cities[-2]][self.visited_cities[-1]]

    def return_to_initial_city(self):
        self.current_city = self.visited_cities[0]
        self.visited_cities = [self.visited_cities[0]]
        self.tour_length = 0

    def update_pheromone_level(self):
        start_city = self.visited_cities[0]
        for end_city in self.visited_cities[1:]:
            self.pheromone_graph[start_city][end_city] = (1.0 - self.ro) * self.pheromone_graph[start_city][end_city]
            self.pheromone_graph[end_city][start_city] = (1.0 - self.ro) * self.pheromone_graph[end_city][start_city]
            delta_pheromone = float(self.pheromone_quantity[0]) / self.tour_length
            self.pheromone_graph[start_city][end_city] += delta_pheromone
            self.pheromone_graph[end_city][start_city] += delta_pheromone
            start_city = end_city
