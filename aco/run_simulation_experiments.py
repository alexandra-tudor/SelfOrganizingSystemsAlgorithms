from aco import ACO
from utils import create_graph, print_graph, print_table_data
from tsp import TSP


def main():

    simulation_runs = 5
    num_ants = 5
    q = 10000
    nc_max = 100

    aco = ACO(simulation_runs=simulation_runs,
              num_ants=num_ants,
              pheromone_quantity=q,
              num_colonies=nc_max)
    tsp = TSP()

    graph_nodes_list = [9, 10, 11]
    alpha_list = [1, 3, 7]
    beta_list = [1, 3, 7]
    ro_list = [0.0, 0.5, 0.9, 1.0]
    table_data = []

    for graph_nodes in graph_nodes_list:
        cost_graph = create_graph(graph_nodes)
        # print_graph(cost_graph)
        total_tsp_cost, tsp_solution = tsp.run(cost_graph)
        for alpha in alpha_list:
            for beta in beta_list:
                for ro in ro_list:
                    print("graph_nodes = {3}, alpha = {0}, beta = {1}, ro = {2}".format(alpha, beta, ro, graph_nodes))
                    solution, total_aco_cost = aco.run(cost_graph, alpha, beta, ro)
                    error_value = abs(total_tsp_cost - total_aco_cost)
                    table_data += [[str(graph_nodes),
                                    str(alpha),
                                    str(beta),
                                    str(ro),
                                    str(total_aco_cost),
                                    str(total_tsp_cost),
                                    str(error_value)]]
                    pass

    print_table_data(table_data)


if __name__ == "__main__":
    main()
