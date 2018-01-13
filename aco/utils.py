from random import randint


def create_empty_graph(num_nodes, value=0):
    graph = []

    for n in range(num_nodes):
        next_nodes = []
        for n in range(num_nodes):
            next_nodes += [value]
        graph += [next_nodes]

    return graph


def create_graph(num_nodes):
    template = [
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ]
    graph = create_empty_graph(num_nodes)

    for node_start in range(num_nodes):
        for node_end in range(node_start+1, num_nodes):
            cost = randint(1, 1000)
            graph[node_start][node_end] = cost
            graph[node_end][node_start] = cost

    for i in range(num_nodes):
        for j in range(num_nodes):
            if template[i][j] == 0:
                graph[i][j] = float('inf')
    return graph


def print_graph(graph):
    for i in range(len(graph)):
        print(graph[i])


def print_table_data(table_data):
    for experiment in table_data:
        print("|".join(experiment))

