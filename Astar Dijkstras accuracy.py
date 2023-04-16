import random

import numpy as np
import pandas as pd
import networkx as nx
import heapq




def random_connected_graph(n, x, w):
    # Generate a random positive connected graph
    G = nx.fast_gnp_random_graph(n, x)
    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(n, x)
        print("not connected ")
    # Assign random weights to the edges
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(w[0], w[1])
    # Create an adjacency matrix with the weights
    adj_matrix = np.zeros((n, n))
    for (u, v, data) in G.edges(data=True):
        adj_matrix[u][v] = data['weight']
        adj_matrix[v][u] = data['weight']
    return adj_matrix


def dijkstra(graph, start):
    distances = [float("inf") for _ in range(len(graph))]
    visited = [False for _ in range(len(graph))]
    distances[start] = 0
    while True:
        shortest_distance = float("inf")
        shortest_index = -1
        for i in range(len(graph)):
            if distances[i] < shortest_distance and not visited[i]:
                shortest_distance = distances[i]
                shortest_index = i
        if shortest_index == -1:
            return distances
        for i in range(len(graph[shortest_index])):
            if graph[shortest_index][i] != 0 and distances[i] > distances[shortest_index] + graph[shortest_index][i]:
                distances[i] = distances[shortest_index] + graph[shortest_index][i]
        visited[shortest_index] = True


def manhattan_dist(a, b, coords):
    """Calculate Manhattan distance between nodes a and b."""
    x1, y1 = coords[a]
    x2, y2 = coords[b]
    return abs(x1 - x2) + abs(y1 - y2)

def astar(start, goal, adj_matrix):
    """Find the shortest path from start to goal using A* search algorithm."""
    G = nx.Graph()
    n = len(adj_matrix)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i, n):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i][j])

    coords = nx.spring_layout(G)

    queue = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while queue:
        _, current = heapq.heappop(queue)

        if current == goal:
            path = [current]
            total_weight = 0
            while current != start:
                prev, weight = came_from[current]
                path.append(prev)
                total_weight += weight
                current = prev
            path.reverse()
            return path, total_weight

        for neighbor, weight_dict in G[current].items():
            weight = weight_dict['weight']
            if weight == 0:
                continue

            new_cost = cost_so_far[current] + weight
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan_dist(neighbor, goal, coords)
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = (current, weight)

    return None, None
Astar_return = []

diff = []
w = (1,10)
numb_nodes = 490



all_nodes = range(0, numb_nodes, 1)
avg = range(0, 10, 1)
k = 0
for i in avg:
    t = 0
    Astar_return = []
    d = []
    print("avg:", i)
    adj_matrix = random_connected_graph(numb_nodes, 0.05, w)
    d = dijkstra(adj_matrix, 0)
    print(d)

    for node in all_nodes:
        path, total_weight = astar(0, node, adj_matrix)
        Astar_return.insert(t,  round(total_weight, 2))
        t = t+1
    array_diff = np.not_equal(d, Astar_return)
    num_diff = np.sum(array_diff)
    print("how many differences")
    print(num_diff)
    diff.insert(k, num_diff)
    k = k + 1
print(diff)
print(sum(diff))

