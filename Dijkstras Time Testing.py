import heapq

import openpyxl

import random
import timeit

import networkx as nx


import numpy as np
from typing import List, Tuple
from heapq import heappop, heappush


def random_connected_graph(n, x, w):

    # Generate a random positive connected graph
    G = nx.fast_gnp_random_graph(n, x)
    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(n, x)
    # Assign random weights to the edges
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(w[0], w[1])
    # Create an adjacency matrix with the weights
    adj_matrix = np.zeros((n, n))
    for (u, v, data) in G.edges(data=True):
        adj_matrix[u][v] = data['weight']
        adj_matrix[v][u] = data['weight']
    return adj_matrix

def dijkstra(dist_matrix, start_node, end_node):
    n = len(dist_matrix)  # number of nodes
    pq = [(0, start_node)]  # priority queue with start node
    visited = {}  # visited nodes and their distances
    parent = {}  # parent node of each visited node

    # initialize distances to infinity except for start node
    for i in range(n):
        visited[i] = float('inf')
    visited[start_node] = 0

    while pq:
        # get node with smallest distance from start node
        (dist, curr_node) = heapq.heappop(pq)

        # check if we've reached the end node
        if curr_node == end_node:
            path = []
            node = curr_node
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(start_node)
            path.reverse()
            return (path, dist)

        # update distances of neighbors
        for neighbor, weight in enumerate(dist_matrix[curr_node]):
            if weight > 0:  # if edge exists
                new_dist = dist + weight
                if new_dist < visited[neighbor]:
                    visited[neighbor] = new_dist
                    parent[neighbor] = curr_node
                    heapq.heappush(pq, (new_dist, neighbor))

    # end node not reachable from start node
    return (None, None)

# Define the range of graph sizes
graph_sizes = range(100, 6000, 100)

# Initialize empty lists to store the average time it takes to execute each algorithm
avg_dijkstra_times = []



    # Iterate over the graph sizes and calculate the time it takes to execute each algorithm

t = 0
for n in graph_sizes:

    dijkstra_times = []
    print("graph size:", n)
    ll = 0
    for run in range(50):
        print(run)
        rn= random.randint(0, n-1)  # generate a random integer between 1 and n
        x = 0.04  # Specify the probability of two nodes being connected
        w = (1, 10)
        adj_matrix = random_connected_graph(n, x, w)

        start_time2 = timeit.default_timer()
        z = dijkstra(adj_matrix,0, rn)
        dijkstra_time = timeit.default_timer() - start_time2
        dijkstra_times.insert(ll, dijkstra_time)
        ll = ll + 1
    d = np.mean(dijkstra_times)
    avg_dijkstra_times.insert(t, d)
    t = t + 1

# Print the final results
print("Final results:")
print(f"Average Dijkstra's times: {avg_dijkstra_times}")
i = 0
workbook = openpyxl.Workbook()
worksheet = workbook.active
for i in range(len(avg_dijkstra_times)):
    worksheet.cell(row=i+1, column=2, value=avg_dijkstra_times[i])
workbook.save("Dijkstras time testing prob 0.04.xlsx")
