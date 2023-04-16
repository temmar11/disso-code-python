import heapq
from heapq import heappop, heappush
import random
from typing import List, Tuple, Dict

import networkx as nx
import pandas as pd
import numpy as np




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





##########################################################################################################################################

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



######################################################################################################################################

def a_star_search(adj_matrix: np.ndarray, start_node: int, end_node: int) -> Tuple[List[int], float]:
    num_nodes = len(adj_matrix)
    h_score = np.zeros(num_nodes) # Initialize the h-score (heuristic) of all nodes to 0

    # Initialize the start and goal nodes
    start_node_index = start_node  # adjust to 0-based indexing
    end_node_index = end_node   # adjust to 0-based indexing
    g_score = np.full(num_nodes, np.inf)
    g_score[start_node_index] = 0
    f_score = np.full(num_nodes, np.inf)
    f_score[start_node_index] = euclidean_distance(adj_matrix[start_node_index], adj_matrix[end_node_index])

    # Initialize the set of explored nodes and the priority queue
    explored_set = set()
    priority_queue = [(f_score[start_node_index], start_node_index)]

    # Initialize the came_from dictionary with the start node
    came_from = {start_node_index: -1}

    # Run the A* algorithm
    while priority_queue:
        # Get the node with the lowest f-score from the priority queue
        _, curr_node_index = heappop(priority_queue)

        # If the current node is the goal node, return the path
        if curr_node_index == end_node_index:
            path = reconstruct_path(came_from, end_node_index)
            return path, g_score[end_node_index]

        # Add the current node to the explored set
        explored_set.add(curr_node_index)

        # Explore the neighbors of the current node
        for neighbor_index, weight in enumerate(adj_matrix[curr_node_index]):
            if weight > 0:
                if neighbor_index not in explored_set:
                    tentative_g_score = g_score[curr_node_index] + weight
                    if tentative_g_score < g_score[neighbor_index]:
                        # This path to neighbor is better than any previous one. Record it!
                        came_from[neighbor_index] = curr_node_index
                        g_score[neighbor_index] = tentative_g_score
                        f_score[neighbor_index] = tentative_g_score + euclidean_distance(
                            adj_matrix[neighbor_index], adj_matrix[end_node_index]) + h_score[neighbor_index]
                        if neighbor_index not in priority_queue:
                            heappush(priority_queue, (f_score[neighbor_index], neighbor_index))

    # If the goal node cannot be reached, return an empty path and infinity distance
    return [], np.inf


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


def reconstruct_path(came_from: np.ndarray, end_node_index: int) -> List[int]:
    path = [end_node_index]
    while came_from[path[0]] >= 0:
        path.insert(0, came_from[path[0]])
    return path
######################################################################################################################################

w = (1,10)
adj_matrix = random_connected_graph(500, 0.05, w)


repeat = 501
avgcalc = []
totalsum1 = []
totalsum2 = []
amount = 10
for avg in range(amount):
    count = 0
    Aweight = []
    Dweight = []
    ll = 0
    for run in range(repeat):
        print("RUN:", run)
        rn1 = random.randint(0, 272)
        rn2 = random.randint(0, 272)
        path1, weight1 = a_star_search(adj_matrix, rn1, rn2)
        path2, weight2 = dijkstra(adj_matrix, rn1, rn2)
        if path1 != path2:
            count = count + 1
        Aweight.insert(ll, weight1)
        Dweight.insert(ll, weight2)
        ll = ll + 1


    totalsum1.insert(avg, sum(Aweight))
    totalsum2.insert(avg, sum(Dweight))
    avgcalc.insert(avg, count)



print("avrg wrong calc" ,np.mean(avgcalc))
print("avrg weight A*" ,np.mean(totalsum1))
print("avrg weight Dijk" ,np.mean(totalsum2))