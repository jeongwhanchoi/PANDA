from torch_geometric.utils import to_networkx
from numba import jit, int64
import networkx as nx
import numpy as np
from math import inf


@jit(nopython=True)
def choose_edge_to_add(x, edge_index, degrees):
	# chooses edge (u, v) to add which minimizes y[u]*y[v]
	n = x.size
	m = edge_index.shape[1]
	y = x / ((degrees + 1) ** 0.5)
	products = np.outer(y, y)
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		products[u, v] = inf
	for i in range(n):
		products[i, i] = inf
	smallest_product = np.argmin(products)
	return (smallest_product % n, smallest_product // n)

@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
	# returns array of degrees of all nodes
	if num_nodes is None:
		num_nodes = np.max(edge_index) + 1
	degrees = np.zeros(num_nodes)
	m = edge_index.shape[1]
	for i in range(m):
		degrees[edge_index[0, i]] += 1
	return degrees

@jit(nopython=True)
def add_edge(edge_index, u, v):
	new_edge = np.array([[u, v],[v, u]])
	return np.concatenate((edge_index, new_edge), axis=1)

@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
	# given an edge_index, computes Ax, where A is the corresponding adjacency matrix
	n = x.size
	y = np.zeros(n)
	m = edge_index.shape[1]
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		y[u] += x[v]
	return y

@jit(nopython=True)
def compute_spectral_gap(edge_index, x):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
	for i in range(n):
		if x[i] > 1e-9:
			return 1 - y[i]/x[i]
	return 0.

@jit(nopython=True)
def _edge_rewire(edge_index, edge_type, x=None, num_iterations=50, initial_power_iters=50):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	for i in range(initial_power_iters):
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
		edge_index = add_edge(edge_index, i, j)
		degrees[i] += 1
		degrees[j] += 1
		edge_type = np.append(edge_type, 1)
		edge_type = np.append(edge_type, 1)
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return edge_index, edge_type, x

def edge_rewire(edge_index, x=None, edge_type=None, num_iterations=50, initial_power_iters=5):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	if edge_type is None:
		edge_type = np.zeros(m, dtype=np.int64)
	return _edge_rewire(edge_index, edge_type=edge_type, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)