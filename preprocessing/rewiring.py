import torch
import torch_geometric
import numpy as np
from numpy.random import random
import networkx as nx
from math import inf
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

degree = torch_geometric.utils.degree
softmax = torch.nn.Softmax(dim=0)

def to_undirected(data):
	# makes the graph undirected and passes to the largest connected component
	G = to_networkx(data, to_undirected=True)
	largest_cc = max(nx.connected_components(G), key=len)
	G = G.subgraph(largest_cc).copy()
	new_data = from_networkx(G)
	return new_data.edge_index, new_data.num_nodes

def argmin(d):
	smallest = inf
	for i in d:
		if d[i] <= smallest:
			smallest = d[i]
			key_of_smallest = i
	return key_of_smallest
def argmax(d):
	largest = -inf
	for i in d:
		if d[i] > largest:
			largest = d[i]
			key_of_largest = i
	return key_of_largest

def spectral_gap(G):
	if not nx.is_connected(G):
		return 0.
	return nx.normalized_laplacian_spectrum(G)[1]

def lower_bound_cheeger(G, d):
	return (nx.normalized_laplacian_spectrum(G)[1])/2

def number_of_triangles(G):
	triangles = 0
	for (i, j) in G.edges:
		i_nbhd = set(G.neighbors(i))
		j_nbhd = set(G.neighbors(j))
		triangles += len(i_nbhd.intersection(j_nbhd))
	return triangles / 3

def sample(weights, temperature=1, use_softmax=True):
	# samples randomly from a list of weights
	weights = torch.tensor(weights)
	seed = random()
	if use_softmax:
		probabilities = softmax(temperature * weights)
	else:
		probabilities = weights / sum(weights)
	N = len(weights)
	for i in range(N):
		seed -= probabilities[i]
		if seed < 0:
			return i
	return N - 1

def second_neighborhood(i, G):
	# returns all vertices of distance at most 2 from a given vertex i in G
	second_neighbors = set()
	for j in G.neighbors(i):
		second_neighbors.add(j)
		for k in G.neighbors(j):
			second_neighbors.add(k)
	second_neighbors.add(i)
	return second_neighbors

def balanced_forman(i, j, G):
	# Calculates Ric(i, j) for a graph G of type networkx.Graph
	di = G.degree(i)
	dj = G.degree(j)
	if di <= 1 or dj <= 1:
		return 0
	neighbors_of_i = set(G.neighbors(i))
	neighbors_of_j = set(G.neighbors(j))
	num_triangles = 0
	triangles = neighbors_of_i.intersection(neighbors_of_j)
	num_triangles = len(triangles)
	potential_squares = set()
	neighbors_of_i_only = neighbors_of_i.difference(neighbors_of_j).difference({j})
	neighbors_of_j_only = neighbors_of_j.difference(neighbors_of_i).difference({i})
	for v in neighbors_of_i_only:
		for w in G.neighbors(v):
			if w in neighbors_of_j_only:
				potential_squares.add((v, w))
	squares_at_i = {v for (v, w) in potential_squares}
	squares_at_j = {w for (v, w) in potential_squares}
	for (v, w) in potential_squares:
		squares_at_i.add(v)
		squares_at_j.add(w)
	num_squares_i = len(squares_at_i)
	num_squares_j = len(squares_at_j)
	gamma_max = 0
	for k in squares_at_i:
		potential_gamma = 0
		for w in G.neighbors(k):
			if w in neighbors_of_j and not w in neighbors_of_i:
				potential_gamma += 1
		potential_gamma -= 1
		gamma_max = max(gamma_max, potential_gamma)
	for k in squares_at_j:
		potential_gamma = 0
		for w in G.neighbors(k):
			if w in neighbors_of_i and not w in neighbors_of_j:
				potential_gamma += 1
		potential_gamma -= 1
		gamma_max = max(gamma_max, potential_gamma)
	triangle_term = 2 * num_triangles / max(di, dj) + num_triangles / min(di, dj)
	if gamma_max == 0:
		square_term = 0
	else:
		square_term = (num_squares_i + num_squares_j)/(gamma_max * max(di, dj))
	ric = 2/di + 2/dj - 2 + triangle_term + square_term
	return ric

class CurvatureGraph:
	# data structure that keeps track of curvature changes in a graph
	def __init__(self, G):
		self.G = G
		self.num_nodes = len(G.nodes)
		self.num_edges = len(G.edges)
		self.curvatures = {}
		self.total_curvature = 0
		for edge in G.edges:
			(u, v) = edge
			ric_uv = balanced_forman(u, v, G)
			self.curvatures[(u,v)] = ric_uv
			self.total_curvature += ric_uv
	def mean_curvature(self):
		return self.total_curvature / self.num_edges
	def ric(edge):
		return self.curvatures[edge]
	def update_curvature(self, edge_set):
		for (u, v) in edge_set:
			old_curvature = self.curvatures[(u,v)]
			new_curvature = balanced_forman(u, v, G)
			self.total_curvature += (new_curvature - old_curvature)

def compute_curvature(G):
	# computes Ric(i, j) for all edges (i, j)
	curvatures = {}
	for edge in G.edges:
		(u, v) = edge
		(a, b) = (min(u,v), max(u,v))
		curvatures[(a,b)] = balanced_forman(a, b, G)
	return curvatures
def average_curvature(G, curvatures=None):
	if curvatures == None:
		curv = compute_curvature(G)
	else:
		curv = curvatures
	total = 0
	for edge in curv:
		total += curv[edge]
	return total/len(curv)
def randomized_average_curvature(G, num_samples=100):
	# estimates average curvature of G based on a random sample
	edges = list(G.edges)
	num_edges = len(edges)
	curvatures = []
	for i in range(num_samples):
		choice = np.random.randint(0, num_edges)
		(u, v) = edges[choice]
		curvatures.append(balanced_forman(u, v, G))
	return np.average(curvatures)
def sdrf(G, curvatures=None, max_iterations=1, temperature=5, C_plus=None):
	# stochastic discrete ricci flow
	num_nodes = len(G.nodes)
	num_edges = len(G.edges)
	if curvatures == None:
		curvatures = compute_curvature(G)
	for iteration in range(max_iterations):
		#print(iteration)
		(u, v) = argmin(curvatures)
		#print(u, v)
		#print(u, v, curvatures[(u,v)])
		ric_uv = curvatures[(u, v)]
		improvements = {}
		for k in G.neighbors(u):
			for l in G.neighbors(v):
				a = min(k, l)
				b = max(k, l)
				if not (a, b) in G.edges:
					G.add_edge(a, b)
					improvements[(a,b)] = balanced_forman(u, v, G) - ric_uv
					G.remove_edge(a, b)
		if improvements != {}:
			improvements_list = [[k, l, improvements[(k,l)]] for (k, l) in improvements]
			improvement_values = [x[2] for x in improvements_list]
			chosen_index = sample(improvement_values,temperature=temperature)
			i = improvements_list[chosen_index][0]
			j = improvements_list[chosen_index][1]
			G.add_edge(i, j)
			#print(i,j)
			# need to update curvatures at neighbors of i and j
			edges_to_update = set()
			for w in G.neighbors(i):
				a = min(w, i)
				b = max(w, i)
				edges_to_update.add((a,b))
			for x in G.neighbors(j):
				a = min(x, j)
				b = max(x, j)
				edges_to_update.add((a,b))
			for w in G.neighbors(i):
				for x in G.neighbors(j):
					if x in G.neighbors(w):
						a = min(w, x)
						b = max(w, x)
						edges_to_update.add((a,b))
			for edge in edges_to_update:
				(w, x) = edge
				curvatures[(w, x)] = balanced_forman(w, x, G)
			if C_plus != None:
				highest_curvature_edge = argmax(curvatures)
				(i, j) = highest_curvature_edge
				(i, j) = (min(i, j), max(i,j))
				if curvatures[highest_curvature_edge] > C_plus:
					G.remove_edge(i, j)
					curvatures.pop((i,j))
			edges_to_update = set()
			for w in G.neighbors(i):
				a = min(w, i)
				b = max(w, i)
				edges_to_update.add((a,b))
			for x in G.neighbors(j):
				a = min(x, j)
				b = max(x, j)
				edges_to_update.add((a,b))
			for w in G.neighbors(i):
				for x in G.neighbors(j):
					if x in G.neighbors(w):
						a = min(w, x)
						b = max(w, x)
						edges_to_update.add((a,b))
			for edge in edges_to_update:
				(w, x) = edge
				curvatures[(w, x)] = balanced_forman(w, x, G)
	return G, curvatures

def rlef(G):
	# algorithm 1 from Overleaf (Random Local Edge Flip)
	edge_list = list(G.edges)
	chosen_edge = edge_list[np.random.randint(len(edge_list))]
	(u, v) = chosen_edge
	i = np.random.choice(list(G.neighbors(u)))
	if  i in G.neighbors(v) or i == v:
		return G
	else:
		eligible_nodes = set(G.neighbors(v)).difference(set(G.neighbors(u))).difference({u})
		if eligible_nodes == set():
			return G
		else:
			j = np.random.choice(list(eligible_nodes))
			G.remove_edge(i,u)
			G.remove_edge(j,v)
			G.add_edge(i,v)
			G.add_edge(j,u)
		return G
def greedy_rlef(G, triangle_data=None):

	# samples greedily according to inverse triangle count
	if triangle_data == None:
		triangle_data = {}
		for (u,v) in G.edges:
			u_nbhd = set(G.neighbors(u))
			v_nbhd = set(G.neighbors(v))
			num_triangles = len(u_nbhd.intersection(v_nbhd))
			triangle_data[(u,v)] = num_triangles
			triangle_data[(v,u)] = num_triangles

	(u, v) = argmin(triangle_data)
	if not (u, v) in G.edges:
		print("ERROR")
	eligible_i = list(set(G.neighbors(u)).difference(set(G.neighbors(v))).difference({v}))
	if not eligible_i:
		return triangle_data
	i = np.random.choice(eligible_i)
	eligible_j = list(set(G.neighbors(v)).difference(set(G.neighbors(u))).difference({u}))
	if not eligible_j:
		return triangle_data
	j = np.random.choice(eligible_j)
	print(u, v)
	G.remove_edge(j,v)
	G.remove_edge(i,u)
	G.add_edge(i,v)
	G.add_edge(j,u)

	triangle_data.pop((i,u))
	triangle_data.pop((u,i))
	triangle_data.pop((j,v))
	triangle_data.pop((v,j))

	u_nbhd = set(G.neighbors(u))
	v_nbhd = set(G.neighbors(v))
	i_nbhd = set(G.neighbors(i))
	j_nbhd = set(G.neighbors(j))

	triangle_data[(i,v)] = len(i_nbhd.intersection(v_nbhd))
	triangle_data[(v,i)] = len(i_nbhd.intersection(v_nbhd))
	triangle_data[(j,u)] = len(j_nbhd.intersection(u_nbhd))
	triangle_data[(u,j)] = len(j_nbhd.intersection(u_nbhd))
	return triangle_data

def grlef(G, triangle_data=None, temperature=5):
	# samples greedily according to inverse triangle count
	
	if triangle_data == None:
		triangle_data = {}
		for (u,v) in G.edges:
			u_nbhd = set(G.neighbors(u))
			v_nbhd = set(G.neighbors(v))
			num_triangles = len(u_nbhd.intersection(v_nbhd))
			triangle_data[(u,v)] = num_triangles
			triangle_data[(v,u)] = num_triangles
	
	triangle_data_list = [[e, triangle_data[e]] for e in triangle_data]
	edge_list = [x[0] for x in triangle_data_list]
	
	weights = [1/(2 + x[1]) for x in triangle_data_list]
	selected_index = sample(weights, temperature=temperature)
	(u, v) = edge_list[selected_index]

	u_nbhd = set(G.neighbors(u))
	v_nbhd = set(G.neighbors(v))
	eligible_i = list(u_nbhd.difference(v_nbhd).difference({v}))
	if not eligible_i:
		return triangle_data
	
	# choose the value of i which removes as many triangles as possible

	i_scores = {}
	for node in eligible_i:
		node_nbhd = set(G.neighbors(node))
		triangles_added = len(v_nbhd.intersection(node_nbhd))
		triangles_removed = len(u_nbhd.intersection(node_nbhd))
		i_scores[node] = triangles_added - triangles_removed
	i = argmin(i_scores)



	eligible_j = list(v_nbhd.difference(u_nbhd).difference({u}))
	if not eligible_j:
		return triangle_data

	# choose the value of j which removes as many triangles as possible

	j_scores = {}
	for node in eligible_j:
		node_nbhd = set(G.neighbors(node))
		triangles_added = len(u_nbhd.intersection(node_nbhd))
		triangles_removed = len(v_nbhd.intersection(node_nbhd))
		j_scores[node] = triangles_added - triangles_removed
	j = argmin(j_scores)
	#print(u, v, i, j)
	G.remove_edge(j,v)
	G.remove_edge(i,u)
	G.add_edge(i,v)
	G.add_edge(j,u)

	triangle_data.pop((i,u))
	triangle_data.pop((u,i))
	triangle_data.pop((j,v))
	triangle_data.pop((v,j))

	u_nbhd = set(G.neighbors(u))
	v_nbhd = set(G.neighbors(v))
	i_nbhd = set(G.neighbors(i))
	j_nbhd = set(G.neighbors(j))

	triangle_data[(i,v)] = len(i_nbhd.intersection(v_nbhd))
	triangle_data[(v,i)] = len(i_nbhd.intersection(v_nbhd))
	triangle_data[(j,u)] = len(j_nbhd.intersection(u_nbhd))
	triangle_data[(u,j)] = len(j_nbhd.intersection(u_nbhd))

	return triangle_data

def greedy_rlef_3(G):
	edge_list = list(G.edges)
	(u, v) = edge_list[np.random.randint(0, len(edge_list))]

	u_nbhd = set(G.neighbors(u))
	v_nbhd = set(G.neighbors(v))
	eligible_i = list(u_nbhd.difference(v_nbhd).difference({v}))
	if not eligible_i:
		return G
	
	# choose the value of i which removes as many triangles as possible

	i_scores = {}
	for node in eligible_i:
		node_nbhd = set(G.neighbors(node))
		triangles_added = len(v_nbhd.intersection(node_nbhd))
		triangles_removed = len(u_nbhd.intersection(node_nbhd))
		i_scores[node] = triangles_added - triangles_removed
	i = argmin(i_scores)

	eligible_j = list(v_nbhd.difference(u_nbhd).difference({u}))
	if not eligible_j:
		return G

	# choose the value of j which removes as many triangles as possible

	j_scores = {}
	for node in eligible_j:
		node_nbhd = set(G.neighbors(node))
		triangles_added = len(u_nbhd.intersection(node_nbhd))
		triangles_removed = len(v_nbhd.intersection(node_nbhd))
		j_scores[node] = triangles_added - triangles_removed
	j = argmin(j_scores)

	#print(u, v, i, j)
	G.remove_edge(j,v)
	G.remove_edge(i,u)
	G.add_edge(i,v)
	G.add_edge(j,u)

	u_nbhd = set(G.neighbors(u))
	v_nbhd = set(G.neighbors(v))
	i_nbhd = set(G.neighbors(i))
	j_nbhd = set(G.neighbors(j))

def augment_degree(G):
	i = argmin(dict(G.degree))
	neighbors_of_i = G.neighbors(i)
	second_neighbors_of_i = second_neighborhood(i, G)
	second_neighbors_of_i = second_neighbors_of_i.difference(set(neighbors_of_i)).difference({i})
	if second_neighbors_of_i == set():
		return None
	lowest_degree = inf
	for j in second_neighbors_of_i:
		if G.degree(j) < lowest_degree:
			best_second_neighbor = j
			lowest_degree = G.degree(j)
	G.add_edge(i, j)
	return G

# DIGL pre-processing, from https://github.com/gasteigerjo/gdc.git

def get_adj_matrix(dataset) -> np.ndarray:
    num_nodes = dataset.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.edge_index[0], dataset.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def digl(base, alpha, k=None, eps=None):
	# generate adjacency matrix from sparse representation
    adj_matrix = get_adj_matrix(base)
    # obtain exact PPR matrix
    ppr_matrix = get_ppr_matrix(adj_matrix, alpha=alpha)

    if k != None:
            print(f'Selecting top {k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps != None:
            print(f'Selecting edges with weight greater than {eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError

        # create PyG Data object
    edges_i = []
    edges_j = []
    edge_attr = []
    for i, row in enumerate(ppr_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(ppr_matrix[i, j])
    edge_index = [edges_i, edges_j]

    data = Data(
        x=base.x,
        edge_index=torch.LongTensor(edge_index),
        y=base.y
    )        
    return data