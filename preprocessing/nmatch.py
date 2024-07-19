import numpy as np
import networkx as nx
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx, from_networkx

def path_of_cliques(num_cliques, size_of_clique):
	G = nx.Graph([])
	for i in range(num_cliques):
		for j in range(size_of_clique):
			for k in range(j):
				G.add_edge(i*size_of_clique+j, i*size_of_clique+k)
		if i != num_cliques - 1:
			G.add_edge((i+1)*size_of_clique - 1, (i+1)*size_of_clique)
	return G

def ring_of_cliques(n, d):
	# ring of cliques graph with n vertices of degree d
	G = nx.Graph([])
	k = d + 1
	# encodes vertex by which clique it's in
	f = lambda x: (x % k, x // k)
	for x1 in range(n):
		for x2 in range(x1):
			(u1, v1) = f(x1)
			(u2, v2) = f(x2)
			if v1 == v2 and u1 - u2 != k - 1:
				G.add_edge(x1, x2)
			elif u2 - u1 == k - 1 and v1 - v2 == 1:
				G.add_edge(x1, x2)
	G.add_edge(n - 1, 0)
	return G

def create_neighborsmatch_labels(G, root_vertex, vertices_to_label):
	# generates a dataset for the neighborsmatch problem
	# vertices_to_label consists of vertices which we will randomly label with different one-hot vectors
	# root_vertex is the vertex which needs to guess which vertex has the matching one-hot label
	num_classes = len(vertices_to_label)
	num_nodes = len(list(G.nodes))
	class_labels = torch.randperm(num_classes)
	vertex_index_list = []
	matching_entry = torch.randint(0, num_classes, ())
	for i in range(len(G.nodes)):
		if i in vertices_to_label:
			entry = class_labels[vertices_to_label.index(i)]
			if entry == matching_entry:
				y = i
		elif i == root_vertex:
			entry = matching_entry
		else:
			entry = torch.tensor(num_classes)
		vertex_index_list.append(entry)
	vertex_one_hot_tensor = one_hot(torch.stack(vertex_index_list))

	# encoding of node numbers, so the root can distinguish between them
	node_indictors = one_hot(torch.arange(len(G.nodes)))
	vertex_features = torch.concat([vertex_one_hot_tensor, node_indictors], dim=1)
	root_mask = torch.zeros(num_nodes, dtype=int)
	root_mask[root_vertex] = 1
	root_mask = root_mask.bool()
	return vertex_features, y, root_mask

def create_neighborsmatch_dataset(G, root_vertex, vertices_to_label, sample_size):
	data_list = []
	edge_index = from_networkx(G).edge_index
	for i in range(sample_size):
		x, y, root_mask = create_neighborsmatch_labels(G, root_vertex, vertices_to_label)
		data_list.append(Data(x=x, y=y, edge_index=edge_index, root_mask=root_mask))
	return data_list

if __name__ == "__main__":
	G = path_of_cliques(5, 10)
	vertices_to_label = list(range(0, 100))
	nmatch = create_neighborsmatch_dataset(G, 499, vertices_to_label, 10000)


