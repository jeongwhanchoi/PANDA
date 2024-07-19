from torch_geometric.utils import to_networkx
import networkx as nx
import networkit as nk
import torch
import os
import pickle

def save_centrality(centrality_values, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as file:
        pickle.dump(centrality_values, file)

def load_centrality(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return None

def measure_centrality(graph, centrality_measure='degree', index=0, save_path=None):
    full_save_path = os.path.join(save_path, f"{centrality_measure}_{index}.pkl") if save_path else None
    if full_save_path:
        precomputed_centrality = load_centrality(full_save_path)
        if precomputed_centrality is not None:
            return precomputed_centrality
    # Convert PyG graph to NetworkX graph for centrality measures
    G = to_networkx(graph, to_undirected=True)
    # Placeholder for centrality computation
    centrality = None
    # Compute centrality based on the selected measure
    if centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == 'eigen':
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_measure == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_measure == 'pagerank':
        centrality = nx.pagerank(G)
    elif centrality_measure == 'katz':
        centrality = nx.katz_centrality(G)
    elif centrality_measure == 'laplacian':
        centrality = nx.laplacian_centrality(G)
    elif centrality_measure == 'second':
        centrality = nx.second_order_centrality(G)
    elif centrality_measure == 'harmonic':
        centrality = nx.harmonic_centrality(G)
    elif centrality_measure == 'load':
        centrality = nx.load_centrality(G)
    elif centrality_measure == 'current':
        centrality = nx.current_flow_betweenness_centrality(G)
    elif centrality_measure == 'betweenness-nk':
        G = nk.nxadapter.nx2nk(G)
        centrality = nk.centrality.Betweenness(G)
        centrality.run()
        centrality = centrality.scores()
    elif centrality_measure == 'approx_betweenness-nk':
        G = nk.nxadapter.nx2nk(G)
        centrality = nk.centrality.ApproxBetweenness(G, epsilon=0.1)
        centrality.run()
        centrality = centrality.scores()
    elif centrality_measure == 'closeness-nk':
        G = nk.nxadapter.nx2nk(G)
        centrality = nk.centrality.Closeness(G)
        centrality.run()
        centrality = centrality.scores()
    elif centrality_measure == 'approx_closeness-nk':
        G = nk.nxadapter.nx2nk(G)
        centrality = nk.centrality.ApproxCloseness(G, nSamples=10, epsilon=0.1)
        centrality.run()
        centrality = centrality.scores()
    else:
        raise NameError('Centrality measure not found')

    # Convert centrality to tensor and sort
    if centrality_measure in ['betweenness-nk', 'approx_betweenness-nk', 'closeness-nk', 'approx_closeness-nk', 'eigen-nk']:
        centrality_values = torch.tensor(centrality)
    else:
        centrality_values = torch.tensor(list(centrality.values()))
    # Save centrality values
    if full_save_path:
        save_centrality(centrality_values, full_save_path)
    
    return centrality_values