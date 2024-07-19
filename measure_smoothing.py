from numba import jit
import numpy as np

@jit(nopython=True)
def dirichlet_energy(X, edge_index):
    # computes Dirichlet energy of a vector field X with respect to a graph with a given edge index
    n = X.shape[0]
    m = len(edge_index[0])
    l = X.shape[1]
    degrees = np.zeros(n)
    for I in range(m):
        u = edge_index[0][I]
        degrees[u] += 1
    y = np.linalg.norm(X.flatten()) ** 2
    for I in range(m):
        for i in range(l):
            u = edge_index[0][I]
            v = edge_index[1][I]
            y -= X[u][i] * X[v][i] / (degrees[u] * degrees[v]) ** 0.5
    return y
def dirichlet_normalized(X, edge_index):
    energy = dirichlet_energy(X, edge_index)
    norm_squared = sum(sum(X ** 2))
    return energy / norm_squared