from copy import deepcopy
import itertools
import numpy as np
import torch 
from numpy import linalg as LA
import networkx as nx
from discreteMarkovChain import markovChain

def create_matrix(eigenvectors, eigenvalues):
    """
    Create a matrix from eigenvectors and eigenvalues.
    """
    return np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))

def create_orthogonal_vectors(n):
    # create a matrix of random numbers
    A = np.random.rand(n, n)
    # make it (right) stochastic 
    A = A / A.sum(axis=1)
    # get the eigenvectors
    lambda_, v = np.linalg.eig(A.T)
    return v

def create_eigenvalues(n, v):
    eps = 1e-5
    return [1] + [1-v] + np.random.uniform(eps, 1 - v - 0.1, n - 2).tolist()

def create_markov_matrix_by_velocity(N, v, eps = 1e-3):
    P = np.random.rand(N, N)
    P = P / P.sum(axis=0)  # make it left stochastic (column sum to 1) 
    eps = 1e-3
    calc_v = get_velocity(P)
    while calc_v < v - eps or calc_v > v + eps:
        P = np.random.rand(N, N)
        P = P / P.sum(axis=0)
        calc_v = get_velocity(P)

    return torch.tensor(P, dtype=torch.double)

def get_velocity(P):
    if isinstance(P, torch.Tensor):
        P = P.numpy()
    eigenvalues, _ = LA.eig(P)
    eigenvalues = sorted(np.abs(eigenvalues), reverse=True)
    return eigenvalues[0] - eigenvalues[1]

def create_initial_states(N, M):
    walker_states = torch.zeros((N, M), dtype=torch.double)
    for walker_id in range(M):
        random_state = np.random.randint(0, N)
        walker_states[random_state, walker_id] = 1

    return walker_states

def kl_divergence(p, q):
    return sum([p[g_k_sample]['probability'] * np.log(p[g_k_sample]['probability']) - p[g_k_sample]['probability'] * np.log(q[g_k_sample]['probability']) for g_k_sample in p.keys()])

def create_random_policy(N):
    P = torch.rand(N, N, dtype=torch.double)
    P = P / P.sum(dim=0)
    return P

def generate_rwig_transition_matrix(sequence, sample_space_size):
    transition_matrix = np.zeros((sample_space_size, sample_space_size))
    for g_id in range(len(sequence)-1):
        transition_matrix[sequence[g_id], sequence[g_id+1]] += 1

    norm_factor = np.where(transition_matrix.sum(axis=1)[:, None] == 0, 1, transition_matrix.sum(axis=1)[:, None])
    return transition_matrix / norm_factor

def generate_initial_condition_from_graph(graph, N):
    M = graph.number_of_nodes()
    initial_condition = np.zeros((N, M), dtype=int)
    cliques = sorted(nx.connected_components(graph), key=len, reverse=True)
    for i, clique in enumerate(cliques):
        for node in clique:
            initial_condition[i, node] = 1
    return torch.tensor(initial_condition, dtype = torch.double)

def generate_all_initial_conditions_from_graph(graph, N):
    # Generate all permutations of N
    permutations = list(itertools.permutations(range(N)))
    initial_conditions = generate_initial_condition_from_graph(graph, N)
    all_initial_conditions = []
    for perm in permutations:
        new_initial_conditions = torch.tensor(deepcopy(initial_conditions)[[perm]], dtype=torch.double)
        all_initial_conditions.append(new_initial_conditions)
    return all_initial_conditions

def get_steady_state_distribution(P):
    if P[0].sum() != 1:  # Make sure P is row-stochastic (right stochastic)
        P = P.T  
    mc = markovChain(P.numpy())
    mc.computePi('linear')  # Compute stationary distribution
    pi = torch.tensor(mc.pi, dtype=torch.double)
    return pi
