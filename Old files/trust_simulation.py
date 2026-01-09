import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import numba
import scipy as sp
import inspect
from collections import Counter
from metrics import normalize_each_row_sum, calculate_metrics
from agents import initialize_beliefs, get_likelihoods
from networks import create_graphs

@numba.jit(nopython=True)
def initialize_weights(w_type, neighbor_indices, adj_matrix, trust_matrix, agent, public_beliefs, private_beliefs, true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs):
        epsilon = 1e-6
        neighbor_beliefs = public_beliefs[neighbor_indices]
        neighbor_beliefs = np.clip(neighbor_beliefs, epsilon, None)
        num_neighbors = neighbor_beliefs.shape[0]

        # -- Random weights -- (oops - uniform - change later)
        if w_type == "random":
            #weights = adj_matrix[neighbor_indices, agent] # Ingen confirmation bias
            #weights = 0.5*weights + 0.5*trust_matrix[neighbor_indices, agent]
            weights = trust_matrix[neighbor_indices, agent] + 1e-12
            #weights = np.random.rand(num_neighbors).astype(np.float64)

        # -- Sigmoid weights --
        elif w_type == "sigmoid":
            sigmoid_factor = 4
            x = np.zeros(num_neighbors, dtype=np.float64)
            for k in range(num_neighbors):
                x[k] = 1 - np.linalg.norm(np.abs(private_beliefs[agent] - neighbor_beliefs[k]))
            weights = 1/(1 + np.exp(sigmoid_factor*(x - 0.5)))

            if true_mega_node_bool:
                weights[0] = np.dot(private_beliefs[agent], true_mega_node_beliefs)
            if consp_mega_node_bool:
                weights[0] = np.dot(private_beliefs[agent], consp_mega_node_beliefs)

        weights = weights / np.sum(weights)

        # if weights.shape[0] != neighbor_beliefs.shape[0]:
        #     print(f"BAD SHAPE DETECTED: weights.shape = {weights.shape[0]}, np.log(neighbor_beliefs).shape = {np.log(neighbor_beliefs).shape[0]}")
        #     print(f"Failure at simulation {counter1}, iteration {counter2}, agent {i}.")
        #     raise ValueError("Expected 1D arrays for dot product")
        
        return weights, neighbor_beliefs

@numba.jit(nopython=True)
def update_beliefs(N, M, private_beliefs, w_type, adj_matrix, trust_matrix, likelihood, counter1, counter2, conspirator_bool, conspirators, true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs, neighbor_prob, neighbor_number):
    """For every time-step this function will update the beliefs of the agents in the network.
    1. prepare belief vectors
    2. if turned on, initalize "sociopaths" and "conspirators"
    3. If turned on, update private beliefs according to cognitive dissonance
    4. each agent receives a signal from distribution
    5. each agent performs a Bayesian update based on their neighbors beliefs

    Args:
        M (int): Number of hypotheses in the network
        private_beliefs (NxM-array): private belief vectors for all agents
        adj_matrix: The graph network as a np.array
        likelihood (NxM-array): Signal likelihood for all agents
        true_hypothesis (int): index for the true hypothesis
        cap (float): maximum likelihood value
        sociopaths (1d-array, optional): indices for the "sociopathic" agents. Defaults to None.
        conspirators (1d-array, optional): indices for the conspirating agents. Defaults to None.
        mega_node_bool (bool): if True, the graph will have a mega node.
        mega_node_beliefs (nd-array sjape=(M)): The beliefs of the mega node.

    Returns:
        T (float), C (float): Truth and cognitive dissonance of the network
    """
    # epsilon = 1e-6
    # scaling_factor = 10
    # true_hypothesis = M-1
    # cap = 1

    # LRTU_exp = 1 + scaling_factor*np.abs(np.clip(private_beliefs, epsilon, 1) - np.clip(likelihood, epsilon, 1))
    LRTU_exp = 1 # Hva er dette?
    # private_beliefs[:, true_hypothesis] += cap*likelihood[:, true_hypothesis]
    # private_beliefs = normalize_each_row_sum(private_beliefs, N, M)

    public_beliefs = likelihood**LRTU_exp*private_beliefs
    public_beliefs = normalize_each_row_sum(public_beliefs, N, M)

    if true_mega_node_bool:
        public_beliefs[0] = true_mega_node_beliefs
    if consp_mega_node_bool:
        public_beliefs[-1] = consp_mega_node_beliefs

    if conspirator_bool:
        conspirator_beliefs = np.zeros(M, dtype=np.float64)
        conspirator_beliefs[0] = 1
        public_beliefs[conspirators] = conspirator_beliefs

    for i, belief in enumerate(public_beliefs):

        neighbor_indices_temp = np.where(adj_matrix[:, i] > 0)[0]  # Get neighbors from adjacency matrix
        neighbor_indices = np.copy(neighbor_indices_temp)


        if true_mega_node_bool and i != 0:
            if private_beliefs[i].ndim != 1 or true_mega_node_beliefs.ndim != 1:
                print(f"BAD SHAPE DETECTED: private_beliefs[{i}].shape = {private_beliefs[i].shape}, true_mega_node_beliefs.shape = {true_mega_node_beliefs.shape}")
                print(f"Failure at simulation {counter1}, iteration {counter2}, agent {i}.")
                raise ValueError("Expected 1D arrays for dot product")
            dot_product = np.dot(private_beliefs[i], true_mega_node_beliefs)
            # dot_product = private_beliefs[i, true_hypothesis]*true_mega_node_beliefs[true_hypothesis]
            mega_node_idx = 0
            if mega_node_idx in neighbor_indices and np.random.rand() > dot_product:
                neighbor_indices = neighbor_indices[neighbor_indices != mega_node_idx]
            
        if consp_mega_node_bool and i != N-1:
            if private_beliefs[i].ndim != 1 or consp_mega_node_beliefs.ndim != 1:
                print(f"BAD SHAPE DETECTED: private_beliefs[{i}].shape = {private_beliefs[i].shape}, consp_mega_node_beliefs.shape = {true_mega_node_beliefs.shape}")
                print(f"Failure at simulation {counter1}, iteration {counter2}, agent {i}.")
                raise ValueError("Expected 1D arrays for dot product")
            dot_product = np.dot(private_beliefs[i], consp_mega_node_beliefs)
            # dot_product = private_beliefs[i, 0]*consp_mega_node_beliefs[0]
            mega_node_idx = N-1
            if mega_node_idx in neighbor_indices and np.random.rand() > dot_product:
                neighbor_indices = neighbor_indices[neighbor_indices != mega_node_idx]

        if len(neighbor_indices) == 0:
            continue 
        elif true_mega_node_bool and i==0:
            continue
        elif consp_mega_node_bool and i==N-1:
            continue

        weights, neighbor_beliefs = initialize_weights(w_type, neighbor_indices, adj_matrix, trust_matrix, i, public_beliefs, private_beliefs, 
                                                       true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs)


        if weights.shape[0] != neighbor_beliefs.shape[0]:
            print(f"BAD SHAPE DETECTED: weights.shape = {weights.shape[0]}, np.log(neighbor_beliefs).shape = {np.log(neighbor_beliefs).shape[0]}")
            print(f"Failure at simulation {counter1}, iteration {counter2}, agent {i}.")
            raise ValueError("Expected 1D arrays for dot product")

        if conspirator_bool and i in conspirators:
            continue
        else:
            log_sum = np.dot(weights, np.log(neighbor_beliefs))
            exp_log_sum = np.exp(log_sum)
            private_beliefs[i] = exp_log_sum/np.sum(exp_log_sum, axis=0)

    # C_agent = public_beliefs - private_beliefs
    # for j in range(M):
    #     private_beliefs[:, j] += 0.7*C_agent[:, j]
    # private_beliefs = normalize_each_row_sum(private_beliefs, N, M)

    return private_beliefs, public_beliefs