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
def remove_links(A, prob):
    """Works for symmetric matrices"""
    A = np.copy(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                if np.random.rand() < prob:
                    A[i, j] = 0
                    A[j, i] = 0
    return A


@numba.jit(nopython=True)
def initialize_weights(w_type, neighbor_indices, adj_matrix, agent, public_beliefs, private_beliefs, true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs):
        epsilon = 1e-6
        neighbor_beliefs = public_beliefs[neighbor_indices]
        neighbor_beliefs = np.clip(neighbor_beliefs, epsilon, None)
        num_neighbors = neighbor_beliefs.shape[0]

        # -- Random weights -- (oops - uniform - change later)
        if w_type == "random":
            weights = adj_matrix[neighbor_indices, agent] # Ingen confirmation bias
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
def update_beliefs(N, M, private_beliefs, w_type, adj_matrix, likelihood, counter1, counter2, conspirator_bool, conspirators, true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs, neighbor_prob, neighbor_number):
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

    # --- Remove a percentage of the links in the network, to ensure pairwise interactions ---
    #adj_matrix = remove_links(adj_matrix, neighbor_prob)
    # ----------
    
    for i, belief in enumerate(public_beliefs):

        neighbor_indices_temp = np.where(adj_matrix[:, i] > 0)[0]  # Get neighbors from adjacency matrix
        neighbor_indices = np.copy(neighbor_indices_temp)

        # --- Random neighbors by fraction
        # mask = np.random.rand(*np.shape(neighbor_indices)) < neighbor_prob
        # neighbor_indices = neighbor_indices[mask]
        # ----

        # --- Random neighbors by number
        # if len(neighbor_indices) > neighbor_number: # interact with a set maximum number of neighbours
        #     neighbor_indices = np.random.choice(neighbor_indices, size = neighbor_number, replace=False) # choose a number of neighbors to interact with

        # --- Random neighbors by fraction, with cutoff value
        # mask = np.random.rand(*np.shape(neighbor_indices)) < neighbor_prob
        # # print(len(neighbor_indices))
        # neighbor_indices = neighbor_indices[mask]
        # # print(len(neighbor_indices))
        # if len(neighbor_indices) > neighbor_number:
        #     neighbor_indices = neighbor_indices[0:neighbor_number] # reduce number of neighbors
        # print(len(neighbor_indices))
        # print("---")

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

        weights, neighbor_beliefs = initialize_weights(w_type, neighbor_indices, adj_matrix, i, public_beliefs, private_beliefs, 
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

@numba.jit(nopython=True)
def simulator(adj_matrix, N, M, w_type, num_iterations, conspirator_bool, conspirators, true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs, counter1, true_hypothesis, neighbor_prob, neighbor_number):
    """Simulates the network over one initialization

    Args:
        adj_matrix: The graph network as a np.array
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network
        true_hypothesis (int): index of the one true hypothesis
        num_iterations (int): number of iterations for one initialization
        cap (float): maximum signal value
        sociopaths (nd-array): 1d-array of indices for the sociopaths
        conspirators (nd-array): 1d-array of indices for the conspirators
        sociopath_bool (bool): turn on or off sociopaths
        conspirator_bool (bool): turn on or off conspirators
        mega_node_bool (bool): if True, the graph will have a mega node.
        mega_node_beliefs (nd-array sjape=(M)): The beliefs of the mega node.

    Returns:
        private_belief_history (num_iterationsxNxM-array): the resulting private belief vectors for each time-step
        public_belief_history (num_iterationsxNxM-array): the resulting public belief vectors for each time-step
        T_history (num_iterations-array): the resulting truthfulness for each time-step
        C_history (num_iterations-array): the resulting cognitive dissonance for each time-step
    """
    
    private_beliefs = initialize_beliefs(N, M)
    public_beliefs_0 = private_beliefs.copy()

    private_belief_history = np.empty((num_iterations + 1, N, M), dtype=np.float64)
    public_belief_history = np.empty((num_iterations + 1, N, M), dtype=np.float64)
    T_private_history = np.empty(num_iterations + 1, dtype=np.float64)
    T_public_history = np.empty(num_iterations + 1, dtype=np.float64)
    C_history = np.empty(num_iterations + 1, dtype=np.float64)
    
    private_belief_history[0] = private_beliefs
    public_belief_history[0] = public_beliefs_0
    # T_private_0, C_0 = calculate_metrics(N, M, private_beliefs, public_beliefs_0, true_hypothesis)
    # T_public_0, _ = calculate_metrics(N, M, public_beliefs_0, private_beliefs, true_hypothesis)
    
    # T_private_history[0] = T_private_0
    # T_public_history[0] = T_public_0
    # C_history[0] = C_0

    counter2 = 0

    for i in range(1, num_iterations+1):

        likelihoods = get_likelihoods(N, M, true_hypothesis)

        if true_mega_node_bool:
            private_beliefs[0] = true_mega_node_beliefs
        if consp_mega_node_bool:
            private_beliefs[-1] = consp_mega_node_beliefs

        private_beliefs, public_beliefs = update_beliefs(N, M, private_beliefs, w_type, adj_matrix, likelihoods, counter1, counter2, conspirator_bool, conspirators, true_mega_node_bool, 
                                                         consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs, neighbor_prob, neighbor_number)
        private_belief_history[i] = private_beliefs
        public_belief_history[i] = public_beliefs

        T_private, C = calculate_metrics(N, M, private_beliefs, public_beliefs, true_hypothesis)
        T_public, _ = calculate_metrics(N, M, public_beliefs, private_beliefs, true_hypothesis)
        # T_private_history[i] = T_private
        # T_public_history[i] = T_public
        # C_history[i] = C
        # print(f"Iteration {i}: T type: {type(T)}, C type: {type(C)}")

        counter2 += 1

    return private_belief_history, public_belief_history, T_private_history, T_public_history, C_history



def run_simulations2(N, M, graph_func, w_type, num_conspirators, savefile="none", num_simulations=200, num_iterations=150, true_mega_node_beliefs=None, consp_mega_node_beliefs=None, neighbor_prob=1, neighbor_number=5, seed = int(time.time()), **graph_vars):
    
    # ----- Define some parameters
    true_hypothesis = M-1

    if true_mega_node_beliefs is None:
        true_mega_node_beliefs = np.zeros(M, dtype=np.float64)
        true_mega_node_bool = False
    else:
        true_mega_node_bool = True

    if consp_mega_node_beliefs is None:
        consp_mega_node_beliefs = np.zeros(M, dtype=np.float64)
        consp_mega_node_bool = False
    else:
        consp_mega_node_bool = True

    # if true_mega_node_bool and consp_mega_node_bool:
    #     p1 = 0.01
    #     p2 = 0.01
    #     p3 = 0.01
    #     p4 = 1-p1-p2-p3
    #     true_mega_node_beliefs = np.array([p1, p2, p3, p4], dtype=np.float64)
    #     consp_mega_node_beliefs = np.array([p4, p3, p2, p1], dtype=np.float64)

    # Create conspirators
    if num_conspirators == 0:
        conspirators = np.array([], dtype=np.int64)
        conspirator_bool = False
    else:
        random_agents = np.random.permutation(N)[:N].astype(np.int64)
        conspirators = random_agents[0:num_conspirators]
        conspirator_bool = True

    # Create adjacency matrix
    adj_matrices = create_graphs(num_simulations, N, seed, graph_func=graph_func, **graph_vars) 
    
    # Initialise arrays
    private_belief_histories = np.zeros((num_simulations, num_iterations+1, N, M), dtype=np.float64)
    public_belief_histories = np.zeros_like(private_belief_histories)
    counter1 = 0

    for i in tqdm.tqdm(range(num_simulations), desc="Running simulations", position=0, leave=True):

        # Create network graph
        adj_matrix = adj_matrices[i]

        # Simulate belief updates
        private_belief_history, public_belief_history, _, _, _ = simulator(adj_matrix, N, M, w_type, num_iterations, conspirator_bool, conspirators, true_mega_node_bool, consp_mega_node_bool, 
                                                                           true_mega_node_beliefs, consp_mega_node_beliefs, counter1, true_hypothesis, neighbor_prob, neighbor_number)
        # Store results
        private_belief_histories[i] = private_belief_history
        public_belief_histories[i] = public_belief_history

        counter1 += 1

    if savefile != "none":    
        np.savez_compressed(savefile,
            private=private_belief_histories,
            public=public_belief_histories) 
    
        

    return private_belief_histories, public_belief_histories