import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
from typing import Sequence
from numba import jit
from numba.typed import List
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from metrics import normalize_each_row_sum
from networks import create_barabasi_albert_network, create_graphs, create_2d_grid_network, create_2d_crossed_grid, create_erdos_renyi_network
from agents import *
from textwrap import wrap
import time
import os


# ------------------------------------------- Simulations ------------------------------------------- 
@jit(nopython=True, forceobj=False)
def initialize_weights(w_type, neighbor_indices, adj_matrix, trust_matrix, agent, public_beliefs, private_beliefs, true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs):
        epsilon = 1e-12
        neighbor_beliefs = public_beliefs[neighbor_indices]
        num_neighbors = neighbor_beliefs.shape[0]

        if w_type == "uniform":
            weights = adj_matrix[agent, neighbor_indices].astype(np.float64) # Ingen confirmation bias

        elif w_type == "trust":
            weights = trust_matrix[agent, neighbor_indices] 

        elif w_type == "conf_bias":
            weights = confirmation_bias(private_beliefs[agent], neighbor_beliefs) 

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

        # if agent == 3:
        #     print("From initialize_weights")
        #     print(neighbor_indices)
        #     print(private_beliefs[agent])
        #     print(neighbor_beliefs)
        #     print(trust_matrix[agent, neighbor_indices])
        #     print(confirmation_bias(private_beliefs[agent], neighbor_beliefs))
        #     print("---")

        #     print(trust_matrix[agent, neighbor_indices]/np.sum(trust_matrix[agent, neighbor_indices]))
        #     print(confirmation_bias(private_beliefs[agent], neighbor_beliefs)/np.sum(confirmation_bias(private_beliefs[agent], neighbor_beliefs)))
        #     print("====")

        weights = weights / np.sum(weights)
        #neighbor_beliefs = np.clip(neighbor_beliefs, epsilon, 1) # This line matters! Clip at 1e-06 changes dynamics dratically
        
        return weights, neighbor_beliefs

@jit(nopython=True, forceobj=False)
def update_beliefs(N, 
                   M, 
                   private_beliefs, 
                   p_prev, 
                   q_prev, 
                   w_type, 
                   adj_matrix, 
                   trust_matrix, 
                   likelihood, 
                   flexibilities, 
                   counter1, 
                   counter2, 
                   conspirator_bool, 
                   conspirators, 
                   true_mega_node_bool, 
                   consp_mega_node_bool, 
                   true_mega_node_beliefs, 
                   consp_mega_node_beliefs, 
                   neighbor_prob, 
                   neighbor_number
                   ):


    LRTU_exp = 1 # Hva er dette?

    #public_beliefs = likelihood**LRTU_exp*private_beliefs
    public_beliefs = likelihood**LRTU_exp*private_beliefs*flexibilities.reshape(-1, 1) + p_prev*(1 - flexibilities.reshape(-1, 1))
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

        neighbor_indices_temp = np.where(adj_matrix[i, :] > 0)[0]  # Get neighbors from adjacency matrix
        #neighbor_indices_temp = np.where(adj_matrix[:, i] > 0)[0]
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

        weights, neighbor_beliefs = initialize_weights(w_type, neighbor_indices, adj_matrix, trust_matrix, i, public_beliefs, private_beliefs, 
                                                       true_mega_node_bool, consp_mega_node_bool, true_mega_node_beliefs, consp_mega_node_beliefs)


        if weights.shape[0] != neighbor_beliefs.shape[0]:
            print(f"BAD SHAPE DETECTED: weights.shape = {weights.shape[0]}, np.log(neighbor_beliefs).shape = {np.log(neighbor_beliefs).shape[0]}")
            print(f"Failure at simulation {counter1}, iteration {counter2}, agent {i}.")
            raise ValueError("Expected 1D arrays for dot product")

        if conspirator_bool and i in conspirators:
            continue
        else:
            # --- Audun exp
            # log_sum = np.dot(weights, np.log(neighbor_beliefs))
            # exp_log_sum = np.exp(log_sum)*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])
            # private_beliefs[i] = exp_log_sum/np.sum(exp_log_sum, axis=0)

            # --- Audun lineær
            # num_neighbors = len(neighbor_beliefs)
            # weights = weights / num_neighbors # Trenger ikke normalisere egentlig, så lenge flex fikses
            # update_sum = np.dot(weights, neighbor_beliefs)*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])
            # private_beliefs[i] = update_sum/np.sum(update_sum, axis=0) # Dette er vel feil

            # --- Riazi exp + flex
            log_sum = np.dot(weights, np.log(neighbor_beliefs))
            # print(log_sum)
            # print(np.exp(log_sum))
            # log_sum = np.dot(weights, np.log(neighbor_beliefs + 1e-6))
            # print(log_sum)
            # print(np.exp(log_sum))
            # print("---")
            exp_log_sum = np.exp(log_sum)
            private_beliefs[i] = (exp_log_sum/np.sum(exp_log_sum, axis=0))*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i]) # in place updating

            # --- Lineær
            # update_sum = np.dot(weights, neighbor_beliefs)
            # private_beliefs[i] = (update_sum/np.sum(update_sum, axis=0))*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])



    return private_beliefs, public_beliefs


@jit(nopython=True, forceobj=False)
def update_public_beliefs(N, M,
                          private_beliefs,
                          p_prev,
                          likelihood,
                          flexibilities,
                          true_mega_node_bool,
                          consp_mega_node_bool,
                          true_mega_node_beliefs,
                          consp_mega_node_beliefs,
                          conspirator_bool,
                          conspirators,
                          updating_func):

    LRTU_exp = 1 # Hva er dette?
    public_beliefs = likelihood**LRTU_exp*private_beliefs*flexibilities.reshape(-1, 1) + p_prev*(1 - flexibilities.reshape(-1, 1))
    public_beliefs = normalize_each_row_sum(public_beliefs, N, M)

    if true_mega_node_bool:
        public_beliefs[0] = true_mega_node_beliefs
    if consp_mega_node_bool:
        public_beliefs[-1] = consp_mega_node_beliefs

    if conspirator_bool:
        conspirator_beliefs = np.zeros(M, dtype=np.float64)
        
        if updating_func == "linear":
            conspirator_beliefs[0] = 1
        elif updating_func == "log":
            epsilon = 1e-12
            np.clip(conspirator_beliefs, epsilon, 1) # clipping the conspirators
            conspirator_beliefs[0] = 1 - (M-1)*epsilon
        
        public_beliefs[conspirators] = conspirator_beliefs
        private_beliefs[conspirators] = conspirator_beliefs
    
    return public_beliefs


@jit(nopython=True, forceobj=False)
def update_private_beliefs(N, M,
                           q_prev,
                           w_type,
                           adj_matrix,
                           trust_matrix,
                           public_beliefs,
                           private_beliefs,
                           flexibilities,
                           counter1,
                           counter2,
                           conspirator_bool,
                           conspirators,
                           true_mega_node_bool,
                           consp_mega_node_bool,
                           true_mega_node_beliefs,
                           consp_mega_node_beliefs,
                           neighbor_prob,
                           neighbor_number, 
                           updating_func):

    for i, belief in enumerate(public_beliefs):

        neighbor_indices = np.where(adj_matrix[i, :] > 0)[0]

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


        weights, neighbor_beliefs = initialize_weights(
            w_type, neighbor_indices, adj_matrix, trust_matrix,
            i, public_beliefs, private_beliefs,
            true_mega_node_bool, consp_mega_node_bool,
            true_mega_node_beliefs, consp_mega_node_beliefs
        )

        if weights.shape[0] != neighbor_beliefs.shape[0]:
            raise ValueError("Shape mismatch in private update")

        if conspirator_bool and i in conspirators:
            continue
        else:
            # --- Audun exp
            # log_sum = np.dot(weights, np.log(neighbor_beliefs))
            # exp_log_sum = np.exp(log_sum)*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])
            # private_beliefs[i] = exp_log_sum/np.sum(exp_log_sum, axis=0)

            # --- Audun lineær
            # weights = weights / num_neighbors # Trenger ikke normalisere egentlig, så lenge flex fikses
            # update_sum = np.dot(weights, neighbor_beliefs)*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])
            # private_beliefs[i] = update_sum/np.sum(update_sum, axis=0) # Dette er vel feil

            if updating_func == "log":
            # --- Riazi exp + flex
                log_sum = np.dot(weights, np.log(neighbor_beliefs))
                exp_log_sum = np.exp(log_sum)
                private_beliefs[i] = (exp_log_sum/np.sum(exp_log_sum, axis=0))*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])

            elif updating_func == "linear":
            # --- Lineær
                update_sum = np.dot(weights, neighbor_beliefs)
                private_beliefs[i] = (update_sum/np.sum(update_sum, axis=0))*flexibilities[i] + q_prev[i, :]*(1 - flexibilities[i])

    return private_beliefs


@jit(nopython=True, forceobj=False)
def simulator(adj_matrix, N, M, w_type, num_iterations, conspirator_bool, conspirators, counter1, true_hypothesis, params, neighbor_prob, neighbor_number, flex_strength, updating_func):
    alpha, beta, trust_threshold = params

    # ---- Initialise trust matrices and beliefs ----
    A = (adj_matrix > 0).astype(np.int64)
    S = np.random.rand(N, N) * A
    T = np.copy(S)
    private_beliefs = initialize_beliefs(N, M)
    public_beliefs = private_beliefs.copy()

    private_belief_history = np.empty((num_iterations + 1, N, M), dtype=np.float64)
    public_belief_history = np.empty((num_iterations + 1, N, M), dtype=np.float64)
    T_history = np.empty(num_iterations+1, dtype=np.float64)
    S_history = np.empty(num_iterations+1, dtype=np.float64)
    private_belief_history[0] = private_beliefs
    public_belief_history[0] = public_beliefs
    T_history[0] = np.mean(T)
    S_history[0] = np.sum(S)/np.sum(A>0)

    
    counter2 = 0

    for i in range(1, num_iterations+1):
        # T_new, S_new = update_trust_2(A, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta, trust_threshold=trust_threshold)
        likelihoods = get_likelihoods_gaussian(N, M, true_hypothesis)
        flexibilities = get_flexibilities(N, flex_strength=flex_strength, flex_interval=None) # all agents have flexibility 0.8. Set to 1 to turn off.

        if i == 0:
            p_prev = public_belief_history[0]
            q_prev = private_belief_history[0]
        else:
            p_prev = public_belief_history[i-1]
            q_prev = private_belief_history[i-1]
        
        public_beliefs = update_public_beliefs(N, M, private_beliefs, p_prev, likelihoods, flexibilities, False, False, 
                                               np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), conspirator_bool, conspirators, updating_func)

        T_new, S_new = update_trust_2(A, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta, trust_threshold=trust_threshold)
        private_beliefs = update_private_beliefs(N, M, q_prev, w_type, adj_matrix, T_new, public_beliefs, private_beliefs, flexibilities, counter1, counter2, 
                                                 conspirator_bool, conspirators, False, False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), neighbor_prob, neighbor_number, updating_func)
        
        # private_beliefs, public_beliefs = update_beliefs(N, M, private_beliefs, p_prev, q_prev, w_type, adj_matrix, T_new, likelihoods, flexibilities, counter1, counter2, conspirator_bool, conspirators, False, 
        #                                                     False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), neighbor_prob, neighbor_number)
        
        private_belief_history[i] = private_beliefs
        public_belief_history[i] = public_beliefs
        
        T = T_new
        S = S_new
        T_history[i] = np.mean(T)
        S_history[i] = np.sum(S)/np.sum(A>0)

        counter2 += 1

    return private_belief_history, public_belief_history, T_history, S_history, T, S, A



def run_simulations(N, M, graph_func, w_type, num_conspirators, flex_strength=0.8, params=[1.0, 0.5, 1.0], savefile=True, 
                    num_simulations=200, num_iterations=150, folder_name = None, updating_func="linear", neighbor_prob=None, 
                    neighbor_number=None, seed = int(time.time()), **graph_vars):
    
    # ----- Define some parameters
    true_hypothesis = M-1

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
    T_histories = np.zeros((num_simulations, num_iterations+1), dtype=np.float64)
    S_histories = np.zeros((num_simulations, num_iterations+1), dtype=np.float64)
    counter1 = 0

    for i in tqdm.tqdm(range(num_simulations), desc="Running simulations", position=0, leave=True):

        # Create network graph
        adj_matrix = adj_matrices[i]

        # Simulate belief updates
        private_belief_history, public_belief_history, T_history, S_history, _, _, _ = simulator(adj_matrix, N, M, w_type, num_iterations, conspirator_bool, conspirators, counter1, true_hypothesis, params, neighbor_prob, neighbor_number, flex_strength, updating_func)

        # Store results
        private_belief_histories[i] = private_belief_history
        public_belief_histories[i] = public_belief_history
        T_histories[i] = T_history
        S_histories[i] = S_history

        counter1 += 1

    if savefile:   
        if graph_func == create_barabasi_albert_network:
            graph = "BA"

        elif graph_func == create_erdos_renyi_network:
            graph = "ER"
        if w_type == "trust":
            filename = f"N{N}_{w_type}_a{params[0]:.1f}_b{params[1]:.1f}_thr{params[2]:.1f}_numsim{num_simulations}_timesteps{num_iterations}_flex{flex_strength:.1f}_{graph}_consp{num_conspirators}_timeev.npz" 
        else:
            filename = f"N{N}_{w_type}_numsim{num_simulations}_iter{num_iterations}_flex{flex_strength}_{graph}_consp{num_conspirators}_timeev.npz" 
        
        if folder_name is None:
         folder_name = os.getcwd() 
        
        os.makedirs(folder_name, exist_ok=True)
        path = os.path.join(folder_name, filename)

        np.savez_compressed(path,
        private=private_belief_histories,
        public=public_belief_histories,
        avg_indtrust = T_histories,
        avg_dirtrust = S_histories) 
    
        print(f"Saved as {filename}")

    return private_belief_histories, public_belief_histories, T_histories


def run_consp_sim(N, M, w_type, num_simulations, num_iterations, conspirator_fraction, graph_func, params, seed, flex_strength, updating_func = "linear", folder_name = None, **graph_vars):
    consp_length = len(conspirator_fraction)
    q_arr = np.zeros((num_simulations, consp_length, N, M))
    dir_trust_arr = np.zeros((num_simulations, consp_length))
    indir_trust_arr = np.zeros((num_simulations, consp_length))
    adj_matrices = create_graphs(num_simulations, N, seed, graph_func, **graph_vars)


    for simulation in tqdm.tqdm(range(num_simulations), desc="Running simulations", position=0, leave=True):
        adj_matrix = adj_matrices[simulation]
        q_sim = np.zeros((consp_length, N, M))
        dir_trust = np.zeros((consp_length))
        indir_trust = np.zeros((consp_length))

        for i in range(consp_length):
            frac = conspirator_fraction[i]
            num_conspirators = int(N*frac)
        
            # Create conspirators
            if num_conspirators == 0:
                conspirators = np.array([], dtype=np.int64)
                conspirator_bool = False
            else:
                random_agents = np.random.permutation(N)[:N].astype(np.int64)
                conspirators = random_agents[0:num_conspirators]
                conspirator_bool = True

            private_belief_history, public_belief_history, T_history, S_history, _, _, _ = simulator(adj_matrix, N, M, w_type, num_iterations, conspirator_bool, conspirators, 0, M-1, params, 0, 0, flex_strength, updating_func)
            q = private_belief_history[-1]
            S = S_history[-1]
            T = T_history[-1]

            q_sim[i] = q
            dir_trust[i] = S
            indir_trust[i] = T
   
        q_arr[simulation] = q_sim
        dir_trust_arr[simulation] = dir_trust
        indir_trust_arr[simulation] = indir_trust


    if graph_func == create_barabasi_albert_network:
        graph = "BA"

    elif graph_func == create_erdos_renyi_network:
        graph = "ER"

    filename = f"N{N}_{w_type}_a{params[0]:.1f}_b{params[1]:.1f}_thr{params[2]:.1f}_numsim{num_simulations}_iter{num_iterations}_flex{flex_strength:.1f}_{graph}_truth_linear.npz" 
    if folder_name is None:
         folder_name = os.getcwd() 
        
    os.makedirs(folder_name, exist_ok=True)
    
    fullfile = os.path.join(folder_name, filename)
    np.savez_compressed(fullfile, conspirator_frac = conspirator_fraction, private_beliefs = q_arr, 
                        dir_trust = dir_trust_arr,
                        indir_trust = indir_trust_arr)
    
    print(f"Saved as {filename}")

    return q_arr, dir_trust_arr, indir_trust_arr

# ------------------------------------------- Trust updating ------------------------------------------- 

@jit(nopython=True, forceobj=False)
def f(diff, alpha, thres):
    """ 
    This function ensures that negative interactions are weighted more strongly than positive interactions, so that trust
    is slow to build but quick to break. If the difference is large, alpha increases sharply above the threshold value.

    diff = difference in beliefs
    alpha = weight of current interaction, (between 0 and 1)
    thres = threshold value of difference (between 0 and 1). If thres=1, this is the same as not using f.
    """
    if diff <= thres:
        return alpha
    
    alpha_new = alpha + (1-alpha)/(1-thres) * (diff-thres)
    return alpha_new



@jit(nopython=True, forceobj=False)
def update_trust_2(A, S, T, private_beliefs, public_beliefs, alpha, beta, trust_threshold):
    """Updates both T and S
        alpha = strength of current interaction vs. history, beta = strength of neighbors
    """
    N = np.shape(A)[0]
    T_new = np.copy(T)
    S_new = np.copy(S)
    S_norm = np.empty_like(S)

    for i in range(N):
        neighbor_indices = np.where(A[i, :] > 0)[0] 

        # Update S
        strengths = confirmation_bias(private_beliefs[i], public_beliefs[neighbor_indices])
        S_new[i, neighbor_indices] = (1-alpha)*S[i, neighbor_indices] + alpha*strengths

        # Normalise
        row_sum = np.sum(S_new[i])
        if row_sum > 0:
            S_norm[i] = S_new[i]/row_sum
        else:
            S_norm[i] = S_new[i]

        # Precompute next neighbour sums
        neighbor_sum = np.zeros(N)
        if len(neighbor_indices) > 0:
            for j in range(N):
                neighbor_sum[j] = np.dot(S_norm[i, neighbor_indices], T[neighbor_indices, j])

            #neighbor_sum = neighbor_sum / len(neighbor_indices)
            
        # Update T
        for j in range(N):
            if i != j:
                #T_new[i, j] = min(1, max(0, S_norm[i, j] + beta*neighbor_sum[j])) # bounded between 0 and 1
                T_new[i, j] = S_norm[i, j] + beta*neighbor_sum[j]

    return T_new, S_new


def plot_trust(G, T, private_beliefs, params, true_hypothesis=3, network_type="2d_grid", conspirators = None):
    n = int(np.sqrt(T.shape[0]))  # grid size
    node_values_belief = private_beliefs[:, true_hypothesis]

    fig, ax1 = plt.subplots(1, 1, figsize = (12, 9))
    sm1 = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))

    fig.colorbar(sm1, ax=ax1, label='Trust')
    fig.colorbar(sm2, ax=ax1, label='Belief in the true hypothesis')

    # Positioning
    if network_type == "2d_grid":
        # Ensure G is directed
        if not isinstance(G, nx.DiGraph):
            G = nx.DiGraph(G)

        #G = nx.convert_node_labels_to_integers(G)
        #pos = {node: (node % n, node // n) for node in range(G.number_of_nodes())}
        pos = {node: (node[0], node[1]) for node in G.nodes()}

        nx.draw_networkx_nodes(G, pos, ax=ax1,
                           node_color=node_values_belief, cmap='coolwarm',
                           vmin=0, vmax=1, node_size=300)

        node_to_idx = {node: k for k, node in enumerate(G.nodes())}
        
        # Draw edges excluding periodic boundary connections
        for i, j in G.edges():
            # Compute positions
            x_i, y_i = pos[i]
            x_j, y_j = pos[j]

            # Skip edges that wrap around horizontally or vertically
            if abs(x_i - x_j) > 1 or abs(y_i - y_j) > 1:
                continue

            #trust_value = T[i, j]
            trust_value = T[node_to_idx[i], node_to_idx[j]]

            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], ax=ax1,
                                edge_color=[trust_value],
                                edge_cmap=plt.cm.viridis,
                                edge_vmin=0, edge_vmax=1,
                                arrows=True, width=1.5,
                                connectionstyle='arc3,rad=0.25')
    
    elif network_type == "BA" or network_type=="ER":
        pos = nx.spring_layout(G, seed=42)
        edges = list(G.edges())
        edge_colors = [T[i, j] for (i, j) in edges]

        nx.draw_networkx_nodes(
        G, pos, ax=ax1,
        node_color=node_values_belief, cmap='coolwarm',
        vmin=0, vmax=1, node_size=300
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax1,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
            edge_vmin=0, edge_vmax=1,
            arrows=True,
            width=1.5
        )

    #nx.draw_networkx_labels(G, pos, ax=ax1, font_color='black')
    label_map = {node: node_to_idx[node] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=label_map, ax=ax1, font_color='black')


    if conspirators is not None:
        #title = ax1.set_title("\n".join(wrap(f"Network trust and belief in true hypothesis without trust in update function, alpha = {params[0]:.2f}, beta = {params[1]:.2f}, thres = {params[2]:.2f}, conspirators = {conspirators}", 60)))
        title = ax1.set_title("\n".join(wrap(f"Network trust and belief in true hypothesis, alpha = {params[0]:.2f}, beta = {params[1]:.2f}, thres = {params[2]:.2f}, conspirators = {conspirators}", 60)))
    else:
        title = ax1.set_title("\n".join(wrap(f"Network trust and belief in true hypothesis, alpha = {params[0]:.2f}, beta = {params[1]:.2f}, thres = {params[2]:.2f}", 60)))

    plt.tight_layout()
    title.set_y(1.05)
    plt.savefig("trust_" + network_type + f"_{params[0]:.2f}_{params[1]:.2f}_{params[2]:.2f}.png", dpi=300)
    plt.show()

    

# ----------------------------------------------------- Clusters -----------------------------------------------------------

def split_into_clusters(A, private_beliefs, true_hypothesis):
    belief = np.rint(private_beliefs[:, true_hypothesis]).astype(np.int8)

    # --- Sparse edge filtering  ---
    rows, cols = A.nonzero()
    mask = belief[rows] == belief[cols]

    # Build new sparse graph using only same-belief edges
    adj_split = csr_matrix((A[rows[mask], cols[mask]], 
                            (rows[mask], cols[mask])), 
                           shape=A.shape)


    # Run connected components on the filtered graph
    n_components, labels = connected_components(adj_split, directed=False)

    # Group nodes by component
    order = np.argsort(labels)
    split_idx = np.flatnonzero(np.diff(labels[order])) + 1
    clusters = np.split(order, split_idx)

    # Keep clusters larger than one
    clusters = [c for c in clusters if len(c) > 1]
    cluster_sizes = [len(c) for c in clusters]

    return clusters, cluster_sizes

@jit(nopython=True, forceobj=False)
def find_avg_truthfulness_trust(clusters, S, private_beliefs, true_hypothesis):
    num_clusters = len(clusters)
    avg_truthfulness = np.empty(num_clusters, dtype=np.float64)
    avg_trust = np.empty(num_clusters, dtype=np.float64)

    for i in range(num_clusters):
        cluster = clusters[i]
        avg_truthfulness[i] = np.mean(private_beliefs[cluster, true_hypothesis])

        # Compute trust within cluster (both directions)
        S_cluster = S[cluster][:, cluster].ravel()
        avg_trust[i] = np.average(S_cluster[S_cluster > 0]) #wrong?

    return num_clusters, avg_truthfulness, avg_trust

#@jit(nopython=True, forceobj=False)
def cluster_simulator(N, M, adj_matrix, w_type, conspirators, conspirator_bool, params, num_iterations, flex_strength, updating_func):
    true_hypothesis = M-1

    private_belief_history, public_belief_history, _, _, _, S, A = simulator(adj_matrix, N, M, w_type, num_iterations, conspirator_bool, conspirators, 0, true_hypothesis, params, None, None, flex_strength, updating_func)
    private_beliefs = private_belief_history[-1] # at convergence

    # ----- Clusters -----
    # n_components, labels, cluster_sizes = split_into_clusters_jit(A, private_beliefs, true_hypothesis)
    # num_clusters, avg_truth_cluster, avg_trust_cluster = find_avg_truthfulness_trust_jit(labels, n_components, T, private_beliefs, true_hypothesis)
    
    clusters, cluster_sizes = split_into_clusters(A, private_beliefs, true_hypothesis)
    num_clusters, avg_truth_cluster, avg_trust_cluster = find_avg_truthfulness_trust(clusters, S, private_beliefs, true_hypothesis)

    largest_cluster_ind = np.argmax(cluster_sizes)
    largest_cluster_size = cluster_sizes[largest_cluster_ind]
    largest_cluster_truth = avg_truth_cluster[largest_cluster_ind]
    largest_cluster_trust = avg_trust_cluster[largest_cluster_ind]

    return num_clusters, largest_cluster_size, largest_cluster_truth, largest_cluster_trust, private_belief_history, public_belief_history


def run_clusters(N, M, w_type, num_simulations, num_iterations, conspirator_fraction, graph_func, params, seed, flex_strength, updating_func = "linear", folder_name = None, **graph_vars):
    consp_length = len(conspirator_fraction)
    num_clusters_arr = np.zeros((num_simulations, consp_length))
    largest_cluster_size_arr = np.zeros((num_simulations, consp_length))
    largest_cluster_truth_arr = np.zeros((num_simulations, consp_length))
    largest_cluster_trust_arr = np.zeros((num_simulations, consp_length))
    adj_matrices = create_graphs(num_simulations, N, seed, graph_func, **graph_vars)


    for simulation in tqdm.tqdm(range(num_simulations), desc="Running simulations", position=0, leave=True):
        adj_matrix = adj_matrices[simulation]
        num_clusters = np.zeros(consp_length)
        largest_cluster_size = np.zeros(consp_length)
        largest_cluster_truth = np.zeros(consp_length)
        largest_cluster_trust = np.zeros(consp_length)


        for i in range(consp_length):
            frac = conspirator_fraction[i]
            num_conspirators = int(N*frac)
        
            # Create conspirators
            if num_conspirators == 0:
                conspirators = np.array([], dtype=np.int64)
                conspirator_bool = False
            else:
                random_agents = np.random.permutation(N)[:N].astype(np.int64)
                conspirators = random_agents[0:num_conspirators]
                conspirator_bool = True

            num_clusters[i], largest_cluster_size[i], largest_cluster_truth[i], largest_cluster_trust[i], _, _ = cluster_simulator(N, M, adj_matrix, w_type, conspirators, 
                                                                                                                                    conspirator_bool, params, num_iterations, flex_strength, updating_func)
        
        num_clusters_arr[simulation] = num_clusters
        largest_cluster_size_arr[simulation] = largest_cluster_size
        largest_cluster_truth_arr[simulation] = largest_cluster_truth
        largest_cluster_trust_arr[simulation] = largest_cluster_trust

    if graph_func == create_barabasi_albert_network:
        graph = "BA"

    elif graph_func == create_erdos_renyi_network:
        graph = "ER"



    filename = f"N{N}_{w_type}_a{params[0]:.1f}_b{params[1]:.1f}_thr{params[2]:.1f}_numsim{num_simulations}_iter{num_iterations}_flex{flex_strength:.1f}_{graph}_clusters_linear.npz" 
    if folder_name is None:
         folder_name = os.getcwd() 
        
    os.makedirs(folder_name, exist_ok=True)
    
    fullfile = os.path.join(folder_name, filename)
    np.savez_compressed(fullfile, conspirator_frac = conspirator_fraction, num_clusters=num_clusters_arr, 
                        largest_cluster_size=largest_cluster_size_arr, 
                        truthfulness_largest_cluster = largest_cluster_truth_arr, trust_largest_cluster = largest_cluster_trust_arr)
    
    print(f"Saved as {filename}")

    return num_clusters_arr, largest_cluster_size_arr, largest_cluster_truth_arr, largest_cluster_trust_arr


@jit(nopython=True, forceobj=False)
def connected_components_numba(adj_split):
    N = adj_split.shape[0]
    visited = np.zeros(N, dtype=np.uint8)
    labels = np.full(N, -1, dtype=np.int32)

    stack = np.empty(N, dtype=np.int32)
    current_label = 0

    for start in range(N):
        if visited[start] == 1:
            continue
        
        # start DFS
        top = 0
        stack[top] = start
        visited[start] = 1
        labels[start] = current_label

        while top >= 0:
            node = stack[top]
            top -= 1

            row = adj_split[node]
            # iterate neighbours
            for j in range(N):
                if row[j] == 1 and visited[j] == 0:
                    visited[j] = 1
                    labels[j] = current_label
                    top += 1
                    stack[top] = j

        current_label += 1

    return labels, current_label


@jit(nopython=True, forceobj=False)
def split_into_clusters_jit(A, private_beliefs, true_hypothesis):
    N = A.shape[0]

    # belief mask
    belief = np.rint(private_beliefs[:, true_hypothesis]).astype(np.int8)

    # same_belief = belief[:, None] == belief[None, :]
    # adj_split = A * same_belief

    # same-belief adjacency
    adj_split = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if belief[i] == belief[j]:
                adj_split[i, j] = A[i, j]

    # connected components
    labels, n_components = connected_components_numba(adj_split)

    # compute sizes
    cluster_sizes = np.zeros(n_components, dtype=np.int32)
    for i in range(N):
        lab = labels[i]
        if lab >= 0:
            cluster_sizes[lab] += 1
    
    # Must pad this
    #clusters = [np.where(labels == k)[0] for k in range(n_components) if cluster_sizes[k] > 1]

    return n_components, labels, cluster_sizes

@jit(nopython=True, forceobj=False)
def find_avg_truthfulness_trust_jit(labels, num_clusters, T, private_beliefs, true_hypothesis):

    avg_truthfulness = np.empty(num_clusters, dtype=np.float64)
    avg_trust = np.empty(num_clusters, dtype=np.float64)

    for i in range(num_clusters):
        cluster = np.where(labels==i)[0] #indices of cluster
        avg_truthfulness[i] = np.mean(private_beliefs[cluster, true_hypothesis])

        # Compute trust within cluster (both directions)
        trust_vals = T[cluster][:, cluster].ravel()
        avg_trust[i] = np.mean(trust_vals)

    return num_clusters, avg_truthfulness, avg_trust



