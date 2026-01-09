import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import numba
import scipy as sp
import inspect
from collections import Counter
#from simulation2 import run_simulations2
from trust_functions import run_simulations
from networks import (create_graphs, 
                     create_erdos_renyi_network,
                     create_fully_connected_network,
                     create_price_network,
                     create_barabasi_albert_network)

# -------------- Run the simulations ------------
k = 10
N = 100
M = 4
m = int(k/2)
num_simulations = 50
num_iterations = 200
# num_conspirators = int(N*0.2)
flex = 0.8


# --------------------------------- RANDOM CHOICE OF NEIGHBOURS ----------------------------------
# ---- Random neighbours no CB -----
# probs = np.array([0.05, 0.1, 0.2, 0.4, 1])
# #probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
# for prob in probs:
#     #run_simulations2(N, M, create_erdos_renyi_network, "random", 0, f"random_{prob}.npz", p_er=p_er, neighbor_prob=prob)
#     run_simulations2(N, M, create_barabasi_albert_network, "random", 0, f"random_{prob}_ba.npz", m=5, neighbor_prob=prob) # k = 2m

# ---- Random neighbours with CB -----
# probs = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
# for prob in probs:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 0, f"random_CB_{prob}.npz", p_er=p_er, neighbor_prob=prob)

# probs = np.arange(0, 1.05, 0.05)
# for prob in probs:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 0, f"random_CB_{prob}_25timesteps.npz", num_iterations=25, p_er=p_er, neighbor_prob=prob)

# ---- Random neighbours with CB and conspirators -----
# probs = np.array([0.2, 0.4, 0.6, 0.8, 1])
# for prob in probs:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 10, f"random_CB_consp_{prob}.npz", p_er=p_er, neighbor_prob=prob)

# -----------------------------------------------------------------
#------------------------------------------------------------------

# ---- Random neighbours no CB, fixed number of neighbours -----
# num_neigh = np.array([1, 2, 3, 4, 5, 100])
# for num in num_neigh:
#     # run_simulations2(N, M, create_erdos_renyi_network, "random", 0, f"random{num}.npz", p_er=p_er, neighbor_number=num)
#     run_simulations2(N, M, create_barabasi_albert_network, "random", 0, f"random{num}_ba.npz", m=5, neighbor_number=num)

# ---- Random neighbours with CB, fixed number of neighbours -----
# num_neigh = np.array([1, 2, 3, 4, 5, 100])
# for num in num_neigh:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 0, f"random_{num}_CB.npz", p_er=p_er, neighbor_number=num)

# ---- Random neighbours with CB and conspirators, fixed number of neighbours -----
# num_neigh = np.array([1, 2, 3, 4, 5, 100])
# for num in num_neigh:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 10, f"random_{num}_CB_consp.npz", p_er=p_er, neighbor_number=num)

# -----------------------------------------------------------------
#------------------------------------------------------------------

# ---- Random neighbours no CB with cutoff -----
# probs = np.array([0.05, 0.1, 0.2, 0.4, 1])
# probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
# max_neighbours = 2 # maximum number of neighbors
# for prob in probs:
#     #run_simulations2(N, M, create_erdos_renyi_network, "random", 0, f"random_cutoff_{prob}.npz", p_er=p_er, neighbor_prob=prob, neighbor_number=max_neighbours)
#     run_simulations2(N, M, create_price_network, "random", 0, f"random_cutoff_{prob}_price.npz", m=10, neighbor_prob=prob, neighbor_number=max_neighbours)

# ---- Random neighbours with CB -----
# probs = np.array([0.2, 0.4, 0.6, 0.8, 1])
# for prob in probs:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 0, f"random_CD_{prob}.npz", p_er=p_er, neighbor_prob=prob)

# ---- Random neighbours with CB and conspirators -----
# probs = np.array([0.2, 0.4, 0.6, 0.8, 1])
# for prob in probs:
#     run_simulations2(N, M, create_erdos_renyi_network, "sigmoid", 10, f"random_CB_consp_{prob}.npz", p_er=p_er, neighbor_prob=prob)

#run_simulations2(N, M, create_erdos_renyi_network, "random", 0, f"test.npz", p_er=p_er, neighbor_prob=0.5, neighbor_number=2)


# ---------------------------------------- REMOVING A FRACTION OF LINKS -------------------------------------------------------
# ---- Undirected ER-network ----
# probs = np.array([0, 0.5, 0.8, 0.9, 0.95, 0.98])
# for prob in probs:
#     run_simulations2(N, M, create_erdos_renyi_network, "random", 0, f"link_removal_{prob}.npz", k=k, neighbor_prob=prob)

# ---- Undirected BA-network ----
# probs = np.array([0, 0.5, 0.8, 0.9, 0.95, 0.98])
# for prob in probs:
#     run_simulations2(N, M, create_barabasi_albert_network, "random", 0, f"link_removal_BA_{prob}.npz", m=m, neighbor_prob=prob)


# num_simulations = 50
# num_iterations = 200
# num_conspirators = int(N*0.2)
# flex = 0.8
# alpha_arr = np.arange(0.2, 1.2, 0.2)

# for alpha in alpha_arr:
#     run_simulations(N, M, create_barabasi_albert_network, "trust", num_conspirators, flex_strength=flex, 
#                     params = [alpha, 0.5, 1.0], num_simulations=num_simulations, num_iterations=num_iterations, m=5)
    