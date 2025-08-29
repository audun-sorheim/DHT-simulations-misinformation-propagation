import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import numba
import scipy as sp
import inspect
from collections import Counter
from simulation import simulator
from networks import (create_graphs, 
                     create_2d_grid_network, 
                     create_barabasi_albert_network,
                     create_directed_barabasi_albert_graph,
                     create_erdos_renyi_network,
                     create_fully_connected_network,
                     create_price_network,
                     create_stochastic_block_model_network,
                     create_watts_strogatz_network
)
from agents import assign_hypothesis_groups

def run_simulations(adj_matrices, num_simulations, N, M, true_hypothesis, num_iterations, 
                    cap, sociopaths, conspirators, sociopath_bool, conspirator_bool, 
                    true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs, num_groups, D, seed):
    """Runs all simulations for the DHT model. Do NOT run this function in parallel,
    as it will cause perturbations from results run in series.

    Args:
        adj_matrices (ndarray): numpy array version of networkx graph, compatible with numba.
        num_simulations (int): number of simulations to run.
        N (int): Number of agnets.
        M (int): Number of hypotheses.
        true_hypothesis (int): The true hypothesis index, often M-1.
        num_iterations (int): Number of time-steps per simulation.
        cap (float): The maximum signal strenght.
        sociopaths (ndarray, shape(num_sociopaths)): THe indices of the sociopaths in the network.
        conspirators (ndarray, shape(num_conspirators)): The indices of the conspirators in the network.
        sociopath_bool (bool): True if there are sociopaths in the network. False otherwise.
        conspirator_bool (bool): True if there are conspirators in the network. False otherwise.
        mega_node_bool (bool): if True, the graph will have a mega node.
        seed (int): The system's seed.
        mega_node_beliefs (nd-array sjape=(M)): The beliefs of the mega node.

    Returns:
        private_belief_histories (ndarray, shape(num_simulations, num_iterations+1, N, M)): Private beliefs (q)
        public_belief_histories (ndarray, shape(num_simulations, num_iterations+1, N, M)): Public Beliefs (p)
        T_private_histories (ndarray, shape(num_simulations, num_iterations+1, N)): Truthfulness wrt. private beliefs
        T_public_histories (ndarray, shape(num_simulations, num_iterations+1, N)): Truthfulness wrt. public beliefs
        C_histories (ndarray, shape(num_simulations, num_iterations+1, N)): Cognitive dissonance (C)
    """
    local_seed = seed
    private_belief_histories = np.zeros((num_simulations, num_iterations+1, N, M), dtype=np.float64)
    public_belief_histories = np.zeros_like(private_belief_histories)
    T_private_histories = np.zeros((num_simulations, num_iterations+1), dtype=np.float64)
    T_public_histories = np.zeros_like(T_private_histories)
    C_histories = np.zeros_like(T_private_histories)
    counter1 = 0

    for i in tqdm.tqdm(range(num_simulations), desc="Running simulations", position=0, leave=True):

        # Create network graph
        adj_matrix = adj_matrices[i]

        A = assign_hypothesis_groups(N, M, num_groups)
        mu_mat = np.random.randn(N, num_groups, D)  # or D if D ≠ M

        # Simulate belief updates
        private_belief_history, public_belief_history, _, _, _ = simulator(
            adj_matrix, N, M, true_hypothesis, num_iterations, cap, 
            sociopaths, conspirators, sociopath_bool, conspirator_bool, 
            true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs,
            A, mu_mat, counter1
        )
        

        # Store results
        private_belief_histories[i] = private_belief_history
        public_belief_histories[i] = public_belief_history
        # T_private_histories[i] = T_private_history
        # T_public_histories[i] = T_public_history
        # C_histories[i] = C_history

        counter1 += 1

    return private_belief_histories, public_belief_histories, T_private_histories, T_public_histories, C_histories

get_barabasi_scale_plot = False
run = True

if __name__=="__main__":

    if run:

        N = 100
        M = 4
        true_hypothesis = M - 1
        num_sociopaths = int(np.round(0.018*N, 0)) # 1.8% is the fraction of people with antisocial personality disorder according to "Store medisinske leksikon (2025)", 
        # Diagnostic and Statistical Manual of Mental Disorders lists the prevalence of antisocial personality disorder as between 0.2% and 3.3% in the general population.
        num_conspirators = int(np.round(0.1*N, 0)) # 10% is the fraction of people who believe the earth is flat according to POLES 2021 survey.
        num_conpsirators_arr = np.arange(0.01, 0.11, 0.01)
        sociopath_bool = False
        conspirator_bool = False
        true_mega_node_bool = False
        consp_mega_node_bool = False
        # mega_node_beliefs = np.zeros(M, dtype=np.float64)
        # mega_node_beliefs[true_hypothesis] = 1
        p1 = 0.01
        p2 = 0.01
        p3 = 0.01
        p4 = 1-p1-p2-p3
        true_mega_node_beliefs = np.array([p1, p2, p3, p4], dtype=np.float64)
        consp_mega_node_beliefs = np.array([p4, p3, p2, p1], dtype=np.float64)
        num_iterations = 150
        num_simulations = 200
        cap = 1
        k = 10
        # ks = np.arange(1, 5)
        m = 5
        ms = np.arange(1,21)
        num_groups = M//2
        D = 5
        seed = int(time.time())
        # seed = 1742024984
        # np.random.seed(seed)
        # print(seed)

        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_watts_strogatz_network, k=4, p_ws=0.2)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_erdos_renyi_network, 
        #                         p_er=k/(N-1), true_mega_node_bool=true_mega_node_bool, consp_mega_node_bool=consp_mega_node_bool)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_directed_barabasi_albert_graph, m=10)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_barabasi_albert_network, m=5)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_stochastic_block_model_network, N_groups, P)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_2d_grid_network)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_price_network, m=m)
        # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_fully_connected_network)

        # check_bernoulli_distinguishability(M, np.array([(1 + k) / (M + 1) for k in range(M)], dtype=np.float64))


        # for i in tqdm.trange(len(ms)):
            # num_conspirators = int(np.round(num_conpsirators_arr[i]*N, 0))

        random_agents = np.random.permutation(N)[:N].astype(np.int64)
        sociopaths = random_agents[:num_sociopaths] if sociopath_bool else np.array([], dtype=np.int64)
        conspirators = random_agents[num_sociopaths:(num_sociopaths + num_conspirators)] if conspirator_bool else np.array([], dtype=np.int64)

        adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_fully_connected_network)
        print(f"Running {num_simulations} simulations on a BA graph with p=1 \nwith true mega-node={true_mega_node_bool} with beliefs {true_mega_node_beliefs} \nand conspiring mega-node={consp_mega_node_bool} with beliefs {consp_mega_node_beliefs}\n")
        # Call parallelized function
        private_belief_histories, public_belief_histories, _, _, _ = run_simulations(
            adj_matrices, num_simulations, N, M, true_hypothesis, num_iterations, cap, 
            sociopaths, conspirators, sociopath_bool, conspirator_bool, 
            true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs, num_groups, D, seed
        )
        
        np.savez_compressed(f"DHT_k100_BA_directed.npz",
                    private=private_belief_histories,
                    public=public_belief_histories)
        

    if get_barabasi_scale_plot:
        # Parameters
        N = 100000
        m = 3
        seed = 1

        # G = create_directed_barabasi_albert_graph(N, m, seed)
        G = nx.barabasi_albert_graph(N, m, seed)
        degrees = [d for n, d in G.degree()]


        # Step 2: Get in-degrees
        # in_degrees = [d for n, d in G.in_degree()]
        # out_degrees = [d for n, d in G.out_degree()]
        # degrees = in_degrees + out_degrees

        # Step 3: Compute degree distribution
        degree_counts = Counter(degrees)
        # out_degree_counts = Counter(out_degrees)
        # print(in_degree_counts)
        # print(out_degree_counts)
        k = np.array(list(degree_counts.keys()), dtype=np.float64)
        pk = np.array(list(degree_counts.values()), dtype=np.float64)
        pk /= pk.sum()  # Normalize to get probability
        k_fit = np.logspace(0.56, np.log10(max(k)), 100)
        p_fit = k_fit**(-3)
        p_fit *= pk.max() / p_fit.max()

        # Step 4: Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(k, pk, color='blue', s=10, label=r'degree distribution', zorder=4)
        plt.plot(k_fit, p_fit, linestyle='--', color='green', label=r'slope=$\gamma=3$', zorder=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$k$', fontsize=14)
        plt.ylabel(r'$P(k)$', fontsize=14)
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.8)
        plt.tight_layout()
        plt.savefig("ba_undirected_degree_distribution.png", dpi=300)
        plt.show()