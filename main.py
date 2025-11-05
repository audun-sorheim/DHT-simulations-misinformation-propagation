import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import numba
import scipy as sp
import inspect
import argparse
import os
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
                     create_watts_strogatz_network,
                     create_triangular_grid_network
)
from agents import assign_hypothesis_groups
from metrics import truthfulness, cognitive_dissonance

def run_simulations(
    adj_matrices, 
    num_simulations, 
    N, 
    M, 
    true_hypothesis, 
    num_iterations, 
    confbias_bool=True,
    log_beliefs_bool=False,
    gaussian_bool=True,
    cap=1,
    std_draw=0.5,
    std_likelihood=0.5,
    flex_strength=0.8,
    flex_interval=None,
    sigmoid_factor=4,
    s=0.6,
    conspirators=None, 
    conspirator_bool=False, 
    true_mega_node_bool=False, 
    true_mega_node_beliefs=None, 
    consp_mega_node_bool=False, 
    consp_mega_node_beliefs=None,
    seed=None,
    **kwargs
):
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

        # Simulate belief updates
        private_belief_history, public_belief_history, _, _, _ = simulator(
        adj_matrix, 
        N, 
        M, 
        true_hypothesis, 
        num_iterations, 
        confbias_bool=confbias_bool,
        log_beliefs_bool=log_beliefs_bool,
        gaussian_bool=gaussian_bool,
        cap=cap, 
        std_draw=std_draw,
        std_likelihood=std_likelihood,
        flex_strength=flex_strength,
        flex_interval=flex_interval,
        sigmoid_factor=sigmoid_factor,
        s=s,
        conspirators=conspirators, 
        conspirator_bool=conspirator_bool, 
        true_mega_node_bool=true_mega_node_bool, 
        true_mega_node_beliefs=true_mega_node_beliefs, 
        consp_mega_node_bool=consp_mega_node_bool, 
        consp_mega_node_beliefs=consp_mega_node_beliefs,
        counter1=0
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

def main():
    parser = argparse.ArgumentParser(description="Run DHT simulations with configurable parameters.")

    # === Core parameters ===
    parser.add_argument("--N", type=int, default=100, help="Number of agents (default: 100)")
    parser.add_argument("--M", type=int, default=4, help="Number of hypotheses (default: 4)")
    parser.add_argument("--num_conspirators_frac", type=float, default=0.05, help="Fraction of conspirators (default: 0.05)")
    parser.add_argument("--conspirator_bool", action="store_true", help="Enable conspirators (default: False)")
    parser.add_argument("--true_mega_node_bool", action="store_true", help="Enable true mega-node (default: False)")
    parser.add_argument("--consp_mega_node_bool", action="store_true", help="Enable conspiring mega-node (default: False)")
    parser.add_argument("--num_iterations", type=int, default=150, help="Number of iterations (default: 150)")
    parser.add_argument("--num_simulations", type=int, default=200, help="Number of simulations (default: 200)")
    parser.add_argument("--k", type=float, default=None, help="Average degree parameter (default: 0.1 * N)")
    parser.add_argument("--m", type=int, default=5, help="Number of edges per new node (BA graph only, default: 5)")
    parser.add_argument("--graph", type=str, default="ER", choices=["ER", "BA", "PRICE", "SQUARE", "TRIANGULAR"], help="Graph type: ER, BA, Price, Square, triangular (default: ER)")
    parser.add_argument("--cap", type=float, default=1.0, help="Maximum signal strength (default: 1.0)")

    # === Additional new parameters ===
    parser.add_argument("--sigmoid_factor", type=float, default=4.0, help="Sigmoid factor (default: 4)")
    parser.add_argument("--s", type=float, default=0.6, help="Confirmation bias factor, the standard deviation in a Gaussian function (default: 0.6)")
    parser.add_argument("--std_draw", type=float, default=1.0, help="Std dev for draw (default: 1.0)")
    parser.add_argument("--std_likelihood", type=float, default=1.0, help="Std dev for likelihood (default: 1.0)")
    parser.add_argument("--flex_strength", type=float, default=0.5, help="Flexibility strength (default: 0.5)")
    parser.add_argument("--flex_interval", type=float, nargs=2, default=None, help="Flexibility interval (two floats, e.g. 0.3 0.7)")    
    parser.add_argument("--log_beliefs_bool", action="store_true", help="Enable log-belief mode (default: False)")
    parser.add_argument("--confbias_bool", action="store_true", help="Enable confirmation bias (default: True)")
    parser.add_argument("--gaussian_bool", action="store_true", help="Use Gaussian signals (default: True)")
    parser.add_argument("--save_all", action="store_true", help="Save all data to npz (default: False)")
    parser.add_argument("--dir", type=str, default="test", help="Save results to this folder (default: test)")

    args = parser.parse_args()

    # === Set default boolean values ===
    parser.set_defaults(log_beliefs_bool=False, confbias_bool=True, gaussian_bool=True,
                        conspirator_bool=False, true_mega_node_bool=False, consp_mega_node_bool=False)

    # === Initialize derived parameters ===
    N = args.N
    k = args.k
    num_conspirators = int(np.round(args.num_conspirators_frac * N, 0))

    # === Misc. setup ===
    M = args.M
    true_hypothesis = M - 1
    seed = int(time.time())

    # === Mega-node beliefs ===
    p1 = p2 = p3 = 0.01
    p4 = 1 - p1 - p2 - p3
    true_mega_node_beliefs = np.array([p1, p2, p3, p4], dtype=np.float64)
    consp_mega_node_beliefs = np.array([p4, p3, p2, p1], dtype=np.float64)

    # === Sociopaths / Conspirators ===
    random_agents = np.random.permutation(N)[:N].astype(np.int64)
    conspirators = random_agents[:num_conspirators] if args.conspirator_bool else np.array([], dtype=np.int64)

    if args.flex_interval[0] == 0 and args.flex_interval[0] == 0:
        flex_interval = None
    else:
        flex_interval = np.array([args.flex_interval[0], args.flex_interval[1]], dtype=np.float64)

    # === Graph generation ===
    if args.graph == "ER":
        adj_matrices = create_graphs(
            args.num_simulations, N, seed,
            graph_func=create_erdos_renyi_network,
            p_er=k/(N-1),
            true_mega_node_bool=args.true_mega_node_bool,
            consp_mega_node_bool=args.consp_mega_node_bool
        )
        graph_desc = f"ER_k{k}"
    elif args.graph == "BA":
        adj_matrices = create_graphs(
            args.num_simulations, N, seed,
            graph_func=create_barabasi_albert_network,
            m=args.m
        )
        graph_desc = f"BA_m{args.m}"
    elif args.graph == "PRICE":
        adj_matrices = create_graphs(
            args.num_simulations, N, seed,
            graph_func=create_price_network,
            m=args.m
        )
        graph_desc = f"PRICE"
    elif args.graph == "SQUARE":
        adj_matrices = create_graphs(
            args.num_simulations, N, seed,
            graph_func=create_2d_grid_network
        )
        graph_desc = f"SQUARE"
    elif args.graph == "TRIANGULAR":
        L, K = int(np.sqrt(N)), int(np.sqrt(N))
        adj_matrices = create_graphs(
            args.num_simulations, N, seed,
            graph_func=create_triangular_grid_network,
            K=K, L=L
        )
        print(K, L)
        graph_desc = f"TRIANGULAR"
    else:
        raise ValueError(f"{args.graph} is an invalid graph type, must be 'ER', 'BA' or 'PRICE'.")
    
    print(f"# simulations: {args.num_simulations}   # iterations: {args.num_iterations}")
    print(f"graph-type: {graph_desc}    N: {N}  k: {k}  m: {args.m}")
    print(f"STD_DRAW: {args.std_draw}  STD_LIKELIHOOD: {args.std_likelihood}    s: {args.s}    sigmoid factor: {args.sigmoid_factor}")
    print(f"flexibility strength: {args.flex_strength}    flexibility interval: {args.flex_interval}")
    print(f"log-beliefs: {args.log_beliefs_bool}    confirmation bias: {args.confbias_bool}    gaussian signal: {args.gaussian_bool}")
    print(f"\nRunning {args.num_simulations} simulations on a {args.graph} graph "
          f"with N={N}, k={k}, conspirators={args.conspirator_bool}, "
          f"true_mega_node={args.true_mega_node_bool}, consp_mega_node={args.consp_mega_node_bool}")

    # === Run simulation ===
    private_belief_histories, public_belief_histories, _, _, _ = run_simulations(
        adj_matrices=adj_matrices,
        num_simulations=args.num_simulations,
        N=N,
        M=M,
        true_hypothesis=true_hypothesis,
        num_iterations=args.num_iterations,
        confbias_bool=args.confbias_bool,
        log_beliefs_bool=args.log_beliefs_bool,
        gaussian_bool=args.gaussian_bool,
        cap=args.cap,
        std_draw=args.std_draw,
        std_likelihood=args.std_likelihood,
        flex_strength=args.flex_strength,
        flex_interval=flex_interval,
        sigmoid_factor=args.sigmoid_factor,
        s=args.s,
        conspirators=conspirators,
        conspirator_bool=args.conspirator_bool,
        true_mega_node_bool=args.true_mega_node_bool,
        true_mega_node_beliefs=true_mega_node_beliefs,
        consp_mega_node_bool=args.consp_mega_node_bool,
        consp_mega_node_beliefs=consp_mega_node_beliefs,
        seed=seed
    )

    # === Save output ===
    filename_base = (
        f"DHT_N{N}_{graph_desc}"
        f"{'_logbeliefs' if args.log_beliefs_bool else '_linbeliefs'}"
        f"_gaussian-stds{args.std_draw}_{str(args.flex_strength).replace('.','')}flex_confbias-{args.confbias_bool}"
        f"_T{args.num_iterations}_{args.num_simulations}sims"
    )

    dir = args.dir

    os.makedirs(dir, exist_ok=True)


    if args.save_all:
        # Ensure unique filename
        i = 1
        filename = f"{filename_base}_{i}.npz"
        file_path = os.path.join(dir, filename)
        while os.path.exists(file_path):
            i += 1
            filename = f"{filename_base}_{i}.npz"
            file_path = os.path.join(dir, filename)
        np.savez_compressed(file_path,
                            private=private_belief_histories,
                            public=public_belief_histories)
    else:
        # Ensure unique filename
        i = 1
        filename = f"METRICS_{filename_base}_{i}.npz"
        file_path = os.path.join(dir, filename)
        while os.path.exists(file_path):
            i += 1
            filename = f"METRICS_{filename_base}_{i}.npz"
            file_path = os.path.join(dir, filename)
        np.savez_compressed(file_path,
                            private=private_belief_histories,
                            public=public_belief_histories)
        file_path = os.path.join(dir, filename)
        T0_f = truthfulness(private_belief_histories, 0)
        T1_f = truthfulness(private_belief_histories, 1)
        T2_f = truthfulness(private_belief_histories, 2)
        T3_f = truthfulness(private_belief_histories, 3)
        CD_f = cognitive_dissonance(private_belief_histories, public_belief_histories)
        np.savez_compressed(file_path,
                            T0=T0_f,
                            T1=T1_f,
                            T2=T2_f,
                            T3=T3_f,
                            CD=CD_f)

    if args.num_simulations == 1:
        graph_file_path = os.path.join(dir, "GRAPH-" + filename)
        np.savez_compressed(graph_file_path, adj_matrices)
        print(f"Saved the adjacency matrix to {graph_file_path}")

    print(f"\nGreat Success!  Results saved to: {file_path}\n")


if __name__ == "__main__":
    main()

# if __name__=="__main__":

#     if run:

#         N = 100
#         M = 4
#         true_hypothesis = M - 1
#         num_sociopaths = int(np.round(0.018*N, 0)) # 1.8% is the fraction of people with antisocial personality disorder according to "Store medisinske leksikon (2025)", 
#         # Diagnostic and Statistical Manual of Mental Disorders lists the prevalence of antisocial personality disorder as between 0.2% and 3.3% in the general population.
#         num_conspirators = int(np.round(0.05*N, 0)) # 10% is the fraction of people who believe the earth is flat according to POLES 2021 survey.
#         num_conpsirators_arr = np.arange(0.01, 0.11, 0.01)
#         sociopath_bool = False
#         conspirator_bool = False
#         true_mega_node_bool = False
#         consp_mega_node_bool = False
#         # mega_node_beliefs = np.zeros(M, dtype=np.float64)
#         # mega_node_beliefs[true_hypothesis] = 1
#         p1 = 0.01
#         p2 = 0.01
#         p3 = 0.01
#         p4 = 1-p1-p2-p3
#         true_mega_node_beliefs = np.array([p1, p2, p3, p4], dtype=np.float64)
#         consp_mega_node_beliefs = np.array([p4, p3, p2, p1], dtype=np.float64)
#         num_iterations = 300
#         num_simulations = 500
#         cap = 1
#         k = int(0.1*N)
#         # ks = np.arange(1, 5)
#         m = 5
#         ms = np.arange(1,21)
#         num_groups = M//2
#         D = 5
#         seed = int(time.time())
#         # seed = 1742024984
#         # np.random.seed(seed)
#         # print(seed)

#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_watts_strogatz_network, k=4, p_ws=0.2)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_erdos_renyi_network, 
#         #                         p_er=k/(N-1), true_mega_node_bool=true_mega_node_bool, consp_mega_node_bool=consp_mega_node_bool)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_directed_barabasi_albert_graph, m=10)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_barabasi_albert_network, m=5)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_stochastic_block_model_network, N_groups, P)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_2d_grid_network)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_price_network, m=m)
#         # adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_fully_connected_network)

#         # check_bernoulli_distinguishability(M, np.array([(1 + k) / (M + 1) for k in range(M)], dtype=np.float64))


#         # for i in tqdm.trange(len(ms)):
#             # num_conspirators = int(np.round(num_conpsirators_arr[i]*N, 0))

#         random_agents = np.random.permutation(N)[:N].astype(np.int64)
#         sociopaths = random_agents[:num_sociopaths] if sociopath_bool else np.array([], dtype=np.int64)
#         conspirators = random_agents[num_sociopaths:(num_sociopaths + num_conspirators)] if conspirator_bool else np.array([], dtype=np.int64)

#         adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_erdos_renyi_network, 
#                                 p_er=k/(N-1), true_mega_node_bool=true_mega_node_bool, consp_mega_node_bool=consp_mega_node_bool)
#         print(f"Running {num_simulations} simulations on a ER graph with p={k/(N-1)} \nwith true mega-node={true_mega_node_bool} with beliefs {true_mega_node_beliefs} \nand conspiring mega-node={consp_mega_node_bool} with beliefs {consp_mega_node_beliefs}\n")
#         # Call parallelized function
#         private_belief_histories, public_belief_histories, _, _, _ = run_simulations(
#             adj_matrices, num_simulations, N, M, true_hypothesis, num_iterations, cap, 
#             sociopaths, conspirators, sociopath_bool, conspirator_bool, 
#             true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs, num_groups, D, seed
#         )

#         np.savez_compressed(f"bullshittest_DHT_k100_ER_logbeliefs_gaussian_10flex_uniformweights_T{num_iterations}_{num_simulations}sims.npz",
#                     private=private_belief_histories,
#                     public=public_belief_histories)
        

#     if get_barabasi_scale_plot:
#         # Parameters
#         N = 100
#         m = 3
#         seed = 1

#         # G = create_directed_barabasi_albert_graph(N, m, seed)
#         G = nx.barabasi_albert_graph(N, m, seed)
#         degrees = [d for n, d in G.degree()]


#         # Step 2: Get in-degrees
#         # in_degrees = [d for n, d in G.in_degree()]
#         # out_degrees = [d for n, d in G.out_degree()]
#         # degrees = in_degrees + out_degrees

#         # Step 3: Compute degree distribution
#         degree_counts = Counter(degrees)
#         # out_degree_counts = Counter(out_degrees)
#         # print(in_degree_counts)
#         # print(out_degree_counts)
#         k = np.array(list(degree_counts.keys()), dtype=np.float64)
#         pk = np.array(list(degree_counts.values()), dtype=np.float64)
#         pk /= pk.sum()  # Normalize to get probability
#         k_fit = np.logspace(0.56, np.log10(max(k)), 100)
#         p_fit = k_fit**(-3)
#         p_fit *= pk.max() / p_fit.max()

#         # Step 4: Plot
#         plt.figure(figsize=(8, 6))
#         plt.scatter(k, pk, color='blue', s=10, label=r'degree distribution', zorder=4)
#         plt.plot(k_fit, p_fit, linestyle='--', color='green', label=r'slope=$\gamma=3$', zorder=1)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel(r'$k$', fontsize=14)
#         plt.ylabel(r'$P(k)$', fontsize=14)
#         plt.legend()
#         plt.grid(True, which="both", ls="--", lw=0.8)
#         plt.tight_layout()
#         plt.savefig("ba_undirected_degree_distribution.png", dpi=300)
#         plt.show()