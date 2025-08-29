import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import numba
import scipy as sp
import inspect
from collections import Counter

def create_watts_strogatz_network(N, k, p_ws, seed, mega_node_bool=False, dir=True):
    """Function that creates Watts-Strogatz network

    Args:
        N (int): number of agents
        k (int): number of edges to other edges
        p_ws (float): probability of rewiring edges
        seed (int): the seed of the system
        mega_node_bool (bool): if True, the graph will have a mega node, default is False
        dir (bool): if True, the graph is directed, default is True

    Returns:
        A Watts-Strogatz network
    """
    G = nx.watts_strogatz_graph(N, k, p_ws, seed=seed, directed=dir)
    if mega_node_bool:
        mega_node = 0
        targets = np.arange(1, N)
        G.add_edges_from((mega_node, t) for t in targets)
        
    return G

def create_fully_connected_network(N, seed, dir=True):
    """Function that creates a fully connected network

    Args:
        N (int): number of agents
        seed (int): the seed of the system
        dir (bool): if True, the graph is directed, default is True

    Returns:
        A fully connected network
    """
    G = nx.complete_graph(N, create_using=nx.DiGraph() if dir else nx.Graph())
    return G

def create_erdos_renyi_network(N, p_er, seed, true_mega_node_bool=False, consp_mega_node_bool=False, dir=True):
    """Function that creates Erdos-Renyi network

    Args:
        N (int): number of agents
        p_er (float): probability of edge creation
        seed (int): the seed of the system
        true_mega_node_bool (bool): if True, the graph will have a mega node pushing truth, default is False
        consp_mega_node_bool (bool): if True, the graph will have a mega node pushing misinformation, default is False
        dir (bool): if True, the graph is directed, default is True

    Returns:
        An Erdos-Renyi network
    """
    G = nx.erdos_renyi_graph(N, p_er, seed=seed, directed=dir)
    if true_mega_node_bool:
        mega_node = 0
        targets = np.arange(1, N)
        G.add_edges_from((mega_node, t) for t in targets)
    if consp_mega_node_bool:
        mega_node = N-1
        targets = np.arange(0, N-1)
        G.add_edges_from((mega_node, t) for t in targets)
        
    return G

def create_barabasi_albert_network(N, m, seed, true_mega_node_bool=False, consp_mega_node_bool=False):
    """Function that creates an undirected Barabasi-Albert network

    Args:
        N (int): number of agents
        m (int): number of edges to attach from a new node to existing nodes
        seed (int): the seed of the system
        true_mega_node_bool (bool): if True, the graph will have a mega node pushing truth, default is False
        consp_mega_node_bool (bool): if True, the graph will have a mega node pushing misinformation, default is False

    Returns:
        A Barabasi-Albert network
    """
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    if true_mega_node_bool:
        mega_node = 0
        targets = np.arange(1, N)
        G.add_edges_from((mega_node, t) for t in targets)
    if consp_mega_node_bool:
        mega_node = N-1
        targets = np.arange(0, N-1)
        G.add_edges_from((mega_node, t) for t in targets)
        
    return G

def create_price_network(N, m, seed):
    """
    Generate a Price model network.

    Parameters:
    - N (int): Total number of nodes in the network.
    - m (int): Number of edges each new node brings.
    - seed (int, optional): Seed for random number generator.

    Returns:
    - G (networkx.DiGraph): Generated Price model network.
    """
    c=m
    gamma=1

    if m < 1 or m >= N:
        raise ValueError("m must satisfy 1 <= m < N")

    # G = nx.DiGraph()

    if m>1:
        G = nx.DiGraph()
        G.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i):
                G.add_edge(i, j)
    else:
        G = nx.DiGraph()
        for node in range(m):
            G.add_edge(0, node+1)

    for new_node in range(m, N):
        G.add_node(new_node)

        existing_nodes = np.array(G.nodes())
        existing_nodes = existing_nodes[existing_nodes != new_node]

        in_degrees = np.array([G.in_degree(n) for n in existing_nodes])
        attachment_probs = (in_degrees + c) ** gamma
        attachment_probs = attachment_probs / attachment_probs.sum()

        # Choose m unique targets based on the attachment probability
        targets = np.random.choice(existing_nodes, size=m, replace=False, p=attachment_probs)

        for target in targets:
            G.add_edge(new_node, target)

    return G


# def create_directed_barabasi_albert_graph(N, m, seed):
#     """Creates a directed Barabási-Albert graph, a scale-free graph network.
#     NB! Does not work with numba jit, because of the use of networkx.
#     Do NOT call this function inside a numba jitted function.
#     And N must be greater than or equal to m. 

#     Args:
#         N (Int): The number of agents in the network in total.
#         m (Int): The number of edges to attach from a new node to existing nodes.
#         seed (int): the seed of the system, not used here, but needed for compatibility with other functions.

#     Returns:
#         G (DiGraph): A directed Barabási-Albert graph.
#     """
#     G = nx.DiGraph()

#     # Start with a small initial connected graph
#     G.add_nodes_from(range(m))
#     for i in range(m):
#         for j in range(i):
#             G.add_edge(i, j)

#     targets = list(G.nodes())  # initial targets for attachment

#     for new_node in range(m, N):

#         G.add_node(new_node)
#         targets = np.random.choice(new_node, size=m, replace=False) # randomly choose m targets for new edges

#         for target in targets:
#             G.add_edge(new_node, target)

#     return G

def _random_subset(seq, m):
    """Helper: Return m unique elements from seq (assumed len(seq) >= m)."""
    selected = set()
    while len(selected) < m:
        x = np.random.choice(seq)
        selected.add(x)
    return list(selected)

def create_directed_barabasi_albert_graph(N, m, seed=None):
    """This function is gathered from the networkx package, and altered to return a directed graph.
    
    Returns a random graph according to the Barabási-Albert preferential
    Attachment model.

    A graph of ``N`` nodes is grown by attaching new nodes each with ``m``
    Edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    N : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If ``m`` does not satisfy ``1 <= m < N``.
    """
    if m < 1 or  m >=N:
        raise nx.NetworkXError("Barabási-Albert network must have m >= 1"
                               " and m < N, m = %d, N = %d" % (m, N))
    if seed is not None:
        np.random.seed(seed)

    # Start with a small initial connected graph
    if m>1:
        G = nx.DiGraph()
        G.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i):
                G.add_edge(i, j)
    else:
        G = nx.DiGraph()
        for node in range(m):
            G.add_edge(0, node+1)

    G.name = "barabasi_albert_graph(%s,%s)"%(N,m)
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[n for n, d in G.in_degree() for _ in range(d)]
    # Start adding the other n-m nodes. The first node is m.
    source=m
    while source<N:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m)
        source += 1
    return G

def create_stochastic_block_model_network(N, N_groups, P, seed):
    """Function that creates a stochastic block model network

    Args:
        N (int): number of agents
        N_groups (int): number of groups in the network
        P (ndarray): Element (r,s) gives the density of edges going from the nodes of group r to nodes of group s. 
                    P must match the number of groups (len(sizes) == len(P)).
        seed (int): the seed of the system

    Returns:
        A stochastic block model network
    """
    sizes = np.array([N//N_groups])*N_groups
    return nx.stochastic_block_model(sizes, P, seed=seed, directed=True)

def create_2d_grid_network(N):
    """Function that creates a 2d square grid network

    Args:
        N (int): number of agents

    Returns:
        A 2d square grid network
    """
    if np.sqrt(N)%1 != 0:
        print("N is not a perfect square.")
    L = int(np.sqrt(N))
    return nx.grid_2d_graph(L, L, periodic=True)

def normalize_adj_matrix_to_row_stochastic(adj_matrix):
    """
    Converts a binary or weighted adjacency matrix into a row-stochastic matrix.

    Args:
        adj_matrix (np.ndarray): shape (N, N), raw adjacency matrix (0s and 1s)

    Returns:
        np.ndarray: row-stochastic version of adj_matrix
    """
    N = adj_matrix.shape[0]
    row_sums = adj_matrix.sum(axis=1, keepdims=True) + 1e-12  # Avoid divide-by-zero
    return adj_matrix / row_sums

def create_graphs(num_simulations, N, seed, graph_func, **kwargs):
    """Creates adjacency matrices for multiple graph simulations.
    
    Args:
        num_simulations (int): Number of simulations
        N (int): Number of agents
        seed (int): Base seed for graph generation
        graph_func (function): The graph creation function to use
        **kwargs: Additional arguments specific to the chosen graph function
        
    Returns:
        np.ndarray: Array of adjacency matrices
    """
    print(f"Creating {num_simulations} graphs with {graph_func.__name__}, with parameters: {kwargs}")

    adj_matrices = []

    for i in tqdm.tqdm(range(num_simulations), desc="creating graphs"):
        G = graph_func(N, seed=seed + (i+1), **kwargs)
        raw_matrix = np.array(nx.to_numpy_array(G), dtype=np.float64)
        stochastic_matrix = normalize_adj_matrix_to_row_stochastic(raw_matrix)
        adj_matrices.append(stochastic_matrix)

    return np.array(adj_matrices)

@numba.jit(nopython=True)
def normalize_each_row_sum(arr, N, M):
    """A function to normalize arrays of shape(N,M) such that the sums of any row is equal to 1.

    Args:
        arr (ndarray shape(N,M)): The array to be normalized
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network

    Returns:
        normalized_arr (ndarray shape(N,M)): The normalized array
    """
    normalized_arr = np.zeros((N, M), dtype=np.float64)  # Create new array

    for n in range(N):
        row_sum = np.sum(arr[n, :]) + 1e-12 # Compute the sum of the row
        normalized_arr[n] = arr[n] / row_sum  # Normalize by row sum

    return normalized_arr

@numba.jit(nopython=True)
def normalize_each_row_L2norm(arr, N, M):
    normalized_arr = np.zeros((N, M), dtype=np.float64)  # Create new array

    for n in range(N):
        row_sum = np.sqrt(np.sum(arr[n, :] ** 2))  # Compute the sum of the row
        if row_sum > 0:  # Avoid division by zero
            for m in range(M):
                normalized_arr[n, m] = arr[n, m] / row_sum  # Normalize by row sum

    return normalized_arr

@numba.jit(nopython=True)
def initialize_beliefs(N, M):
    """Creates random private belief vectors for all agents in the network, 
    and normalizes them for each agent.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network

    Returns:
        (NxM-array): Random, normalized private belief vectors for all agents.
    """

    private_beliefs = np.random.rand(N, M).astype(np.float64)
    private_beliefs_normalized = normalize_each_row_sum(private_beliefs, N, M)
    return private_beliefs_normalized

@numba.jit(nopython=True)
def assign_hypothesis_groups(N, M, num_groups=2):
    group_size = M // num_groups
    A = np.zeros((N, M), dtype=np.int64)

    base_group = np.empty(M, dtype=np.int64)
    for g in range(num_groups):
        for i in range(group_size):
            base_group[g * group_size + i] = g + 1

    for i in range(N):
        perm = np.random.permutation(M)
        for j in range(M):
            A[i, j] = base_group[perm[j]]

    return A

@numba.jit(nopython=True)
def mvn_pdf(x, mean, cov_inv, det_cov):
    D = x.shape[0]
    diff = x - mean
    exponent = -0.5 * np.dot(diff, np.dot(cov_inv, diff))
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * det_cov)
    return norm_const * np.exp(exponent)

@numba.jit(nopython=True)
def generate_means(N, num_groups, D):
    mu_mat = np.random.randn(N, num_groups, D)
    return mu_mat

def gaussian_pdf(x, mean, std):
    """Generates a gaussian pdf to finde the likelihood of the signal belonging to a hypothesis.

    Args:
        x (float): the signal recieved by the agent 
        mean (float): the mean of the signal
        std (float): the standard deviation of the signal

    Returns:
        likelihood (float): the likelihood of the signal belonging to the hypothesis
    """
    exponent = -0.5*((x - mean)/std)**2
    coeff = 1/(std*np.sqrt(2*np.pi))
    return coeff*np.exp(exponent)

def prepare_signals(N, M, true_hypothesis):
    """Prepares the signals for the agents in the network.
    The signals produced by this function are for the normal distribution.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network
        true_hypothesis (Int): Index of the true hypothesis

    Returns:
        signals (nd-array(floats) shape=(N)): The signals for all agents
        means (nd-array(floats) shape=(N,M)): The means for all agents
        stds (nd-array(floats) shape=(N,M)): The standard deviations for all agents
    """
    base_means = np.array([1.1*i for i in range(M)], dtype=np.float64)
    noise = np.random.normal(0.0, 0.1, size=(N,M)).astype(np.float64)
    means = base_means + noise
    stds = np.clip(np.ones((N, M), dtype=np.float64) + np.random.normal(0, 0.1, size=(N, M)).astype(np.float64), 0.6, 1.4)
    signals = np.zeros(N, dtype=np.float64)
    for i in range(N):
        signals[i] = np.random.normal(loc=means[i, true_hypothesis], scale=stds[i, true_hypothesis])
    return signals, means, stds

def generate_likelihoods(N, M, signals, means, stds):
    """Generates the likelihood functions based on the normal distribution for the agents in the network.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network
        signals (nd-array(floats) shape=(N)): The signals for all agents
        means (nd-array(floats) shape=(N,M)): The means for all agents
        stds (nd-array(floats) shape=(N,M)): The standard deviations for all agents

    Returns:
        likelihoods (nd-array(floats) shape=(N,M)): The likelhood functions for all agents and hypotheses
    """
    likelihoods = np.zeros((N,M), dtype=np.float64)
    for i in range(N):
        for k in range(M):
            likelihoods[i,k] = gaussian_pdf(signals[i], means[i,k], stds[i,k])
    
    return likelihoods

def check_bernoulli_distinguishability(M, probabilities, threshold=1e-5):
    """
    Checks if the global distinguishability condition holds for Bernoulli distributions.
    For each pair of hypotheses (k ≠ l), there must exist some agent i such that
    D_KL(Bern(p_k) || Bern(p_l)) > threshold.

    Args:
        probabilities (np.ndarray): shape (M,), the success probabilities for each hypothesis
        threshold (float): minimum required KL divergence

    Raises:
        ValueError: if any pair (k, l) is indistinguishable across all agents
    """
    for k in range(M):
        for l in range(M):
            if k == l:
                continue
            p_k = probabilities[k]
            p_l = probabilities[l]
            if p_k in [0, 1] or p_l in [0, 1]:
                continue  # Skip edge cases
            kl = (
                p_k * np.log(p_k / p_l + 1e-12) +
                (1 - p_k) * np.log((1 - p_k) / (1 - p_l) + 1e-12)
            )
            if kl <= threshold:
                raise ValueError(f"Hypotheses θ_{k} and θ_{l} are not globally distinguishable (KL={kl:.4e}).")

@numba.jit(nopython=True)
def get_likelihoods(N, M, true_hypothesis):
    """Generates the likelihood functions based on the Bernoulli distribution for the agents in the network.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network
        true_hypothesis (Int): Index of the true hypothesis

    Returns:
        likelihoods (nd-array(floats) shape=(N,M)): The likelhood functions for all agents and hypotheses
    """

    likelihoods = np.zeros((N, M), dtype=np.float64)
    probabilities = np.zeros(M, dtype=np.float64)
    for k in range(M):
        probabilities[k] = (1+k)/(M+1)

    for i in range(N):
        X = np.random.binomial(1, probabilities[true_hypothesis])
        for k in range(M):
            likelihoods[i,k] = probabilities[k]**X * (1-probabilities[k])**(1-X)

    return likelihoods

@numba.jit(nopython=True)
def check_global_bernoulli_distinguishability(agent_probs, threshold=1e-6):
    """
    Ensures global distinguishability: For each hypothesis pair (k, l), at least one agent i satisfies
    D_KL(Bern(p_i_k) || Bern(p_i_l)) > threshold

    Args:
        agent_probs (np.ndarray): shape (N, M), each agent's Bernoulli parameters
        threshold (float): Minimum KL divergence to count as distinguishable

    Raises:
        ValueError: if any pair (k, l) fails to be globally distinguishable
    """
    N, M = agent_probs.shape

    for k in range(M):
        for l in range(M):
            if k == l:
                continue

            found = False
            for i in range(N):
                p_k = agent_probs[i, k]
                p_l = agent_probs[i, l]
                if p_k in [0, 1] or p_l in [0, 1]:
                    continue  # Skip degenerate cases

                # Compute KL divergence between Bernoulli(p_k) and Bernoulli(p_l)
                kl = (
                    p_k * np.log(p_k / (p_l + 1e-12)) +
                    (1 - p_k) * np.log((1 - p_k) / (1 - p_l + 1e-12))
                )

                if kl > threshold:
                    found = True
                    break

            if not found:
                raise ValueError(f"No agent can distinguish between θ_{k} and θ_{l}. Global distinguishability violated.")

@numba.jit(nopython=True)
def generate_agent_probs_with_overlaps(N, M, overlap_rate, seed=None):
    """
    Generates a matrix of shape (N, M) where each agent has their own Bernoulli success probabilities.
    Some entries will deliberately overlap to simulate indistinguishability.

    Args:
        N (int): Number of agents
        M (int): Number of hypotheses
        overlap_rate (float): Probability that two hypotheses for an agent are indistinguishable
        seed (int): Random seed

    Returns:
        agent_probs (ndarray shape (N, M)): Agent-specific success probabilities
    """
    if seed is not None:
        np.random.seed(seed)

    agent_probs = np.zeros((N, M))
    base_values = np.linspace(0.01, 0.99, M)

    for i in range(N):
        probs = base_values + np.random.normal(0, 0.05, M)
        probs = np.clip(probs, 0.01, 0.99)

        # Introduce overlap between randomly chosen pairs
        for k in range(M):
            for l in range(k+1, M):
                if np.random.rand() < overlap_rate:
                    probs[l] = probs[k]  # Set equal -> indistinguishable

        agent_probs[i] = probs

    return agent_probs

@numba.jit(nopython=True)
def get_agent_specific_likelihoods(N, M, true_hypothesis, agent_probs):
    """
    Generate likelihoods for each agent with their own Bernoulli distributions (agent-level distinguishability).

    Args:
        N (int): Number of agents
        M (int): Number of hypotheses
        true_hypothesis (int): Index of true hypothesis
        agent_probs (ndarray shape (N, M)): Each agent's success probabilities for each hypothesis

    Returns:
        likelihoods (ndarray shape (N, M)): Likelihoods per agent and hypothesis
    """
    likelihoods = np.zeros((N, M), dtype=np.float64)
    
    for i in range(N):
        X = np.random.binomial(1, agent_probs[i, true_hypothesis])
        for k in range(M):
            p = agent_probs[i, k]
            likelihoods[i, k] = p**X * (1 - p)**(1 - X)
    
    return likelihoods

@numba.jit(nopython=True)
def create_deviant_agents(N, num_deviant_agents):
    """If turned on, this function will create deviant agents working against truthseeking in the network.

    Args:
        N (int): Number of agents in the network
        num_deviant_agents (int): Number of deviant agents in the network

    Returns:
        num_deviant_agents-array: Array of indices which are deviant agents
    """

    indices = np.arange(N, dtype=np.int64)
    np.random.shuffle(indices)
    return indices[:num_deviant_agents]

@numba.jit(nopython=True)
def update_beliefs(N, M, private_beliefs, adj_matrix, likelihood, true_hypothesis, cap, 
                   sociopaths, conspirators, sociopath_bool, conspirator_bool, 
                   true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs,
                   counter1, counter2
                   ):
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
    epsilon = 1e-6
    scaling_factor = 10

    # LRTU_exp = 1 + scaling_factor*np.abs(np.clip(private_beliefs, epsilon, 1) - np.clip(likelihood, epsilon, 1))
    LRTU_exp = 1
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

        neighbor_beliefs = public_beliefs[neighbor_indices]
        neighbor_beliefs = np.clip(neighbor_beliefs, epsilon, None)
        num_neighbors = neighbor_beliefs.shape[0]
        weights = adj_matrix[neighbor_indices, i]
        # weights = np.random.rand(num_neighbors).astype(np.float64)

        # weights = np.zeros(num_neighbors, dtype=np.float64)
        # for k in range(num_neighbors):
        #     weights[k] = np.dot(neighbor_beliefs[k], private_beliefs[i])

        # sigmoid_factor = 4
        # x = np.zeros(num_neighbors, dtype=np.float64)
        # for k in range(num_neighbors):
        #     x[k] = 1 - np.linalg.norm(np.abs(private_beliefs[i] - neighbor_beliefs[k]))
        # weights = 1/(1 + np.exp(sigmoid_factor*(x - 0.5)))

        # weights = np.dot(neighbor_beliefs, private_beliefs[i])

        # if true_mega_node_bool:
        #     weights[0] = np.dot(private_beliefs[i], true_mega_node_beliefs)
        # if consp_mega_node_bool:
        #     weights[0] = np.dot(private_beliefs[i], consp_mega_node_beliefs)
        weights = weights / np.sum(weights)

        if weights.shape[0] != neighbor_beliefs.shape[0]:
            print(f"BAD SHAPE DETECTED: weights.shape = {weights.shape[0]}, np.log(neighbor_beliefs).shape = {np.log(neighbor_beliefs).shape[0]}")
            print(f"Failure at simulation {counter1}, iteration {counter2}, agent {i}.")
            raise ValueError("Expected 1D arrays for dot product")

        if sociopath_bool and i in sociopaths or conspirator_bool and i in conspirators:
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
def calculate_metrics(N, M, main_beliefs, secondary_beliefs, true_hypothesis):
    """Calculates Truthfulness and cognitive dissonance for each agent and returns the values for the whole network.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network
        main_beliefs (MxN-array): main belief vectors for all agents to calculate truthfulness, can be either private or public
        secondary_beliefs (MxN-array): main belief vectors for all agents, can be either private or public, cannot be equal to main_beliefs
        true_hypothesis (int): index for the true hypothesis

    Returns:
        T (float): Truthfulness for the whole network
        C (float): Cognitive dissonance for the whole network
    """

    T = np.sum(main_beliefs[:, true_hypothesis])/N # truthfulness
    C_agent = np.abs(main_beliefs - secondary_beliefs) # Cognitive dissonance
    C = np.sum(np.sum(C_agent, axis=1))/(N*M)

    return T, C

@numba.jit(nopython=True)
def simulator(adj_matrix, N, M, true_hypothesis, num_iterations, cap, 
              sociopaths, conspirators, sociopath_bool, conspirator_bool, 
              true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs, 
              A, mu_mat, counter1
              ):
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
    # deviant_agents = create_deviant_agents()
    # distribution = initialize_distribution(N, M)

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

    # sociopaths = np.zeros(N, dtype=np.int64)
    # conspirators = np.zeros(N, dtype=np.int64)
    # C_influence = np.random.rand(N)

    # if sociopath_bool: # Turn cognitive dissonance effects on or off
    #     sociopath_indices = np.random.permutation(N, dtype=np.int64)
    #     sociopaths = sociopath_indices[:num_deviant_agents]

    # if conspirator_bool: # Turn conspirators on or off
    #     if sociopath_bool:
    #         conspirators = sociopaths
    #     else:
    #         conspirators_indices = np.random.permutation(N, dtype=np.int64)
    #         conspirators = conspirators_indices[:num_deviant_agents]

    counter2 = 0

    for i in range(1, num_iterations+1):

        # signals, means, stds = prepare_signals(M, M, true_hypothesis)
        # likelihoods = generate_likelihoods(N, M, signals, means, stds)
        likelihoods = get_likelihoods(N, M, true_hypothesis)
        # agent_probs = generate_agent_probs_with_overlaps(N, M, overlap_rate=0.5, seed=seed)
        # likelihoods = get_agent_specific_likelihoods(N, M, true_hypothesis, agent_probs)
        group_id = A[i, -1]  # or any index j
        mu = mu_mat[i, group_id - 1, :]  # shape: (M,) or (D,)
        obs = np.zeros((N, D), dtype=np.float64)
        

        if true_mega_node_bool:
            private_beliefs[0] = true_mega_node_beliefs
        if consp_mega_node_bool:
            private_beliefs[-1] = consp_mega_node_beliefs

        private_beliefs, public_beliefs = update_beliefs(
            N, M, private_beliefs, adj_matrix, likelihoods, true_hypothesis, cap, 
            sociopaths, conspirators, sociopath_bool, conspirator_bool, 
            true_mega_node_bool, true_mega_node_beliefs, consp_mega_node_bool, consp_mega_node_beliefs,
            counter1, counter2
        )

        # print(f"Iteration {i}: private_beliefs shape: {private_beliefs.shape}")
        # print(f"Iteration {i}: public_beliefs shape: {public_beliefs.shape}")

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

@numba.jit(nopython=True)
def calculate_mean(array):
    """Calculates the mean of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the mean for

    returns:
        (nd-array): The mean of the quantity at each time-step
    """
    size = array.shape[0]
    return np.sum(array, axis=0)/size

@numba.jit(nopython=True)
def calculate_std(array):
    """Calculates the standard deviation of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the standard deviation for

    returns:
        (nd-array): The standard deviation of the quantity at each time-step
    """
    size = array.shape[0]
    mean = calculate_mean(array)
    return np.sqrt(np.sum((array - mean)**2, axis=0)/size)

@numba.jit(nopython=True)
def calculate_confidence_interval(array, N, confidence=0.95, z=1.96):
    """_summary_

    Args:
        array (nd-array): _description_
        N (int): _description_
        confidence (float, optional): The confidence level. Defaults to 0.95.
        z (float, optional): The confidence constant. Defaults to 1.96 when confidence is 0.95.
    Returns:
        (nd-array): The Confidence interval of the quantity at each time-step
    """

    std = calculate_std(array)
    return z*std/np.sqrt(N)

# @numba.jit(nopython=True) # Cannot jit this without adding code which is slower than without jitting
def calculate_median(array):
    """Calculates the median of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the median for

    returns:
        (nd-array): The median of the quantity at each time-step
    """
    return np.median(array, axis=0)

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

def plot_belief_history(array_history, M, num_iterations, y_label):
    """Plots the belief history of the network.

    Args:
        array_history (ndarray): The belief history of the network
        M (int): Number of hypotheses in the network
        num_iterations (int): Number of iterations for one initialization
        y_label (string): The label for the y-axis

    Returns:
        None
    """

    plt.figure(figsize=(10, 5))
    for i in range(M):
        plt.plot([np.mean(belief[:, i]) for belief in array_history], label=f'Hypothesis {i}')
    plt.xlabel('Iterations (t)')
    plt.ylabel(y_label)
    plt.legend()
    plt.xlim((-0.5, num_iterations)); plt.ylim((0, 1))
    plt.grid()
    plt.show()

    return None

def plot_history(array_history, num_iterations, y_label, std_array=None, conf_array=None, median_array=None):
    """Plots different quantities in the network.

    Args:
        array_history (ndarray): _description_
        num_iterations (int): _description_
        y_label (string): _description_
        std_array (ndarray, optional): _description_. Defaults to None.
        conf_array (ndarray, optional): _description_. Defaults to None.
        median_array (ndarray, optional): _description_. Defaults to None.

    Returns:
        None
    """

    if std_array is None:
        std_array = 0

    plt.figure()
    plt.plot(np.linspace(0, num_iterations, num_iterations+1), array_history)
    if std_array is not None:
        plt.fill_between(np.linspace(0, num_iterations, num_iterations+1), array_history-std_array, array_history+std_array, alpha=0.3)
    if conf_array is not None:
        plt.fill_between(np.linspace(0, num_iterations, num_iterations+1), array_history-conf_array, array_history+conf_array, alpha=0.3)
    if median_array is not None:
        plt.plot(np.linspace(0, num_iterations, num_iterations+1), median_array, linestyle='--', color='black')
    plt.xlabel('Iterations (t)')
    plt.ylabel(y_label)
    plt.xlim((0, num_iterations))
    if std_array is None:
        plt.ylim((0, np.max(array_history) + 0.01*np.max(array_history)))
    else:
        plt.ylim((0, np.max(array_history + std_array) + 0.01*np.max(array_history + std_array)))
    plt.grid()
    plt.show()

    return None

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
        k = N//10
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

        adj_matrices = create_graphs(num_simulations, N, seed, graph_func=create_erdos_renyi_network, 
                                     p_er=k/(N-1), true_mega_node_bool=true_mega_node_bool, consp_mega_node_bool=consp_mega_node_bool)        
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