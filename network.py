import numpy as np
import networkx as nx

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

    for i in range(num_simulations):
        G = graph_func(N, seed=seed + (i+1), **kwargs)
        raw_matrix = np.array(nx.to_numpy_array(G), dtype=np.float64)
        stochastic_matrix = normalize_adj_matrix_to_row_stochastic(raw_matrix)
        adj_matrices.append(stochastic_matrix)

    return np.array(adj_matrices)