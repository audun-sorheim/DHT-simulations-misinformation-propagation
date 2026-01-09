import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import scipy as sp

def create_watts_strogatz_network(N, k, p, seed):
    """Function that creates Watts-Strogatz network

    Args:
        N (int): number of agents
        k (int): number of edges to other edges
        p (float): probability of rewiring edges
        seed (int): the seed of the system

    Returns:
        A Watts-Strogatz network
    """
    return nx.watts_strogatz_graph(N, k=2, p=0.3, seed=seed)

def create_erdos_renyi_network(N, p, seed):
    """Function that creates Erdos-Renyi network

    Args:
        N (int): number of agents
        p (float): probability of edge creation
        seed (int): the seed of the system

    Returns:
        A Erdos-Renyi network
    """
    return nx.erdos_renyi_graph(N, p, seed=seed)

def create_barabasi_albert_network(N, m, seed):
    """Function that creates Barabasi-Albert network

    Args:
        N (int): number of agents
        m (int): number of edges to attach from a new node to existing nodes
        seed (int): the seed of the system

    Returns:
        A Barabasi-Albert network
    """
    return nx.barabasi_albert_graph(N, m, seed=seed)

def create_stochastic_block_model_network(N, N_groups, p, seed):
    """Function that creates a stochastic block model network

    Args:
        N (int): number of agents
        N_groups (int): number of groups in the network
        p (ndarray): Element (r,s) gives the density of edges going from the nodes of group r to nodes of group s. 
                    p must match the number of groups (len(sizes) == len(p)).
        seed (int): the seed of the system

    Returns:
        A stochastic block model network
    """
    sizes = np.array([N//N_groups])*N_groups
    return nx.stochastic_block_model(sizes, p, seed=seed)

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

def initialize_beliefs(N, M):
    """Creates random private belief vectors for all agents in the network, 
    and normalizes them for each agent.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network

    Returns:
        (NxM-array): Random, normalized private belief vectors for all agents.
    """

    private_beliefs = np.random.rand(N, M)
    return private_beliefs/np.sum(private_beliefs, axis=1, keepdims=True)


def initialize_distribution(N, M):
    """Creates a distribution from which the agents will receive signals.

    Args:
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network

    Returns:
        (NxM-array): Signal distribution 
    """

    distribution = np.random.rand(N, M)
    return distribution


def create_deviant_agents(N, num_deviant_agents):
    """If turned on, this function will create deviant agents working against truthseeking in the network.

    Args:
        N (int): Number of agents in the network
        num_deviant_agents (int): Number of deviant agents in the network

    Returns:
        num_deviant_agents-array: Array of indices which are deviant agents
    """

    return np.random.choice(N, num_deviant_agents, replace=False)

# def update_beliefs_decommissioned(M, private_beliefs, G, distribution, true_hypothesis, sociopaths, C_influence, conspirators):
    """For every time-step this function will perform a Bayesian update of the beliefs of the agents in the network.
    1. prepare belief vectors
    2. if turned on, initalize "sociopaths" and "conspirators"
    3. If turned on, update private beliefs according to cognitive dissonance
    4. each agent receives a signal from distribution
    5. each agent performs a Bayesian update based on their neighbors beliefs

    Args:
        M (int): Number of hypotheses in the network
        private_beliefs (MxN-array): private belief vectors for all agents
        G: The graph network
        distribution (MxN-array): Signal distribution for all agents
        true_hypothesis (int): index for the true hypothesis
        sociopaths (1d-array, optional): indices for the "sociopathic" agents. 
        C_influence (1d-array, optional): Influence of cognitive dissonance on each agent. 
        conspirators (1d-array, optional): indices for the conspirating agents. 

    Returns:
        T (float), C (float): Truth and cognitive dissonance of the network
    """

    # public_beliefs = np.copy(private_beliefs)
    private_beliefs[:, true_hypothesis] += distribution[:, true_hypothesis]
    private_beliefs /= np.sum(private_beliefs, axis=1, keepdims=True)

    public_beliefs = distribution*private_beliefs/np.sum(distribution*private_beliefs, axis=1, keepdims=True)

    if isinstance(sociopaths, np.ndarray): # Turn cognitive dissonance effects on or off

        C_agent = np.abs(private_beliefs - public_beliefs)
        C_influence[sociopaths] = 0
        for j in range(M):
            private_beliefs[:, j] += C_influence*C_agent[:, j]  # Adjust the learning rate as needed
        private_beliefs /= np.sum(private_beliefs, axis=1, keepdims=True)  # Normalize the beliefs

    if isinstance(conspirators, np.ndarray):
        conspirator_beliefs = np.zeros(M)
        conspirator_beliefs[0] = 1
        public_beliefs[conspirators] = conspirator_beliefs

    epsilon = 1e-10
    for i, belief in enumerate(public_beliefs):

        neighbor_beliefs = np.array([public_beliefs[j] for j in G.neighbors(i)])
        neighbor_beliefs = np.clip(neighbor_beliefs, epsilon, None)

        sigmoid_factor = 0
        x = np.abs(public_beliefs[i] - neighbor_beliefs)
        weights = 1/(1 + np.exp(sigmoid_factor*(x - 0.5)))
        weights /= np.sum(weights)

        if len(neighbor_beliefs) == 0:
            continue 

        if isinstance(conspirators, np.ndarray) and i in conspirators:
            continue
        else:
            log_sum = 0
            for j in range(len(weights)):
                log_sum += weights[j]*np.log(neighbor_beliefs[j])
            exp_log_sum = np.exp(log_sum)
            private_beliefs[i] = exp_log_sum/np.sum(exp_log_sum, keepdims=True)

    C_agent = public_beliefs - private_beliefs
    for j in range(M):
        private_beliefs[:, j] += C_influence*C_agent[:, j]
    private_beliefs /= np.sum(private_beliefs, axis=1, keepdims=True)

    return private_beliefs, public_beliefs

def update_beliefs(M, private_beliefs, G, distribution, true_hypothesis, cap, sociopaths, C_influence, conspirators):
    """For every time-step this function will perform a LRTU update of the beliefs of the agents in the network.
    1. prepare public and private belief vectors
    2. if turned on, initalize "sociopaths" and "conspirators"
    3. If turned on, update private beliefs according to cognitive dissonance
    4. each agent receives a signal from distribution
    5. each agent performs a LRTU update based on their neighbors beliefs

    Args:
        M (int): Number of hypotheses in the network
        private_beliefs (MxN-array): private belief vectors for all agents
        G: The graph network
        distribution (MxN-array): Signal distribution for all agents
        true_hypothesis (int): index for the true hypothesis
        cap (float): maximum distribution value
        sociopaths (1d-array, optional): indices for the "sociopathic" agents. 
        C_influence (1d-array, optional): Influence of cognitive dissonance on each agent. 
        conspirators (1d-array, optional): indices for the conspirating agents. 

    Returns:
        T (float), C (float): Truth and cognitive dissonance of the network
    """
    epsilon = 1e-10
    scaling_factor = 0

    # public_beliefs = np.copy(private_beliefs)
    LRTU_exp = 1 + scaling_factor*np.abs(np.clip(private_beliefs, epsilon, 1) - np.clip(distribution, epsilon, 1))
    private_beliefs[:, true_hypothesis] += cap*distribution[:, true_hypothesis]
    private_beliefs /= np.sum(private_beliefs, axis=1, keepdims=True)

    public_beliefs = distribution**LRTU_exp*private_beliefs/np.sum(distribution**LRTU_exp*private_beliefs, axis=1, keepdims=True)

    # private_beliefs[:, true_hypothesis] += distribution[:, true_hypothesis]
    # private_beliefs /= np.sum(private_beliefs, axis=1, keepdims=True)
    
    # max_hypothesis = np.argmax(private_beliefs, axis=1) # Finds the index for the most believed hypothesis for all agents
    # agents = np.arange(len(private_beliefs)) # Creates an index for every agent
    # likelihood_ratios = private_beliefs/np.clip(private_beliefs[agents, max_hypothesis].reshape(-1, 1), epsilon, None)
    # public_beliefs = likelihood_ratios
    # public_beliefs /= np.sum(public_beliefs, axis=1, keepdims=True)

    if isinstance(sociopaths, np.ndarray):
        C_influence[sociopaths] = 0

    if isinstance(conspirators, np.ndarray):
        conspirator_beliefs = np.zeros(M)
        conspirator_beliefs[0] = 1
        public_beliefs[conspirators] = conspirator_beliefs

    for i, belief in enumerate(public_beliefs):

        neighbor_beliefs = np.array([public_beliefs[j] for j in G.neighbors(i)])
        neighbor_beliefs = np.clip(neighbor_beliefs, epsilon, None)
        num_neighbors = neighbor_beliefs.shape[0]

        if len(neighbor_beliefs) == 0:
            continue 

        # sigmoid_factor = 0

        # x = np.zeros(num_neighbors)
        # for k in range(num_neighbors):
        #     x[k] = np.linalg.norm(np.abs(public_beliefs[i] - neighbor_beliefs[k]))

        # weights = 2/(1 + np.exp(sigmoid_factor*(x - 0.5))) # Set amplitude to 2 and sigmoidfactor to 0, if weights should be 1
        # weights /= np.sum(weights)

        weights = np.random.rand(len(neighbor_beliefs))
        weights /= sum(weights)

        if isinstance(conspirators, np.ndarray) and i in conspirators:
            continue
        else:
            log_sum = 0
            for j in range(num_neighbors):
                log_sum += weights[j]*np.log(neighbor_beliefs[j])

            # log_sum = np.sum(weights*np.log(neighbor_beliefs), keepdims=True)
            exp_log_sum = np.exp(log_sum)
            private_beliefs[i] = exp_log_sum/np.sum(exp_log_sum, keepdims=True)
            
            # Compute likelihood ratios
            # max_hypothesis = np.argmax(np.sum(neighbor_beliefs, axis=0))
            # likelihood_ratios = neighbor_beliefs / neighbor_beliefs[:, max_hypothesis:max_hypothesis+1]
            
            # # Apply LRTU update
            # private_beliefs[i] *= np.prod(likelihood_ratios, axis=0)
            # private_beliefs[i] /= np.sum(private_beliefs[i])  # Normalize
        
    C_agent = public_beliefs - private_beliefs
    for j in range(M):
        private_beliefs[:, j] += C_influence*C_agent[:, j]
    private_beliefs /= np.sum(private_beliefs, axis=1, keepdims=True)

    return private_beliefs, public_beliefs

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

    T = np.sum(main_beliefs[:, true_hypothesis])/N # Truthfulness
    C_agent = np.abs(main_beliefs - secondary_beliefs) # Cognitive dissonance
    C = np.sum(np.sum(C_agent, axis=1))/(N*M)

    return T, C

def simulator(G, N, M, true_hypothesis, num_iterations, cap, sociopath_bool, conspirator_bool, num_deviant_agents):
    """Simulates the network over one initialization

    Args:
        G: the network
        N (int): Number of agents in the network
        M (int): Number of hypotheses in the network
        true_hypothesis (int): index of the one true hypothesis
        num_iterations (int): number of iterations for one initialization
        cap (float): maximum signal value
        sociopath_bool (bool): turn on or off sociopaths
        conspirator_bool (bool): turn on or off conspirators
        num_deviant_agents (int): number of bad people in the network

    Returns:
        private_belief_history (num_iterationsxNxM-array): the resulting private belief vectors for each time-step
        public_belief_history (num_iterationsxNxM-array): the resulting public belief vectors for each time-step
        T_history (num_iterations-array): the resulting truthfulness for each time-step
        C_history (num_iterations-array): the resulting cognitive dissonance for each time-step
    """
    
    private_beliefs = initialize_beliefs(N, M)
    public_beliefs_0 = np.copy(private_beliefs)
    # deviant_agents = create_deviant_agents()
    distribution = initialize_distribution(N, M)

    private_belief_history = np.empty((num_iterations + 1, N, M))
    public_belief_history = np.empty((num_iterations + 1, N, M))
    T_private_history = np.empty(num_iterations + 1)
    T_public_history = np.empty(num_iterations + 1)
    C_history = np.empty(num_iterations + 1)
    
    private_belief_history[0] = private_beliefs
    public_belief_history[0] = public_beliefs_0
    T_private_0, C_0 = calculate_metrics(N, M, private_beliefs, public_beliefs_0, true_hypothesis)
    T_public_0, _ = calculate_metrics(N, M, public_beliefs_0, private_beliefs, true_hypothesis)

    T_private_history[0] = T_private_0
    T_public_history[0] = T_public_0
    C_history[0] = C_0
    
    C_influence = np.random.rand(N)

    if sociopath_bool: # Turn sociopaths on or off
        sociopaths = np.random.choice(N, num_deviant_agents, replace=False)
    else:
        sociopaths = None

    if conspirator_bool: # Turn conspirators on or off
        conspirators = sociopaths if 'sociopaths' != None else np.random.choice(N, num_deviant_agents, replace=False)
    else:
        conspirators = None

    for i in range(1, num_iterations+1):

        private_beliefs, public_beliefs = update_beliefs(M, private_beliefs, G, distribution, true_hypothesis, cap, sociopaths, C_influence, conspirators)

        private_belief_history[i] = private_beliefs
        public_belief_history[i] = public_beliefs

        T_private, C = calculate_metrics(N, M, private_beliefs, public_beliefs, true_hypothesis)
        T_public, _ = calculate_metrics(N, M, public_beliefs, private_beliefs, true_hypothesis)
        T_private_history[i] = T_private
        T_public_history[i] = T_public
        C_history[i] = C

    return private_belief_history, public_belief_history, T_private_history, T_public_history, C_history

def calculate_mean(array):
    """Calculates the mean of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the mean for

    returns:
        (nd-array): The mean of the quantity at each time-step
    """

    return np.mean(array, axis=0)

def calculate_std(array):
    """Calculates the standard deviation of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the standard deviation for

    returns:
        (nd-array): The standard deviation of the quantity at each time-step
    """

    return np.std(array, axis=0)

def calculate_confidence_interval(array, N, confidence=0.95):
    """_summary_

    Args:
        array (nd-array): _description_
        N (int): _description_
        confidence (float, optional): The confidence level. Defaults to 0.95.
    Returns:
        (nd-array): The Confidence interval of the quantity at each time-step
    """

    z = sp.stats.norm.ppf(1 - (1 - confidence)/2)
    std = calculate_std(array)
    return z*std/np.sqrt(N)

def calculate_median(array):
    """Calculates the median of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the median for

    returns:
        (nd-array): The median of the quantity at each time-step
    """
    return np.median(array, axis=0)

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
    plt.xlabel('Iteration')
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

    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, num_iterations, num_iterations+1), array_history, label='average')
    if std_array is not None:
        plt.fill_between(np.linspace(0, num_iterations, num_iterations+1), array_history-std_array, array_history+std_array, alpha=0.3, label='standard deviation')
    if conf_array is not None:
        plt.fill_between(np.linspace(0, num_iterations, num_iterations+1), array_history-conf_array, array_history+conf_array, alpha=0.3, label='confidence interval')
    if median_array is not None:
        plt.plot(np.linspace(0, num_iterations, num_iterations+1), median_array, linestyle='--', color='black', label='median')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.xlim((-0.5, num_iterations)); plt.ylim((0, np.max(array_history + std_array) + 0.05*np.max(array_history + std_array)))
    plt.grid()
    plt.legend()
    plt.show()

    return None