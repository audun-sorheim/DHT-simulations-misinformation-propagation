import numpy as np
import numba
from metrics import normalize_each_row_sum, gaussian_pdf

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
def get_likelihoods_gaussian(N, M, true_hypothesis, std_draw=0.75, std_likelihood=0.75):
    """
    DHT-style Gaussian likelihoods:
      - Each hypothesis k has a distinct mean (same for all agents)
      - All have the same standard deviation
      - Each agent i draws a signal X_i from the *true hypothesis*
      - Agents compute likelihoods under *all* hypotheses

    Args:
        N (int): number of agents
        M (int): number of hypotheses
        true_hypothesis (int): index of the true hypothesis
        std_draw (float): standard deviation for drawing signals
        std_likelihood (float): standard deviation for likelihood computation

    Returns:
        likelihoods (N, M): per-agent likelihoods
    """
    means = np.linspace(-1.0, 1.0, M)
    likelihoods = np.zeros((N, M))
    # print(means, means[true_hypothesis])
    for i in range(N):
        mu_true = means[true_hypothesis]
        X_i = np.random.normal(mu_true, std_draw)
        for k in range(M):
            likelihoods[i, k] = gaussian_pdf(X_i, means[k], std_likelihood)
    # print(likelihoods[0])
    return likelihoods

@numba.jit(nopython=True)
def confirmation_bias(private_belief, neighbor_beliefs, s=0.6):
    """Calculates weights based on confirmation bias represented as a Gaussian function with peak at 1.

    Args:
        private_belief (nd-array, shape=(4,)): The private beliefs of a given agent i.
        neighbor_beliefs (nd-array, shape=(num_neighbors, 4)): The public beliefs of agent i's neighbors.
        s (float, default=0.6): The adjusting parameter s>0, large s low confirmation bias effect, low s large confirmation bias effect.

    Returns:
        Weights: The unnormalized weights with which agent i will listen to its neighbors.
    """
    diff = np.abs(private_belief - neighbor_beliefs)
    norms = np.sum(diff*diff, axis=1)
    exponent = -norms/s**2
    return np.exp(exponent)

@numba.jit(nopython=True)
def confirmation_bias_1d(private_belief, neighbor_belief, s=0.6):
    """Calculates weights based on confirmation bias represented as a Gaussian function with peak at 1.

    Args:
        private_belief (nd-array, shape=(4,)): The private beliefs of a given agent i.
        neighbor_beliefs (nd-array, shape=(num_neighbors, 4)): The public beliefs of agent i's neighbors.
        s (float, default=0.6): The adjusting parameter s>0, large s low confirmation bias effect, low s large confirmation bias effect.

    Returns:
        Weights: The unnormalized weights with which agent i will listen to its neighbors.
    """
    diff = np.abs(private_belief - neighbor_belief)
    norms = np.sqrt(np.sum(diff*diff))
    exponent = -(norms)**2/s**2
    return np.exp(exponent)

@numba.jit(nopython=True)
def get_flexibilities(N, flex_strength=0.8, flex_interval=None):
    """Generates the flexibilities for the agents in the network.

    Args:
        N (int): Number of agents in the network
        flex_strength (float): The strength of the flexibilities
        flex_interval (list): The interval for the flexibilities

    Returns:
        likelihoods (nd-array(floats) shape=(N,M)): The likelhood functions for all agents and hypotheses
    """
    if flex_interval is not None:
        flexibilities = np.random.uniform(low=flex_interval[0], high=flex_interval[1], size=N).astype(np.float64)
    else:
        flexibilities = np.ones(N, dtype=np.float64) * flex_strength
    return flexibilities