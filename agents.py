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

#### THE FUNC>TIONS BELOW ARE NOT USED

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