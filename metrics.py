import numpy as np
import numba

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

# Cannot jit this without adding code which is slower than without jitting
def calculate_median(array):
    """Calculates the median of different quantities in the network.

    Args:
        array (nd-array): The quantity to calculate the median for

    returns:
        (nd-array): The median of the quantity at each time-step
    """
    return np.median(array, axis=0)

def truthfulness(q, true_hypothesis):
    q_truth = q[:, :, :, true_hypothesis]
    T = np.mean(q_truth, axis=(0,2))
    return T

def cognitive_dissonance(q, p, true_hypothesis):
    C_agent = np.abs(q - p)
    C = np.mean(C_agent, axis=(0,2,3))
    return C

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