import numpy as np
import matplotlib.pyplot as plt

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