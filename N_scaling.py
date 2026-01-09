import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import time
import numba
import scipy as sp
from collections import Counter
from simulation import simulator
from networks import (create_graphs, 
                     create_erdos_renyi_network,
                     create_fully_connected_network,
                     create_price_network,)

from simulation import run_simulations2

N_arr = np.logspace(np.log10(10), np.log10(2000), 10)
print(N_arr)
times = np.zeros(len(N_arr))
M = 4
num_simulations = 200
num_iterations = 150

# ---- Run one time to compile ----
# N = 10
# k = int(N/10)
# p_er=k/(N-1)
# run_simulations2(N, M, create_erdos_renyi_network, "random", 0, 
#                      num_simulations=num_simulations, num_iterations=num_iterations, p_er=p_er)
# # ----

# for i in range(len(N_arr)):
#     N = int(N_arr[i])
#     k = int(N/10)
#     p_er=k/(N-1)
#     t1 = time.time()
#     run_simulations2(N, M, create_erdos_renyi_network, "random", 0, 
#                      num_simulations=num_simulations, num_iterations=num_iterations, p_er=p_er)
#     t2 = time.time()
#     times[i] = t2-t1

# np.savez("n_scaling.npz", times=times, N_arr=N_arr)

def f(x, a, b, c):
    return a*x**2 + b*x + c

def y(x, a, b):
    return a*x + b

def z(x, a, b):
    return a* x*np.log(x) + b

# Plot
data = np.load("n_scaling.npz")
N_arr = data["N_arr"]
times = data["times"]
popt1, pcov1 = sp.optimize.curve_fit(f, N_arr, times)
a, b, c = popt1

plt.plot(N_arr, times, 'o-', label="measured times")
plt.plot(N_arr, f(N_arr, *popt1), 'r--', label=f"{a:.4f}N^2 + {b:.2f}N + {c:.2f}")
plt.xlabel("N")
plt.ylabel("t")
plt.legend()
plt.show()
