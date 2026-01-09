import numpy as np
import networkx as nx
from trust_functions import *
from networks import create_barabasi_albert_network, create_graphs, create_2d_grid_network, create_2d_crossed_grid
np.random.seed(23)

# ----- Parameters -----
M = 4 
true_hypothesis = M-1
alpha = 1
beta = 0.5
trust_threshold = 1
params = [alpha, beta, trust_threshold]
k=3

# ----- Setup for ER network -------
N = 10
seed = 27
net = create_graphs(1, N, seed, create_erdos_renyi_network, k=k)[0]
adj = np.array(net>0).astype(int)
G =  nx.barabasi_albert_graph(N, M, seed=seed)
conspirator_bool = True
conspirators = np.array([3, 5, 7], dtype=np.int64)

# ---- Initialise trust matrices and beliefs ----
S0 = np.random.rand(N, N) * adj
S = np.copy(S0)
T = np.copy(S)
private_beliefs = initialize_beliefs(N, M)
public_beliefs = private_beliefs.copy()

# ---------- Run ----------
numiter = 60
xticks = np.arange(N)

t1 = time.time()
for i in range(numiter):
    if i % 4 == 0:
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        im1 = ax1.imshow(T, vmin=0, vmax=1, cmap='viridis')
        ax1.set_title(f"Indirect trust")
        ax1.set_xticks(xticks)
        ax1.set_yticks(xticks)
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(S, vmin=0, vmax=1, cmap='viridis')
        ax2.set_title(f"Direct trust")
        ax2.set_xticks(xticks)
        ax2.set_yticks(xticks)
        plt.colorbar(im2, ax=ax2)
        
        if conspirator_bool:
            plt.suptitle(f"Trust matrices at time step {i}, ER-network with <k> = {k}, beta = {beta}, conspirators: {conspirators}")
        else:
            plt.suptitle(f"Trust matrices at time step {i}, ER-network with <k> = {k}, beta = {beta}")
        
        plt.tight_layout()
        plt.savefig(f"trust_adj_{i}.png")

    likelihoods = get_likelihoods(N, M, true_hypothesis)
    flex = get_flexibilities(N, flex_strength=1, flex_interval=None)
    public_beliefs = update_public_beliefs(N, M, private_beliefs, public_beliefs, likelihoods, flex, False, False, 
                                               np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), conspirator_bool, conspirators)
    q_prev = np.copy(private_beliefs)
    T_new, S_new = update_trust_2(adj, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta, trust_threshold=trust_threshold)
    private_beliefs = update_private_beliefs(N, M, q_prev, "trust", adj, T_new, public_beliefs, private_beliefs, flex, 0, 0, 
                                                 conspirator_bool, conspirators, False, False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), 1, 100)

    T = T_new
    S = S_new
    

t2 = time.time()
print(t2-t1)
    

