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

# ----- Setup for 2D grid -----
# N = 144 # must be a perfect square
# seed = 42
# net = create_graphs(1, N, seed, create_2d_grid_network)[0]
# adj = np.array(net>0).astype(int)
# L = int(np.sqrt(N))
# G = nx.grid_2d_graph(L, L, periodic=True)
# #conspirators = [37, 38, 30, 19, 20, 12, 13]    #-- 2 clusters, N=49
# #conspirators = [49, 50, 41, 42, 28, 29, 20, 21] #-- 3 clusters, N=49
# #conspirators = [110, 111, 98, 99, 50, 51, 38, 39, 80, 81, 68, 69] # -- 4 clusters, N = 144
# #conspirators = [110, 111, 98, 99, 50, 51, 38, 39, 80, 81, 68, 69, 54, 53] # -- 3 clusters, N = 144
# conspirators = np.random.randint(0, N, size=int(N/10))

# ----- Setup for 2D crossed grid -----
# N = 64 # must be a perfect square
# seed = 42
# net = create_graphs(1, N, seed, create_2d_crossed_grid)[0]
# adj = np.array(net>0).astype(int)
# G = create_2d_crossed_grid(N, seed)
# # conspirators = np.random.randint(0, N, size=int(N/10))
# #conspirators = np.array([16, 31, 33, 44, 40, 52, 37, 49, 54, 102, 103, 114, 115])
# #conspirators = np.array([45, 69, 93, 43, 91, 41, 65, 67, 89])
# #conspirators = np.array([42, 45, 65, 67, 68, 70, 90, 93])
# conspirators = np.array([8, 10, 17, 23, 28, 29, 36, 37, 41, 47, 52, 53])

# ----- Setup for BA network -------
N = 15
seed = 27
#seed = int(time.time())
net = create_graphs(1, N, seed, create_erdos_renyi_network, k=5)[0]
adj = np.array(net>0).astype(int)
G =  nx.erdos_renyi_graph(N, M, seed=seed)
conspirators = np.array([2, 12])
#conspirators = np.random.randint(0, N, size=int(N*0.3))
#conspirators = np.array([], dtype=np.int64)

# ---- Initialise trust matrices and beliefs ----
S0 = np.random.rand(N, N) * adj
S = np.copy(S0)
T = np.copy(S)
private_beliefs = initialize_beliefs(N, M)
public_beliefs = private_beliefs.copy()

# ---------- Run ----------
numiter = 50

t1 = time.time()
for i in range(numiter): 
    # T_new, S_new = update_trust_2(adj, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta, trust_threshold=trust_threshold)
    likelihoods = get_likelihoods(N, M, true_hypothesis)
    flex = get_flexibilities(N, flex_strength=0.8, flex_interval=None)
    public_beliefs = update_public_beliefs(N, M, private_beliefs, public_beliefs, likelihoods, flex, False, False, 
                                               np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), True, conspirators, "linear")
    q_prev = np.copy(private_beliefs)
    T_new, S_new = update_trust_2(adj, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta, trust_threshold=trust_threshold)
    private_beliefs = update_private_beliefs(N, M, q_prev, "trust", adj, T_new, public_beliefs, private_beliefs, flex, 0, 0, 
                                                 True, conspirators, False, False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), 1, 100, "linear")
    # private_beliefs, public_beliefs = update_beliefs(N, M, private_beliefs, private_beliefs, public_beliefs, "trust", net, T, likelihoods, flex, 0, 0, True, np.array(conspirators, dtype=np.int64), False, 
    #                                                     False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), 1, N)
    T = T_new
    S = S_new
    

t2 = time.time()
print(t2-t1)
    
# ------ Split into clusters ------
# clusters, cluster_sizes = split_into_clusters(adj, private_beliefs, true_hypothesis)
# num_clusters = len(clusters)
# print(f"Number of clusters: {num_clusters}")
# print(f"Size of clusters: {cluster_sizes}")

# ------ Find avg truthfulness within clusters ------
# avg_truthfulness = np.zeros(num_clusters)
# avg_trust = np.zeros(num_clusters)
# avg_trust_direct = np.zeros(num_clusters)
# for j in range(num_clusters):
#     cluster_ind = list(clusters[j])
#     reverse = np.flip(cluster_ind)
#     T_sub = T[cluster_ind][:, cluster_ind]
#     S_sub = S[cluster_ind][:, cluster_ind]
#     A_sub = adj[cluster_ind][:, cluster_ind]
#     avg_truthfulness[j] = np.average(private_beliefs[cluster_ind, true_hypothesis])
#     avg_trust[j] = np.average(T_sub)
#     avg_trust_direct[j] = np.average(S_sub[A_sub==1])

# print(f"Average truthfulness per cluster: {avg_truthfulness}")
# print(f"Average indirect trust per cluster: {avg_trust}")
# print(f"Average direct trust per cluster: {avg_trust_direct}")

# ----- Plotting -----

#plot_trust(G, T, private_beliefs, params, network_type="2d_grid", conspirators=conspirators)
#plot_trust(G, T, private_beliefs, params, network_type="BA", conspirators=conspirators)

fig, (ax3, ax2, ax1) = plt.subplots(1, 3,  figsize=(14, 4))
#fig, (ax2, ax1) = plt.subplots(1, 2,  figsize=(10, 4))
im1 = ax1.imshow(T, vmin=0, vmax=1, cmap='viridis')
ax1.set_title("Indirect trust")
plt.colorbar(im1, ax=ax1)
im2 = ax2.imshow(S, vmin=0, vmax=1, cmap='viridis')
ax2.set_title("Direct trust")
plt.colorbar(im2, ax=ax2)
im3 = ax3.imshow(adj, vmin=0, vmax=1, cmap="viridis")
ax3.set_title(" Adjacency matrix")
for ax in [ax1, ax2, ax3]:
    ax.set_xticks(np.arange(0, 15))
    ax.set_yticks(np.arange(0, 15))
plt.colorbar(im3, ax=ax3)
plt.tight_layout()
plt.show()
