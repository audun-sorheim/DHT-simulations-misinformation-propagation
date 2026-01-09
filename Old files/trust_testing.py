import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import networkx as nx
from networks import create_barabasi_albert_network, create_graphs, create_2d_grid_network, create_erdos_renyi_network, normalize_adj_matrix_to_row_stochastic
from simulation2 import initialize_beliefs, get_likelihoods, remove_links, update_beliefs
#from trust_simulation import update_beliefs
import time
import matplotlib.colors as mcolors


@njit
def update_trust(A, S, T, beta=0.8):
    """No updating of S (only neighbour interactions)"""
    N = np.shape(A)[0]
    T_new = np.copy(T)

    for i in range(N):
        neighbor_indices = np.where(A[i, :] > 0)[0] 
        j_indices = np.array([j for j in range(N) if j!=i])

        for j in j_indices:
            for l in neighbor_indices:
                T_new[i, j] = S[i, j] + beta*S[i, l]*T[l, j]
        
        #T_new[i] = T_new[i]/np.sum(T_new[i])
    
    return T_new

@njit
def f(diff, alpha, thres):
    """ 
    This function ensures that negative interactions are weighted more strongly than positive interactions, so that trust
    is slow to build but quick to break. If the difference is large, alpha increases sharply above the threshold value.

    diff = difference in beliefs
    alpha = weight of current interaction, (between 0 and 1)
    thres = threshold value of difference (between 0 and 1). If thres=1, this is the same as not using f.
    """
    if diff <= thres:
        return alpha
    
    alpha_new = alpha + (1-alpha)/(1-thres) * (diff-thres)
    return alpha_new



@njit
def update_trust_2(A, S, T, private_beliefs, public_beliefs, alpha, beta, trust_threshold):
    """Updates both T and S
        alpha = strength of current interaction vs. history, beta = strength of neighbors
    """
    N = np.shape(A)[0]
    T_new = np.copy(T)
    S_new = np.copy(S)

    for i in range(N):
        neighbor_indices = np.where(A[i, :] > 0)[0] 
        j_indices = np.array([j for j in range(N) if j!=i])

        for j in j_indices:
            #diff = 1/np.sqrt(2) * np.linalg.norm(np.abs(private_beliefs[i] - public_beliefs[j])) # euclidian distance
            diff = 0.5 * np.sum(np.abs(private_beliefs[i] - public_beliefs[j])) # L1 distance
            alpha_new = f(diff, alpha, trust_threshold) # makes trust easily broken
            S_new[i, j] = (1-alpha_new)*S[i, j] + alpha_new*(1 - diff)
            neighbor_sum = np.dot(S_new[i, neighbor_indices], T[neighbor_indices, j])/len(neighbor_indices) # avg neighbor sum
            T_new[i, j] = min(1, max(0, S_new[i, j] + beta*neighbor_sum)) # bounded between 0 and 1

    return T_new, S_new

##############################################################################

def animate(A, S, T, private_beliefs, public_beliefs):
    """ Creates animation"""
    fig, ax = plt.subplots()
    cax = ax.imshow(T, cmap='viridis') 

    def frame_update(frame, grid):
        nonlocal T, S, private_beliefs, public_beliefs
        T, S = update_trust_2(adj, S, T, private_beliefs, public_beliefs, alpha=1)
        likelihoods = get_likelihoods(N, M, true_hypothesis)
        private_beliefs, public_beliefs = update_beliefs(N, M, private_beliefs, "random", net, likelihoods, 0, 0, False, np.array([], dtype=np.int64), False, 
                                                         False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), 1, N)
        cax.set_array(T)
        return cax,

    ani = FuncAnimation(fig, frame_update, frames=10, interval=500, fargs=(T,), blit=False)
    plt.show()

def animate_network(N, M, G, A, S, T, alpha, beta, trust_threshold=1):
    """Animates trust dynamics on a NetworkX graph.
    Edge colours show T_ij (trust from i → j),
    Node colours show average trust received."""

    # --- BA network ---
    # if isinstance(G, np.ndarray):
    #     G = nx.from_numpy_array(G, create_using=nx.DiGraph())
    # pos = nx.spring_layout(G, seed=42)
    
    # --- 2D grid ---
    pos = {(i, j): (i, j) for i, j in G.nodes()}
    G = nx.convert_node_labels_to_integers(G)
    pos = {n: (n % int((np.sqrt(A.shape[0]))), n // int((np.sqrt(A.shape[0])))) for n in range(G.number_of_nodes())}
    # ---------------
    private_beliefs = initialize_beliefs(N, M)
    public_beliefs = private_beliefs.copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sm1 = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm1, ax=ax1, fraction=0.046, pad=0.04, label='Trust')
    fig.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04, label='Private belief')
    ax1.set_title(f"Trust")
    ax2.set_title(f"Belief in true hypothesis")

    true_hypothesis = M-1
    edges = list(G.edges())
    
    def frame_update(frame):
        nonlocal T, S, private_beliefs, public_beliefs

        for n in range(10):
            # Update trust and beliefs
            T, S = update_trust_2(A, S, T, private_beliefs, public_beliefs, alpha, beta, trust_threshold)
            likelihoods = get_likelihoods(N, M, true_hypothesis) # signal

            # ---- Function from trust_simuation ----
            # Leads to polarisation
            # private_beliefs, public_beliefs = update_beliefs(
            #     N, M, private_beliefs, "random", net, T, likelihoods,
            #     0, 0, True, np.array([16, 17, 36, 37, 30, 23], dtype=np.int64), False,
            #     False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64),
            #     1, N
            # )

            # ---- Audun's old function ----
            private_beliefs, public_beliefs = update_beliefs(
                N, M, private_beliefs, "random", net, likelihoods,
                0, 0, True, np.array([16, 17, 36, 37, 30, 23], dtype=np.int64), False,
                False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64),
                1, N
            )

        ax1.clear()
        ax2.clear()
 
        # ---------- Left subplot (trust) -------------
        # Node and edge colouring
        node_values = np.mean(T, axis=0)
        edge_colors = [(T[i, j] + T[j, i]) / 2 for i, j in edges]

        nx.draw_networkx_nodes(
            G, pos, ax=ax1,
            node_color=node_values,
            cmap='plasma',
            vmin=0, vmax=1,
            node_size=300
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax1,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
            edge_vmin=0, edge_vmax=1,
            arrows=True,
            width=1.5
        )

        nx.draw_networkx_labels(G, pos, ax=ax1, font_color='black')
 
        # --------- Right subplot (truthfulness) ---------
        node_values_belief = private_beliefs[:, true_hypothesis]

        nx.draw_networkx_nodes(
            G, pos, ax=ax2,
            node_color=node_values_belief, cmap='coolwarm',
            vmin=0, vmax=1, node_size=300
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax2,
            edge_color='black', arrows=True, width=1
        )
        nx.draw_networkx_labels(G, pos, ax=ax2, font_color='black')
        

        return ax1, ax2

    ani = FuncAnimation(fig, frame_update, frames=10, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

def animate_network_directed(G, A, S, T, private_beliefs, public_beliefs):
    """Animates directional trust dynamics on a NetworkX DiGraph.
    Edge colours show T_ij (trust from i → j),
    Node colours show average trust received, with curved arrows for bidirectional trust."""

    # Convert adjacency matrix to directed graph if needed
    if isinstance(G, np.ndarray):
        G = nx.from_numpy_array(G, create_using=nx.DiGraph())

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots()

    def frame_update(frame):
        nonlocal T, S, private_beliefs, public_beliefs

        # Update trust and beliefs
        T, S = update_trust_2(A, S, T, private_beliefs, public_beliefs, alpha=1)
        likelihoods = get_likelihoods(N, M, true_hypothesis)
        private_beliefs, public_beliefs = update_beliefs(
            N, M, private_beliefs, "random", net, likelihoods,
            0, 0, False, np.array([], dtype=np.int64), False,
            False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64),
            1, N
        )

        ax.clear()
        ax.set_axis_off()

        # Node colours: average trust received
        node_values = T.mean(axis=0)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_values,
            cmap='plasma',
            vmin=0, vmax=1,
            node_size=600
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_color='black')

        # --- Draw curved directed edges ---
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if i != j and T[i, j] > 0:
                    color = plt.cm.viridis(T[i, j])
                    start, end = pos[i], pos[j]

                    # Compute curvature direction based on index ordering
                    rad = 0.2 if i < j else -0.4

                    # Draw curved arrow (arc3 connectionstyle gives a nice bend)
                    ax.annotate(
                        '', xy=end, xytext=start,
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color=color,
                            lw=2.2,
                            alpha=0.9,
                            shrinkA=15, shrinkB=15,
                            mutation_scale=10,
                            connectionstyle=f'arc3,rad={rad}'
                        ),
                        xycoords='data', textcoords='data'
                    )

        return ax,

    ani = FuncAnimation(fig, frame_update, frames=10, interval=1000, blit=False)
    plt.show()

##############################################################################
##############################################################################


# ----- Setup for BA network -------
# N = 40
# seed = 27
# #seed = int(time.time())
# net = create_graphs(1, N, seed, create_barabasi_albert_network, m=5)[0]
# adj = np.array(net>0).astype(int)
# S = np.random.uniform(low = 0, high=1, size=(N, N)) * adj
# #S = np.ones((N, N)) * adj * 0.9
# T = np.copy(S)
# T0 = np.copy(T)

# G =  nx.barabasi_albert_graph(N, M, seed=seed)

# ----- Setup for 2D grid -----
N = 49 # must be a perfect square
k = 10
seed = 27
#seed = int(time.time())
net = create_graphs(1, N, seed, create_2d_grid_network)[0]
#net = create_graphs(1, N, seed, create_erdos_renyi_network, k=k, dir=False)[0]
adj = np.array(net>0).astype(int)

# Different initial conditions
S = np.random.rand(N, N) * adj
#S = np.random.uniform(low = 0, high=0.45, size=(N, N)) * adj
#S = np.ones((N, N)) * adj * 0.1
#S = (np.random.uniform(low = 0, high=0.1, size=(N, N)) + 0.28) * adj

T = np.copy(S)
T0 = np.copy(T)
M = 4 # number of hypotheses
true_hypothesis = M-1
L = int(np.sqrt(N))
G = nx.grid_2d_graph(L, L, periodic=True)

# ----- Run animations ------
M = 4 # number of hypotheses
true_hypothesis = M-1
private_beliefs = initialize_beliefs(N, M)
public_beliefs = private_beliefs.copy()
alpha = 0.5
beta = 0.5
trust_threshold = 0.5

animate_network(N, M, G, adj, S, T, alpha, beta, trust_threshold)

# animate_network_directed(G, adj, S, T, private_beliefs, public_beliefs)

############################################################## RUN ##############################################################
# tol = 1e-4
# maxiter = 10
# alpha = 0.5
# beta = 0.5
# # T = update_trust(adj, S, T) # update once
# T, S = update_trust_2(adj, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta) # update once

# --------------------------------
# for i in range(maxiter): 
#     T_new = update_trust(adj, S, T, beta=0.8)
#     if np.linalg.norm(T-T_new) < tol:
#         print(f"Converged at iteration {i}")
#         break
#     T = T_new

# if i == maxiter-1:
#     print("Maxiter reached")

# t2 = time.time()
# print(f"{t2-t1} seconds")
# ------------------------------------------

# for i in range(maxiter): 
#     if i == 1:
#         t1 = time.time()
#     T_new, S_new = update_trust_2(adj, S, T, private_beliefs, public_beliefs, alpha=alpha, beta=beta)
#     likelihoods = get_likelihoods(N, M, true_hypothesis)
#     private_beliefs, public_beliefs = update_beliefs(N, M, private_beliefs, "random", net, likelihoods, 0, 0, False, np.array([], dtype=np.int64), False, 
#                                                          False, np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64), 1, N)
#     if np.linalg.norm(T-T_new) < tol:
#         print(f"Converged at iteration {i}")
#         break
#     T = T_new
#     S = S_new

# if i == maxiter-1:
#     print("Maxiter reached")

# t2 = time.time()
# print(f"{t2-t1} seconds")

# # --------------------------------------------------------
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(12, 4))
# im1 = ax1.imshow(T0, vmin=0, vmax=1, cmap='viridis')
# ax1.set_title("Initial Network")
# plt.colorbar(im1, ax=ax1)
# im2 = ax2.imshow(T, vmin=0, vmax=1, cmap='viridis')
# ax2.set_title("Updated Network")
# plt.colorbar(im2, ax=ax2)
# im3 = ax3.imshow(T - T0, vmin=-0.2, vmax = 0.2, cmap="seismic")
# ax3.set_title("Difference")
# plt.colorbar(im3, ax=ax3)
# plt.tight_layout()
# plt.show()

