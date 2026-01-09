from trust_functions import *

# ============================= Parameters ================================
M = 4
flex = 0.8
seed = int(time.time())
# params = [1.0, 0.5, 1.0] # alpha, beta, thres (must be floats)

# ----- Small network
N = 100
m = 5
k = 10

# ---- Large network
# N = 500
# m = 25
# k = 50

# ============================= Time evolution =============================
num_simulations = 100
num_iterations = 250

# ----- Linear updates, BA-network
# beta_range =np.array([0, 0.5, 0.9], dtype=np.float64)
# num_conspirators_arr = [20, 35, 50]
# for num_conspirators in num_conspirators_arr:
#      for beta in beta_range:
#           params = [1.0, beta, 1.0]
#           run_simulations(N, M, create_barabasi_albert_network, "trust", num_conspirators, params=params, 
#                          num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                          folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{num_conspirators}_consp_N100_new")
#      run_simulations(N, M, create_barabasi_albert_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], 
#                     num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{num_conspirators}_consp_N100_new")
     

# ----- Linear updates, ER-network
# beta_range =np.array([0, 0.5, 0.9], dtype=np.float64)
# num_conspirators_arr = [20, 35, 50]
# for num_conspirators in num_conspirators_arr:
#      for beta in beta_range:
#           params = [1.0, beta, 1.0]
#           run_simulations(N, M, create_erdos_renyi_network, "trust", num_conspirators, params=params, 
#                          num_simulations=num_simulations, num_iterations=num_iterations, k=k, 
#                          folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\ER_{num_conspirators}_consp_N100_new")
     
#      run_simulations(N, M, create_erdos_renyi_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], 
#                     num_simulations=num_simulations, num_iterations=num_iterations, k=k, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\ER_{num_conspirators}_consp_N100_new")

# ----- Log updates, BA-network
# beta_range =np.array([0, 0.5, 0.9], dtype=np.float64)
# num_conspirators_arr = [1, 4, 10]
# for num_conspirators in num_conspirators_arr:
#      for beta in beta_range:
#           params = [1.0, beta, 1.0]
#           run_simulations(N, M, create_barabasi_albert_network, "trust", num_conspirators, params=params, updating_func="log",
#                          num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                          folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{num_conspirators}_consp_N100_log_new")
#      run_simulations(N, M, create_barabasi_albert_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], updating_func="log",
#                     num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{num_conspirators}_consp_N100_log_new")
     

# ----- Log updates, BA-network
# num_conspirators_arr = [1, 4, 10]
# beta_range = np.array([0, 0.5, 0.9], dtype=np.float64)
# for num_conspirators in num_conspirators_arr:
#      for beta in beta_range:
#           params = [1.0, beta, 1.0]
#           run_simulations(N, M, create_barabasi_albert_network, "trust", num_conspirators, params=params, 
#                          num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                          folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{num_conspirators}_consp_log_N100_new",
#                          updating_func="log")
          
#      run_simulations(N, M, create_barabasi_albert_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], 
#                     num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{num_conspirators}_consp_log_N100_new",
#                     updating_func="log")


run_simulations(N, M, create_barabasi_albert_network, "uniform", 1, params=[1.0, 0.5, 1.0], 
                         num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
                         folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\BA_{1}_consp_log_N100_CAP_1e-6",
                         updating_func="log")

# ============================ No conspirators =========================
# beta_range =np.array([0, 0.5, 0.9], dtype=np.float64)
# num_conspirators = 0
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_simulations(N, M, create_barabasi_albert_network, "trust", num_conspirators, params=params, 
#                     num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100_linear")
#      run_simulations(N, M, create_erdos_renyi_network, "trust", num_conspirators, params=params, 
#                     num_simulations=num_simulations, num_iterations=num_iterations, k=k, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100_linear")

# run_simulations(N, M, create_barabasi_albert_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], 
#                num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100_linear")

# run_simulations(N, M, create_erdos_renyi_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], 
#                num_simulations=num_simulations, num_iterations=num_iterations, k=k, 
#                folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100_linear")

# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_simulations(N, M, create_barabasi_albert_network, "trust", num_conspirators, params=params, updating_func="log",
#                     num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100")
#      run_simulations(N, M, create_erdos_renyi_network, "trust", num_conspirators, params=params, updating_func="log",
#                     num_simulations=num_simulations, num_iterations=num_iterations, k=k, 
#                     folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100")

# run_simulations(N, M, create_barabasi_albert_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], updating_func="log",
#                num_simulations=num_simulations, num_iterations=num_iterations, m=m, 
#                folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100")

# run_simulations(N, M, create_erdos_renyi_network, "uniform", num_conspirators, params=[1.0, 0.5, 1.0], updating_func="log",
#                num_simulations=num_simulations, num_iterations=num_iterations, k=k, 
#                folder_name = rf"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Time evolution\no_consp_N100")


# ============================= Clustering =============================
# num_simulations = 80
# num_iterations = 300

# ----- Linear updates, BA network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Clusters\Clusters_BA_linear_N100_300timesteps_new"
# consp_frac = np.arange(0, 0.64, 0.04)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_clusters(N, M, "trust", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, params, seed, flex, m=m, folder_name=folder, updating_func="linear")
# run_clusters(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, [1.0, 0.5, 1.0], seed, flex, m=m, folder_name=folder, updating_func="linear")
# run_clusters(N, M, "trust", 20, num_iterations, consp_frac, create_barabasi_albert_network, [1.0, 0.9, 1.0], seed, flex, m=m, folder_name=folder, updating_func="linear")

# ----- Linear updates, ER network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Clusters\Clusters_ER_linear_N100_300timesteps_new"
# consp_frac = np.arange(0, 0.64, 0.04)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_clusters(N, M, "trust", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, params, seed, flex, k=k, folder_name=folder, updating_func="linear")
# run_clusters(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, [1.0, 0.5, 1.0], seed, flex, k=k, folder_name=folder, updating_func="linear")

# ----- Log updates, BA network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Clusters\Clusters_BA_log_N100_300timesteps_new"
# consp_frac = np.arange(0, 0.16, 0.01)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_clusters(N, M, "trust", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, params, seed, flex, m=m, folder_name=folder, updating_func="log")
#run_clusters(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, [1.0, 0.5, 1.0], seed, flex, m=m, folder_name=folder, updating_func="log")

# ----- Log updates, ER network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Clusters\Clusters_ER_log_N100_300timesteps_new"
# consp_frac = np.arange(0, 0.16, 0.01)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_clusters(N, M, "trust", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, params, seed, flex, k=k, folder_name=folder, updating_func="log")
# run_clusters(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, [1.0, 0.5, 1.0], seed, flex, k=k, folder_name=folder, updating_func="log")


# ================== Truthfulness of whole system as func. of conspirators ================
# num_simulations = 80
# num_iterations = 300

# ----- Linear updates, BA network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Consp_truth\BA_100_linear"
# consp_frac = np.arange(0, 0.64, 0.04)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_consp_sim(N, M, "trust", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, params, seed, flex, m=m, folder_name=folder, updating_func="linear")
# run_consp_sim(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, [1.0, 0.5, 1.0], seed, flex, m=m, folder_name=folder, updating_func="linear")

# ----- Log updates, BA network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Consp_truth\BA_100_log"
# consp_frac = np.arange(0, 0.16, 0.01)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_consp_sim(N, M, "trust", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, params, seed, flex, m=m, folder_name=folder, updating_func="log")
# run_consp_sim(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_barabasi_albert_network, [1.0, 0.5, 1.0], seed, flex, m=m, folder_name=folder, updating_func="log")

# ----- Linear updates, ER network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Consp_truth\ER_100_linear"
# consp_frac = np.arange(0, 0.64, 0.04)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_consp_sim(N, M, "trust", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, params, seed, flex, k=k, folder_name=folder, updating_func="linear")
# run_consp_sim(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, [1.0, 0.5, 1.0], seed, flex, k=k, folder_name=folder, updating_func="linear")

# ----- Log updates, ER network, N = 100
# folder = r"C:\Users\bibia\OneDrive\OneDrive - NTNU\5. klasse\Prosjektoppgaven\DHT-simulations-misinformation-propagation-main\Data files\Consp_truth\ER_100_log"
# consp_frac = np.arange(0, 0.16, 0.01)
# beta_range = [0.0, 0.5, 0.9]
# for beta in beta_range:
#      params = [1.0, beta, 1.0]
#      run_consp_sim(N, M, "trust", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, params, seed, flex, k=k, folder_name=folder, updating_func="log")
# run_consp_sim(N, M, "uniform", num_simulations, num_iterations, consp_frac, create_erdos_renyi_network, [1.0, 0.5, 1.0], seed, flex, k=k, folder_name=folder, updating_func="log")


