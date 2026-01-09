
import numpy as np
import matplotlib.pyplot as plt

def BA_deg_dist(m, k):
    p_k = 2*m*(m-1)/(k*(k+1)*(k+2))
    return p_k

m = 25
N = 500
k_max = m*np.sqrt(N)

k_range = np.arange(m, k_max + 1)
p_k = BA_deg_dist(m, k_range)
BA_dist = N*p_k
# plt.plot(k_range,p_k)
# plt.xlim(m, k_max)
# # plt.yscale("log")
# # plt.xscale("log")
# plt.xticks(np.arange(m, k_max, 50))
# plt.show()

weights = np.array([0.2, 0.3, 0.7])

# Normaliser
w1 = weights/np.sum(weights)
w2 = weights/3
# print(w1)
# print(w2)

p1 = np.array([0.3, 0.2, 0.1, 0.4])
p2 = np.array([0.1, 0.5, 0.3, 0.1])
p3 = np.array([0.4, 0.4, 0.1, 0.1])

# Lineær oppdatering. Det har ingenting å si om vektene er normalisert så lenge man normaliserer q på slutten, samme resultat.
sum_v1 = w1[0]*p1 + w1[1]*p2 + w1[2]*p3
sum_v2 = w2[0]*p1 + w2[1]*p2 + w2[2]*p3
# print(sum_v1)
# print(sum_v2)

q_v1 = sum_v1/np.sum(sum_v1)
q_v2 = sum_v2/np.sum(sum_v2)
# print(q_v1)
# print(q_v2)

flex = 0.8
q_prev = np.array([0.1, 0.2, 0.3, 0.4])

# Dette er feil, får to ulike resultat selv om q_v1 og q_v2 er like. Må normalisere før flex.
q_v1_flex = (sum_v1*flex + (1-flex)*q_prev)/np.sum(sum_v1*flex + (1-flex)*q_prev)
q_v2_flex = (sum_v2*flex + (1-flex)*q_prev)/np.sum(sum_v2*flex + (1-flex)*q_prev)
# print(q_v1_flex)
# print(q_v2_flex)

# Eksponentiell oppdatering. Her har normaliseringen av vektene noe å si.
log_sum_v1 = w1[0]*np.log(p1) + w1[1]*np.log(p2) + w1[2]*np.log(p3) 
log_sum_v2 = w2[0]*np.log(p1) + w2[1]*np.log(p2) + w2[2]*np.log(p3)
exp_sum_v1 = np.exp(log_sum_v1)
exp_sum_v2 = np.exp(log_sum_v2)

q_exp_1 = exp_sum_v1/np.sum(exp_sum_v1)
q_exp_2 = exp_sum_v2/np.sum(exp_sum_v2)
# print(q_exp_1)
# print(q_exp_2)

##################################


