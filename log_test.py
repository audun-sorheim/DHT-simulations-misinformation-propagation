import numpy as np


def update_private(weights, neighbour_p):
    """Update rule for private beliefs from Riazi's thesis"""
    
    log_sum = np.dot(weights, np.log(neighbour_p))
    # print(f"log sum: {log_sum}")

    exp_log_sum = np.exp(log_sum)
    # print(f"exp log sum: {exp_log_sum}")
    # print(f"sum(exp_log_sum): {np.sum(exp_log_sum, axis=0)}")

    return (exp_log_sum/np.sum(exp_log_sum, axis=0)) # private belief vector


np.random.seed(27) # For comparisons across different epsilon
epsilon_vals = [1e-100, 1e-20, 1e-12, 1e-6, 1e-3]

# Neighbour public beliefs
p1 = np.random.random(size=4)
p2 = np.random.random(size=4)
p3 = np.random.random(size=4)
p4 = np.random.random(size=4)
p1 = p1/np.sum(p1)
p2 = p2/np.sum(p2)
p3 = p3/np.sum(p3)
p4 = p4/np.sum(p4)

weights = np.array([1, 1, 1, 1, 1])/5 # Equal weights
weights_skewed = np.array([0.24, 0.24, 0.24, 0.24, 0.04]) # Lower weight for conspirator

for epsilon in epsilon_vals:
    # Create conspirator
    conspirator = np.array([1-3*epsilon, epsilon, epsilon, epsilon])

    neighbour_beliefs = np.array([p1, p2, p3, p4, conspirator])
    print("=================")
    print(f"epsilon = {epsilon}")
    print("=================")

    print("Neighbour beliefs:")
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    print(conspirator)
    print("-------------------")

    print(f"Updated private belief, equal weights:")
    print(update_private(weights, neighbour_beliefs))
    print(f"Updated private belief, low conspirator weight:")
    print(update_private(weights_skewed, neighbour_beliefs))
    print("\n")
   
