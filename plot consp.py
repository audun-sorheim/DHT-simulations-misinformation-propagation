import numpy as np
import matplotlib.pyplot as plt

def f(a, x):
    return 10**(-a*x)

weights = np.linspace(0, 1, 100)
plt.figure(figsize=(15, 10))
plt.plot(weights, f(300, weights), label="epsilon=1e-300")
plt.plot(weights, f(12, weights), label="epsilon=1e-12")
plt.plot(weights, f(6, weights), label="epsilon=1e-6")
plt.plot(weights, f(3, weights), label="epsilon=1e-3")
plt.plot(weights, f(1, weights), label="epsilon=1e-1")
plt.title("Conspirator 'strength' as a function of weight value, p^W = epsilon^W")
plt.xlabel("Weight W")
plt.ylabel("Weighted belief value p^W")
plt.legend()
#plt.yscale("log")
plt.savefig("epsilon.png")
plt.show()


