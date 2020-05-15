import math
import matplotlib.pyplot as plt 
import numpy as np

names = ["i", "r", "nc", "c"]
results = {x: [] for x in names}
MAX_NODES = 500
networks = open("pov_data/networks.txt", 'r')
networks_lst = networks.readline().split(", ")[:-1]

for network in networks_lst:
    for name in names:
        file = open("pov_data/{}_{}.txt".format(network, name), "r")
        lst = file.readline().split(", ")[:-1]
        float_lst = [float(e) for e in lst]
        results[name].append(float_lst)
        file.close()
    
print(results)
iters, restarts, convex_means, nonconvex_means = [np.mean(results[name], axis=0) for name in names]

print(convex_means)
X = [round(n) for n in np.logspace(0, math.log(MAX_NODES, 2), num=80, base=2.0)]
X = sorted(list(set(X)))
fig, axes = plt.subplots(nrows=1, ncols=2)
i = axes[0].bar(X, iters, label="Iterations")
d = axes[0].bar(X, restarts, label="Detected Cycles")
axes[0].set_title("Iterations to Convergence")
axes[0].legend()
axes[0].set_ylabel('Iteration number')
axes[0].set_xlabel('Network size')

axes[1].plot(X, convex_means, label="Convex Optimizations")
axes[1].plot(X, nonconvex_means, label="Nonconvex Optimizations")

axes[1].set_title("Average Runtime")
axes[1].legend()
axes[1].set_ylabel('Time (s)')
axes[1].set_xlabel('Network size')

# plt.title("Performance of Alternating Best Response for Pure Equilibria in POV")
plt.xlabel('')
plt.show()
