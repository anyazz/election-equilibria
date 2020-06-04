import json
import math
import matplotlib.pyplot as plt 
import numpy as np

names = ["i", "r", "nc", "c", "abr", "sbr"]
results = {x: [] for x in names}
MAX_NODES = 500
networks = [1, 2]
X = np.linspace(0, 300, 16)
X = [int(x) for x in X]

for network in networks:
    network_res = {x: [] for x in names}
    file = open("data_utility/{}.txt".format(network), "r")
    data = file.readline()[:-2]
    data = '{' + data + '}'
    res = json.loads(data)
    print(res)
    for x in X:
        for name in names:
            network_res[name].append(res[str(x)][name]) 
    for name in names:
        results[name].append(network_res[name])
    file.close()
    
print(results)
iters, restarts, convex_means, nonconvex_means, abr, sbr = [np.mean(results[name], axis=0) for name in names]

print(abr)

fig, axes = plt.subplots(nrows=1, ncols=2)
# i = axes[0].bar(X, iters, label="Iterations")
# d = axes[0].bar(X, restarts, label="Detected Cycles")
# axes[0].set_title("Iterations to Convergence")
# axes[0].legend()
# axes[0].set_ylabel('Iteration number')
# axes[0].set_xlabel('Network size')

axes[1].plot(X, sbr, label="SBR Utility")
axes[1].plot(X, abr, label="ABR Utility")

axes[1].set_title("POV Utilities")
axes[1].legend()
axes[1].set_ylabel("POV")
axes[1].set_xlabel(r'$k_A$')

# plt.title("Performance of Alternating Best Response for Pure Equilibria in POV")
plt.xlabel('')
plt.show()
