import json
from classes import Election, Candidate
import random
from visualize import *
import matplotlib.pyplot as plt
from optimize_pov import iterated_best_response, blockPrint, enablePrint
import time
import math

random.seed(410)
MAX_NODES = 500
def run(n, i):
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)
    N = len(data['trustMatrix'])
    opinion_attr = "sex"
    A = Candidate("A", 20, 1, n)
    B = Candidate("B", 30, 0, n)
    e = Election(data, n, [A, B], 5, opinion_attr, rand=False)
    i, r, c_times, nc_times = iterated_best_response(e, 1e-2, 1e3)
    c, nc = 0, 0
    if len(c_times):
        c = np.mean(c_times)
    if len(nc_times):
        nc = np.mean(nc_times)

        # print(e.A.X)
        # print(e.B.X)
        # print('A POV', e.calculate_pov_exact())
        # print('iterations:{}, restarts:{}'.format( i, r))
        # print('avg convex time:{}, avg nonconvex time{}'.format(np.mean(c_times), np.mean(nc_times)))
    return [i, r, c, nc]
def main(network):
    blockPrint()
    convex_means, nonconvex_means = [], []
    iters, restarts = [], []

    N = [round(n) for n in np.logspace(0, math.log(MAX_NODES, 2), num=30, base=2.0)]
    N = sorted(list(set(N)))
    t0 = time.process_time()
    for n in N[:18]:
        i, r, c, nc = run(int(n), network)
        iters.append(i)
        restarts.append(r)
        convex_means.append(c)
        nonconvex_means.append(nc)
    enablePrint()
    print({"i": iters, "r": restarts, "c": convex_means, "nc": nonconvex_means})
    t1 = time.process_time()
    # print("NETWORK {} RUNTIME: ".format(i), t1-t0)

networks = list(range(10, 44))
for x in [18, 19, 21, 37]: 
    networks.remove(x)
print(len(networks))

for i in networks:
    print("Network {}".format(i))
    main(i)
    i += 1

# networks.close()

