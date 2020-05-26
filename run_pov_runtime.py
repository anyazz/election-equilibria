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
    convex_means, nonconvex_means = [], []
    iters, restarts = [], []
    file_n = open("pov_data/{}_n.txt".format(network), "w")
    file_i = open("pov_data/{}_i.txt".format(network), "w")
    file_r = open("pov_data/{}_r.txt".format(network), "w")
    file_c = open("pov_data/{}_c.txt".format(network), "w")
    file_nc = open("pov_data/{}_nc.txt".format(network), "w")

    N = [round(n) for n in np.logspace(0, math.log(MAX_NODES, 2), num=30, base=2.0)]
    N = sorted(list(set(N)))
    t0 = time.process_time()
    for n in N[:18]:
        enablePrint()
        print("n = {}".format(int(n)))
        blockPrint()
        i, r, c, nc = run(int(n), network)
        iters.append(i)
        restarts.append(r)
        convex_means.append(c)
        nonconvex_means.append(nc)

        for file, item in zip([file_n, file_i, file_r, file_c, file_nc], [n, i, r, c, nc]):
            file.write(str(item) + ', ')
            file.flush()
    for file in [file_n, file_i, file_r, file_c, file_nc]:
        file.close()
    enablePrint()
    t1 = time.process_time()
    print("NETWORK {} RUNTIME: ".format(i), t1-t0)

datapts, i = 0, 1
networks = open("pov_data/networks.txt", 'w')
while datapts < 30:
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)
    l = len(data['trustMatrix'])
    if l > 500:
        print("Starting Network {}".format(i))
        networks.write(str(i) + ', ')
        networks.flush()
        main(i)
    i += 1

networks.close()

