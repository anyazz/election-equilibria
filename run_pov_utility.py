import json
from classes import Election, Candidate
from optimize_pov import *
import random
import os
import sys

random.seed(410)
def run(i):
    blockPrint()
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    opinion_attr = "sex"

    # X = [0, 5, 10]

    # X = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 115, 117, 118, 119, 119.5, 120]
    # X = [0, 3, 6, 9, 15, 20, 30, 45, 60, 80, 100, 120, 150, 200, 300]
    X = list(np.linspace(0, 300, 16))
    n = 32
    # file = open("data_utility/{}.txt".format(i), "w")

    A = Candidate("A", 0, 1, n)
    B = Candidate("B", 0, 0, n)
    e = Election(data, n, [A, B], 10, opinion_attr, rand=False)
    # if i == 1:
        # X = X[3:]
    # res = {}
    for x in X:
        budget_dict = {}
        enablePrint()
        # print("BUDGET ", x)
        blockPrint()
        A.k=x
        B.k = max(X)-x
        # print("BUDGETS 1", A.k, B.k)

        e.update_network()
        i, r, c, nc = iterated_best_response(e, 1e-2, 1e3)
        e.update_network()
        alt_pov = e.calculate_pov_exact()

        A.X, _, _, _ = pov_oracle(e, A, 0, e.n, 40, [], [])
        B.X, _, _, _ = pov_oracle(e, B, 0, e.n, 40, [], [])

        e.update_network()

        single_pov = e.calculate_pov_exact()
        budget_dict["i"] = i
        budget_dict["r"] = r
        budget_dict["c"] = np.mean(c)
        budget_dict["nc"] = np.mean(nc)
        budget_dict["abr"] = alt_pov
        budget_dict["sbr"] = single_pov

        print("{}: ".format(x) + json.dumps(budget_dict))
        # file.flush()
        # enablePrint()
        # print("Completed in {} iters with {} restarts".format(i, r))
        # print("ABR POV: {}, SBR POV: {}".format(alt_pov, single_pov))
        # blockPrint()
    # enablePrint()
    # file.close()

    return X

def main():
    for i in range(2, 11):
        print("NETWORK {}".format(i))
        run(i)
    # Xs, ABRs = [], []
    # X, ABR = run(3)
    # Xs.append(X)
    # ABRs.append(ABR)
    # X_mean = np.mean(Xs, axis=0)
    # ABR_mean = np.mean(ABRs, axis=0)
    # print(X_mean, ABR_mean)
    # plt.scatter(X, ABR)
    # plt.show()
    # plt.savefig('ftpl.png', dpi=300)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
main()

