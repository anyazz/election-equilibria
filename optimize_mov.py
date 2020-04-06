import random
import numpy as np
import heapq
import gurobipy as gp
import math
    
def ftpl(e, epsilon):
    iters = 4 * e.n**2 * max(e.A.k, e.B.k)/(epsilon**2)
    for r in range(math.ceil(iters)):
        print("\nITERATION", r)
        ftpl_iter(e, e.A, r, epsilon)
        print("")
        ftpl_iter(e, e.B, r, epsilon)
        e.update_network()
        print('RESULT', e.calculate_mean(), e.theta_0)
    print("\nEquilibrium after {} iters: ".format(iters))

    print("Original theta: {}".format(e.theta))
    print("Final theta: {}".format(e.theta_0))
    print("Original Mean: {}".format(sum(e.theta)))
    print("Final Mean: {}".format(sum(e.theta_0)))
    for cand in [e.A, e.B]:
        cand.X = np.mean(cand.ftpl_history, axis=0)
        print("Final X{}: \t{}".format(cand.id, cand.X))

def ftpl_iter(e, cand, r, epsilon):
    opp = cand.opp
    perturb = [random.uniform(0, 1/epsilon) for _ in range(e.n)]
    if r:
        X_opp = np.mean(opp.ftpl_history[:r], axis=0) + np.multiply(1/r, perturb)
    else:
        X_opp = np.zeros(e.n)
    print("X_opp", X_opp)
    X = mov_oracle(e, cand, X_opp)
    cand.ftpl_history.append(X)
    cand.X = X
    print("best X_" + cand.id + ":", roundl(X, 2))

def mov_oracle(e, cand, X_opp):
    X = np.zeros(e.n)
    remaining = cand.k
    score = cand.marginal_payoff(e, X_opp)
    print("scores", score)
    heap = [(-(score[i]), i) for i in range(len(score))]
    heapq.heapify(heap)
    while remaining > 0:
        if len(heap):
            x_score, x = heapq.heappop(heap)
            max_X = cand.max_expenditure(e, X_opp, x)
            X[x] = min(max_X, remaining)
            assert X[x] > 0
        remaining -= X[x]
    return X

def roundl(lst, precision):
    return [round(x, precision) for x in lst]