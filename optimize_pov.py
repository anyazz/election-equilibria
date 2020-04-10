import random
import numpy as np
import heapq
import gurobipy as gp
from optimize_mov import mov_oracle
import sys, os
from utils import *

def random_allocate(e, cand):
    remaining = cand.k
    cand.X = np.zeros(e.n)
    while remaining > 0:
        i = random.randint(0, e.n-1)
        if cand.p[i] > 0:
            new = min(remaining, random.uniform(0, 1/cand.p[i] - cand.X[i]))
            cand.X[i] += new
            remaining -= new

def iterated_best_response(e, epsilon, max_iters):
    i = 0
    restarts = 0
    profiles_seen = set()
    e.A.pov_old = -float('inf')
    e.A.pov = -float('inf')
    e.B.pov_old = -float('inf')
    e.B.pov = -float('inf')
    while i < max_iters:
        print("iteration", i)
        for cand in [e.A, e.B]:
            print("optimizing for cand {}".format(cand.id))
            min_mu = sum(e.theta)
            cand.X = mov_oracle(e, cand, cand.opp.X)
            e.update_network()
            max_mu = e.calculate_mean()
            min_mu, max_mu = min(min_mu, max_mu), max(min_mu, max_mu) 
            cand.X, cand.pov = pov_oracle(e, cand, min_mu, max_mu, 30)
            if abs(cand.pov - cand.pov_old) < epsilon:
                cand.opp.X = cand.opp.X_old
                cand.opp.pov = cand.opp.pov_old
                print("POV", cand.pov, cand.opp.pov)
                return i, restarts
            cand.X_old = cand.X
            cand.pov_old = cand.pov
        tup = tuple(roundl(np.concatenate([e.A.X, e.B.X]), 3))
        if tup in profiles_seen:
            restarts += 1
            print("CYCLE --> RESTARTING")
            random_allocate(e, e.A)
            print("new XA", e.A.X)
            random_allocate(e, e.B)
            print("new XB", e.B.X)
        profiles_seen.add(tup)
        print("POV", e.A.pov, e.B.pov)
        i += 1

       # max_mean = min_mean + max(e.A.k, e.B.k) * max(max(e.A.p), max(e.B.p))

def pov_oracle(e, cand, min_mu, max_mu, nguesses):
    # blockPrint()
    opp = cand.opp
    mus = np.linspace(min_mu, max_mu, nguesses)
    print(mus)
    stepsize = (max_mu - min_mu) / nguesses
    Xs = []
    povs_exact = []
    povs_approx = []
    results = []
    for mu in mus:
        result = {}
        try:
            X = pov_oracle_iter(e, cand, mu, stepsize)
            result["POVa"] = round(e.calculate_pov_approx(), 3)
            result["POVe"] = round(e.calculate_pov_exact(), 3)
            result["theta"] = [round(i, 3) for i in e.theta_T]
            result["X"] = [round(i, 3) for i in X]
            povs_exact.append(e.calculate_pov_exact())
            povs_approx.append(e.calculate_pov_approx())
            Xs.append(X)
        except AttributeError:
            pass
        results.append(result)
    print(povs_approx)
    print(povs_exact)
    if cand.id == "A":
        x_a = np.argmax(povs_approx)
        x_e = np.argmax(povs_exact)
    else:
        x_a = np.argmin(povs_approx)
        x_e = np.argmin(povs_exact)

    print("i", '\t', "mu",'\t', "POVa", '\t', "POVe", '\t', "theta", '\t', "X")
    losing = True
    print('\n', 'MAXIMIZING VAR')
    for i in range(len(mus)):
        r = results[i]
        if losing and mus[i] > (e.n+1)/2:
            print('\n', 'MINIMIZING VAR')
            losing = False
        if r:
            print(i,'\t', round(mus[i], 3), '\t', r["POVa"],'\t', r["POVe"], '\t', r["theta"], '\t', r["X"])
        else:
            print (i, '\t', round(mus[i], 3), '\t', None)
    # enablePrint()
    cand.X = Xs[x_e]
    return Xs[x_e], povs_exact[x_e]     

def pov_oracle_iter(e, cand, mu, stepsize):
    opp = cand.opp
    # if cand.id == "A":
    #     A, B = cand, cand.opp
    # else:
    #     B, A = cand, cand.opp
    m = gp.Model("POV")
    c = e.theta + (opp.goal - e.theta) * opp.p * opp.X
    u = cand.marginal_payoff(e, opp.X)

    # decision variables, bounded between 0 and 1
    X = m.addVars([i for i in range(e.n)], name=["X" + str(i) for i in range(e.n)])

    # objective function
    obj = gp.QuadExpr()
    for i in range(e.n):
        y = 0
        for j in range(e.n):
            y += e.P_T.item(i, j) * (e.theta[j] + (opp.goal - e.theta[j]) * opp.p[j] * opp.X[j] + (cand.goal-e.theta[j]) * cand.p[j]*X[j] \
             + (2 * e.theta[j] - 1) * cand.p[j] * opp.p[j] * opp.X[j] * X[j]) 
        obj += y * y
    

    # check if either A losing and optimizing for A, or B losing and optimizing for B
    if (mu < (e.n+1)/2 and cand.id == "A") or (mu > (e.n+1)/2 and cand.id == "B"):
        m.setObjective(obj, gp.GRB.MINIMIZE)
    else:
        m.setObjective(obj, gp.GRB.MAXIMIZE)
        m.params.NonConvex = 2

    # budget constraint
    m.addConstr(X.sum() <= cand.k)

    # expected value constraint
    mean = gp.LinExpr()
    sign = 1 if cand.goal else -1
    for i in range(e.n):
        mean += sign * X[i] * u[i] + e.alpha[i] * c[i]
    m.addConstr(mean == mu)


    # min/max opinion constraint
    m.addConstrs((X[i] <= 1/cand.p[i] for i in range(e.n)))
    m.setParam("OutputFlag", 0);

    m.optimize()

    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))
    # print('Obj: %g' % m.objVal)

    X_cand = [i.x for i in m.getVars()]
    cand.X = X_cand
    
    return X_cand


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



