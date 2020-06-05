import random
import numpy as np
import heapq
import gurobipy as gp
from optimize_mov import mov_oracle
import sys, os
from utils import *
import time

def random_allocate(e, cand):
    remaining = cand.k
    cand.X = np.zeros(e.n)
    while remaining > 0:
        i = random.randint(0, e.n-1)
        if cand.p[i] > 0:
            new = min(remaining, random.uniform(0, 1/cand.p[i] - cand.X[i]))
            cand.X[i] += new
            remaining -= new

def iterated_best_response(e, epsilon, nguesses, max_iters):
    i = 0
    restarts = 0
    c_times, nc_times = [], []
    profiles_seen = set()
    e.A.pov_old = -float('inf')
    e.A.pov = -float('inf')
    e.B.pov_old = -float('inf')
    e.B.pov = -float('inf')
    while i < max_iters:
        # enablePrint()
        print("iteration", i)
        blockPrint()
        e.update_network()
        for cand in [e.A, e.B]:
            # print("optimizing for cand {}".format(cand.id))
            cand.X = np.zeros(e.n)
            # if cand.id == "A":
            #     min_mu = e.calculate_mean()
            #     # cand.X = mov_oracle(e, cand, cand.opp.X)
            #     # e.update_network()
            #     max_mu = min(e.n, min_mu + cand.k)
            # else:
            #     max_mu = e.calculate_mean()
            #     min_mu = max(0, max_mu - cand.k)
            # min_mu, max_mu = min(min_mu, max_mu), max(min_mu, max_mu) 
            cand.X, cand.pov, c_times, nc_times = pov_oracle(e, cand, 0, e.n, nguesses, c_times, nc_times)
            if abs(cand.pov - cand.pov_old) < epsilon:
                cand.opp.X = cand.opp.X_old
                cand.opp.pov = cand.opp.pov_old
                # print("final POV", cand.pov, cand.opp.pov)
                return i, restarts, c_times, nc_times
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
        e.A.pov=e.calculate_pov_exact()
        # print("mid POV", e.A.pov)
        i += 1

       # max_mean = min_mean + max(e.A.k, e.B.k) * max(max(e.A.p), max(e.B.p))

def pov_oracle(e, cand, min_mu, max_mu, nguesses, convex_times, nonconvex_times):
    # enablePrint()
    print("**********POV ORACLE OPTIMIZING FOR CAND {}************".format(cand.id))
    blockPrint()
    opp = cand.opp
    exact_mean = e.calculate_mean()
    mus = list(np.linspace(0, e.n, nguesses))
    mus.append(exact_mean)
    # enablePrint()
    print("EXACT MEAN: {}".format(exact_mean))
    blockPrint()
    stepsize = (max_mu - min_mu) / nguesses
    Xs = []
    povs_exact = []
    povs_approx = []
    results = []
    for mu in mus:
        print("mu", mu)
        result = {}
        try:
            X, convex, t = pov_oracle_iter(e, cand, mu, stepsize)
            cand.X = [X[i] if X[i] < 1./cand.p[i] else 1./cand.p[i] for i in range(e.n)]
            print(X)
            e.update_network()
            if convex:
                convex_times.append(t)
            else:
                nonconvex_times.append(t)
            result["POVa"] = round(e.calculate_pov_approx(), 3)
            result["POVe"] = round(e.calculate_pov_exact(), 3)
            result["theta"] = [round(i, 3) for i in e.theta_T]
            result["X"] = [round(i, 3) for i in X]
            povs_exact.append(e.calculate_pov_exact())
            povs_approx.append(e.calculate_pov_approx())
            Xs.append(X)
        except AttributeError:
            print("AttributeError", mu)
            pass
        results.append(result)
    # enablePrint()
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
    print("EXACT POVS: ", povs_exact)
    print("APPROX POVS: ", povs_approx)

    blockPrint()
    try:
        if cand.id == "A":
            x_a = np.argmax(povs_approx)
            x_e = np.argmax(povs_exact)
        else:
            x_a = np.argmin(povs_approx)
            x_e = np.argmin(povs_exact)
    except:
        raise Exception
    cand.X = Xs[x_e]
    # enablePrint()
    print("CAND {} FINAL ALLOCATION: X={}".format(cand.id,cand.X))
    blockPrint()
    return Xs[x_e], povs_exact[x_e], convex_times, nonconvex_times 

def pov_oracle_iter(e, cand, mu, stepsize):
    t0 = time.process_time()
    convex = False
    opp = cand.opp
    # if cand.id == "A":
    #     A, B = cand, cand.opp
    # else:
    #     B, A = cand, cand.opp
    m = gp.Model("POV")
    c = e.theta + (opp.goal - e.theta) * opp.p * opp.X
    u = cand.marginal_payoff(e, opp.X)

    # decision variables, bounded between 0 and 1
    X = m.addVars([i for i in range(e.n)], name=["X" + str(i) for i in range(e.n)], ub=[1./cand.p[i] for i in range(e.n)])

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
        convex = True
        m.setObjective(obj, gp.GRB.MINIMIZE)
    else:
        convex = False
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
    m.addConstrs((X[i] <= cand.max_expenditure(e, opp.X, i) for i in range(e.n)))

    m.setParam("OutputFlag", 0);
    m.setParam('TimeLimit', 4*60)

    m.optimize()

    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))
    # print('Obj: %g' % m.objVal)
    X_cand = []
    for i, var in enumerate(m.getVars()):
        if var.x <=1:
            X_cand.append(var.x)
        else:
            X_cand.append(1/cand.p[i])
    cand.X = X_cand
    print("Probability Xp", np.multiply(cand.X, cand.p))
    assert all([x <=1.001 for x in list(np.multiply(cand.X, cand.p))])
    assert all([x <=1 for x in list(np.multiply(cand.X, cand.p))])
    print("Probability Xp", np.multiply(cand.X, cand.p))
    t1 = time.process_time()
    
    return X_cand, convex, t1-t0


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



