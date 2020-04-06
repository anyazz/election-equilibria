import random
import numpy as np
import heapq
import gurobipy as gp
from optimize_mov import mov_oracle

def double_oracle(e, epsilon):
    S_A = [np.zeros(e.n)]
    S_B = [np.zeros(e.n)]

    
    # max_mean = min_mean + max(e.A.k, e.B.k) * max(max(e.A.p), max(e.B.p))
   

def pov_oracle(e, cand, X_opp, nguesses):
    # A
    if cand_goal:
        min_mu = e.calculate_mean()
        max_mu = min_mu + cand.k
    else:
        max_mu = e.calculate_mean()
        min_mu = max_mu - cand.k
    mus = np.linspace(min_mu, max_mu, nguesses)
    Xs = []
    for mu in mus:
        try:
            X = pov_oracle_iter(e, cand, mu, X_opp)
            povs_exact.append(e.calculate_pov_exact())
            povs_approx.append(e.calculate_pov_approx())
            Xs.append(X)
        except AttributeError:
            pass
    x_a = np.argmax(povs_approx)
    x_e = np.argmax(povs_exact)
    return Xs[x_e], povs_exact[x_e]     

def pov_oracle_iter(e, cand, mu, X_opp):
    opp = cand.opp
    if cand.goal:
        A, B = cand, cand.opp
    else:
        B, A = cand, cand.opp
    m = gp.Model("POV")
    c = e.theta + (opp.goal - e.theta) * opp.p * X_opp
    u = cand.marginal_payoff(e, X_opp)

    # decision variables, bounded between 0 and 1
    X = m.addVars([i for i in range(e.n)], name=["X" + str(i) for i in range(e.n)])

    # objective function
    obj = gp.QuadExpr()
    for i in range(e.n):
        y = 0
        for j in range(e.n):
            y += e.P_T.item(i, j) * (e.theta[j] + (opp.goal - e.theta[j]) * opp.p[j] * X_opp[j] + (cand.goal-e.theta[j]) * cand.p[j]*X[j] \
             + (2 * e.theta[j] - 1) * cand.p[j] * opp.p[j] * X_opp[j] * X[j]) 
        obj += y * y
    

    # check if either A losing and optimizing for A, or B losing and optimizing for B
    if (mu < (e.n+1)/2 and cand.goal) or (mu > (e.n+1)/2 and not cand.goal):
        print("minimize")
        m.setObjective(obj, gp.GRB.MINIMIZE)
    else:
        print("maximize")
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
    m.setParam("LogFile", "gurobi.log");
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)

    cand.X = [i.x for i in m.getVars()]
    e.update_network()
    print('Theta 0', e.theta_0)
    print('Theta T', e.theta_T)
    return cand.X


def coreLP(e, S_a, S_b):
    a, u_a = coreLPA(e, S_a, S_b)
    X_a =  np.average(S_a, axis=0, weights=a)
    b, u_b = coreLPB(e, S_a, S_b)
    X_b =  np.average(S_b, axis=0, weights=a)
    return X_a, X_b, u_a, u_b


def coreLPA(e, S_a, S_b):
    m = gp.Model("CoreLP")
    max_u = m.addVar(name="u")
    J, K = len(S_a), len(S_b)
    a = m.addVars([i for i in range(J)], name=["a" + str(i) for i in range(e.n)], ub=1)
    
    m.setObjective(max_u, GRB.MAXIMIZE)

    # minimax constraint
    for k in range(K):
        u_k = gp.LinExpr()
        for j in range(J):
            u_k += e.calculate_pov_exact(S_a[j], S_b[k]) * a[j]
        m.addConstr(u_k >= max_u)
    
    m.optimize()
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)
    return a, m.objVal

def coreLPB(e, S_a, S_b):
    m = gp.Model("CoreLP")
    min_u = m.addVar(name="u")
    J, K = len(S_a), len(S_b)
    b = m.addVars([i for i in range(J)], name=["b" + str(i) for i in range(e.n)], ub=1)
    
    m.setObjective(min_u, GRB.MINIMIZE)

    # minimax constraint
    for k in range(K):
        u_k = gp.LinExpr()
        for j in range(J):
            u_k += 1-e.calculate_pov_exact(S_a[j], S_b[k]) * b[j]
        m.addConstr(u_k <= min_u)
    
    m.optimize()
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)
    return b, m.objVal



