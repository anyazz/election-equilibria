import random
import numpy as np
import heapq
import gurobipy as gp
    
def pov_convex_opt(e, cand1, cand2, mu):
    m = gp.Model("pov1")

    # decision variables, bounded between 0 and 1
    X = m.addVars([i for i in range(e.n)], name=["X" + str(i) for i in range(e.n)])

    # objective function
    obj = gp.QuadExpr()

    for i in range(e.n):
        y = 0
        for j in range(e.n):
            y += e.P_T.item(i, j) * (e.theta[j] + cand2.margin[j]*cand2.X[j] + cand1.margin[j]*X[j])
        obj += y * y
    
    # check if losing
    if mu < (e.n+1)/2:
        m.setObjective(obj, gp.GRB.MINIMIZE)
    else:
        m.setObjective(obj, gp.GRB.MAXIMIZE)
        m.params.NonConvex = 2

    # budget constraint
    m.addConstr(X.sum() <= cand1.budget)
    # expected value constraint
    mean = gp.LinExpr()
    for i in range(e.n):
        mean += e.alpha[i] * (e.theta[i] + cand2.margin[j]*cand2.X[j] + cand1.margin[i] * X[i])
    m.addConstr(mean == mu)
    # min/max opinion constraint
    m.addConstrs((cand1.margin[i]*X[i] + cand2.margin[i]*cand2.X[i] + e.theta[i] <= 1 for i in range(e.n)))
    m.addConstrs((cand1.margin[i]*X[i] + cand2.margin[i]*cand2.X[i] + e.theta[i] >= 0 for i in range(e.n)))

    m.optimize()
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)

    cand1.X = [i.x for i in m.getVars()]
    e.update_network()
    print('Theta 0', e.theta_0)
    print('Theta T', e.theta_T)


def mov_ftpl(e, iters):
    for i in range(iters):
            print("***ITERATION*** ", i)
            ftpl_iter(e, e.A, e.B, i)
            ftpl_iter(e, e.B, e.A, i)
    for x in [e.A, e.B]:
        print(x.id, "FINAL HIST ", x.ftpl_history)
    e.update_network()

def ftpl_iter(e, cand1, cand2, i):
    # perturb = [random.uniform(0, 1/epsilon) for _ in range(n)]
    theta_2 = e.theta
    if i:
        X_2 = np.mean(cand2.ftpl_history[:i], axis=0) #+ perturb
        print("mean opponent action", X_2)
        theta_2 = e.advertise(X_2, cand2.margin, np.zeros(e.n), np.zeros(e.n))
    print("theta after opponent advertisement", theta_2)
    X_1 = ftpl_best_response(e, cand1, theta_2)
    cand1.ftpl_history.append(X_1)
    cand1.X = X_1
    print(cand1.id, "X", cand1.X)
    e.calculate_mean()
    print("")



def ftpl_best_response(e, cand, theta):
    X = np.zeros(e.n)
    remaining = cand.budget
    score = np.multiply(e.alpha, cand.margin)
    print(cand.id, "score", score)
    heap = [(-abs(score[i]), i) for i in range(len(score))]
    heapq.heapify(heap)
    while remaining:
        if len(heap):
            x_score, x = heapq.heappop(heap)
            assert (cand.goal - e.theta[x])/cand.margin[x] > 0
            X[x] = min(((cand.goal - e.theta[x])/cand.margin[x]), remaining)
        remaining -= X[x]
    return X