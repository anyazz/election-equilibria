from classes import Election, Candidate
from optimize_pov import *
import random
import json
import numpy as np
import matplotlib.pyplot as plt


# n = 3
# network = {'relationsMatrix': np.identity(n)}
# p_a = [0.57, 1, 0.44]
# p_b = np.multiply(-1, [0, 0, 0])
# theta = [0.3, 0.2, 0.1]
# A = Candidate("A", .5, 1, n, p_a)
# B = Candidate("B", 0, 0, n, p_b)
# e2 = Election(network, [A, B], 20, theta)
# e2.update_network()
def calc_pov(e):
    return (round(e3.calculate_mean(), 3))
    # return (round(e3.calculate_pov_exact(), 3))

n = 3
network = {'trustMatrix': np.identity(n)}
p_a = [0.5, 0.1, 1]
p_b = [0.7, 0.4, 1]
theta = [0.3, .2, 1]
A = Candidate("A", 3, 1, n, p_a)
B = Candidate("B", 3, 0, n, p_b)
e3 = Election(network, [A, B], 0, "sex", theta=theta)
# e3.A.X = [1,0,0]
# e3.B.X = [0,0,1]
# e3.update_network()
# print(e3.theta_0)
# print(e3.calculate_mean())

def test2(e):
    e.A.X = [2.0, 0.0, 1.0]
    e.B.X=[1.4285714285714286, 1.5714285714285714, 0.0]
    e.update_network()
    # print(e.theta_T)
    # print(e.calculate_pov_exact())
    # random_allocate(e, e.A)
    iterated_best_response(e, 1e-4)
    print(e.A.X)
    print(e.B.X)
    print(e.calculate_pov_exact())


def test(e):
    print_results(e)
    min_mean = e.calculate_mean()
    # max_mean = min_mean + max(e.A.k, e.B.k) * max(max(e.A.p), max(e.B.p))
   
    print("THETA_T", e.theta_T)
    mus = np.linspace(1, 4, 51)
    results = []
    povs_approx, povs_exact, x = [], [], []
    for mu in mus:
        result = {}
        try:
            pov_oracle_iter(e, B, mu, A.X)
            result["POVa"] = round(e.calculate_pov_approx(), 3)
            result["POVe"] = round(e.calculate_pov_exact(), 3)
            result["theta"] = [round(i, 3) for i in e.theta_T]
            result["X"] = [round(i, 3) for i in B.X]
            povs_exact.append(result["POVe"])
            povs_approx.append(result["POVa"])
            x.append(mu)
        except AttributeError:
            pass
        results.append(result)
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
    i_a = np.argmax(povs_approx)
    i_e = np.argmax(povs_exact)
    print("BEST APPROX:", x[i_a])
    print("BEST EXACT:", x[i_e])
    print((e.n+1)/2)
    # plt.plot(x, povs_exact)
    # plt.plot(x, povs_approx)
    # plt.show()

def print_results(e):
    print('MEAN: ', e.calculate_mean())
    print('POV APPROX: ', e.calculate_pov_approx())
    print('POV EXACT: ', e.calculate_pov_exact())

test2(e3)
