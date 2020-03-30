from classes import Election, Candidate
from optimize import mov_ftpl, pov_convex_opt
import random
import json
import numpy as np

# with open ('ADD_data.json', 'r') as fp:
#   itemlist = json.load(fp)
# print(itemlist[0])
n = 5
alpha = [0, 1, .5, .75, 1]
m_a = [0.2, 0, 0.5, 1, .5]
m_b = np.multiply(-1, [.5, 1, 0.2, .5, .8])
theta = [0, 0.4, 0.6, 0.5, 0.8]
A = Candidate("A", 2.5, 1, n, m_a)
B = Candidate("B", 2, 0, n, m_b)
test1 = Election(n, [A, B], alpha, theta)

n = 3
network = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
m_a = [0.5, 1, 0.5]
m_b = np.multiply(-1, [0, 0, 0])
theta = [0.3, 0.2, 0.1]
A = Candidate("A", .5, 1, n, m_a)
B = Candidate("B", 0, 0, n, m_b)
test2 = Election(n, [A, B], theta)
test2.assign_network(network, 1)
test2.update_network()

def test_ftpl(test):
    print(test.theta)
    mov_ftpl(test, 20)
    print("THETA_T", [round(i, 3) for i in test.theta_T])

def test_pov(test):
    print_results(test)
    min_mean = test.calculate_mean()
    max_mean = n
   
    print("THETA_T", test.theta_T)
    mus = np.linspace(.5, 4, 61)
    results = []
    povs_approx, povs_exact = [], []
    for mu in mus:
        result = {}
        try:
            pov_convex_opt(test, A, B, mu)
            result["POVa"] = round(test.calculate_pov_approx(), 3)
            result["POVe"] = round(test.calculate_pov_exact(), 3)
            result["theta"] = [round(i, 3) for i in test.theta_T]
            result["X"] = [round(i, 3) for i in A.X]
            povs_exact.append(result["POVe"])
            povs_approx.append(result["POVa"])
        except:
            pass
        results.append(result)
    print("i", '\t', "mu",'\t', "POVa", '\t', "POVe", '\t', "theta", '\t', "X")
    for i in range(len(mus)):
        r = results[i]
        if r:
            print(i,'\t', round(mus[i], 3), '\t', r["POVa"],'\t', r["POVe"], '\t', r["theta"], '\t', r["X"])
        else:
            print (i, '\t', round(mus[i], 3), '\t', None)
    i_a = np.argmax(povs_approx)
    i_e = np.argmax(povs_exact)
    print("BEST APPROX:", i_a)
    print("BEST EXACT:", i_e)


def print_results(test):
    print('MEAN: ', test.calculate_mean())
    print('POV APPROX: ', test.calculate_pov_approx())
    print('POV EXACT: ', test.calculate_pov_exact())

# test_ftpl(test2)
test_pov(test2)

