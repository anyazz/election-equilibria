import json
from classes import Election, Candidate
from optimize import mov_ftpl, pov_convex_opt, roundl
import random


def run(i):
	with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
	  network = json.load(fp)

	A = Candidate("A", 2.5, 1)
	B = Candidate("B", 2, 0)

	e = Election(network, [A, B], 20)
	print("N", e.n)
	print("theta", roundl(e.theta))
	mov_ftpl(e, 20, 10)
	print("THETA_T", roundl(e.theta_T))




run(3)
n = 3
network = {'relationsMatrix': np.identity(n)}
p_a = [0.3, 1, 0.6]
p_b = [1, 0.5, 0.8]
theta = [0.1, 0.9, 0.9]
A = Candidate("A", 1, 1, n, p_a)
B = Candidate("B", 2, 0, n, p_b)
# B.X = [1, 0, 0]
e3 = Election(network, [A, B], 20, theta)