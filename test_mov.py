from classes import Election, Candidate
from optimize_mov import *
import random
import json
import numpy as np
from visualize import *
# n = 5
# network = {'relationsMatrix': np.identity(n)}
# alpha = [0, 1, .5, .75, 1]
# m_a = [0.2, 0, 0.5, 1, .5]
# m_b = np.multiply(-1, [.5, 1, 0.2, .5, .8])
# theta = [0, 0.4, 0.6, 0.5, 0.8]
# A = Candidate("A", 2.5, 1, n, m_a)
# B = Candidate("B", 2, 0, n, m_b)
# e1 = Election(network, [A, B], 20, theta)

# n = 2
# network = {'trustMatrix': np.identity(n)}
# p_a = [0.4, .5]
# p_b = np.ones(n)
# theta = [0.2, 0.5]
# A = Candidate("A", 1, 1, n, p_a)
# B = Candidate("B", 2, 0, n, p_b)
# e2 = Election(network, [A, B], 20, theta)
# e2.update_network()

# print("max spend 1", ftpl_max_spend(e2, A, B, 1))


# print("ad", e2.advertise([.5, 2], p_a, np.zeros(2), p_b))
# print("ad", e2.advertise([2.5, 0], p_a, np.zeros(2), p_b))

n = 3
P = [[0, 0.2, 0.8], [0.5, 0.5, 0], [0, 0, 1]] #np.identity(n)
network = {'trustMatrix': P}
p_a = [0.3, 1, 0.6]
p_b = [1, 0.5, 0.8]
theta = [0.1, 0.9, 0.9]
A = Candidate("A", 1, 1, n, p_a)
B = Candidate("B", 2, 0, n, p_b)
# B.X = [1, 0, 0]
e3 = Election(network, [A, B], 20, rand=True)
e3.update_network()
fig, axes = plt.subplots(nrows=2, ncols=2)
thetas = [[0,.1,0.2],[.3,.4,.5],[.9,.8,.7]]
print(thetas)
print(thetas)
axes[-1, -1].axis('off')
draw_networks(fig, axes, e3.P, thetas)
# print("max spend 0", ftpl_max_spend(e2, A, B, 0))
# ftpl(e3, 1)




