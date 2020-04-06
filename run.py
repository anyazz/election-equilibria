import json
from classes import Election, Candidate
from optimize_mov import *
import random
from visualize import *

random.seed(250)
def run(i):
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    network = data['trustMatrix']
    opinion_attr = "sex"
    n = len(network)
    A = Candidate("A", 8, 1, n)
    B = Candidate("B", 12, 0, n)
    e = Election(data, [A, B], 20, opinion_attr)
    e.update_network()
    print("N", e.n)
    print("theta", roundl(e.theta, 3))

    # ftpl(e, 1)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    thetas = [e.theta, e.theta_0, e.theta_T]
    axes[-1, -1].axis('off')
    draw_networks(fig, axes, e.P, thetas)




run(1)
# n = 3
# network = {'relationsMatrix': np.identity(n)}
# p_a = [0.3, 1, 0.6]
# p_b = [1, 0.5, 0.8]
# theta = [0.1, 0.9, 0.9]
# A = Candidate("A", 1, 1, n, p_a)
# B = Candidate("B", 2, 0, n, p_b)
# # B.X = [1, 0, 0]
# e3 = Election(network, [A, B], 20, theta)