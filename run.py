import json
from classes import Election, Candidate
from optimize_mov import *
import random
from visualize import *
from collections import Counter
# random.seed(250)

def run(i):
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    network = data['trustMatrix']
    opinion_attr = "sex"
    n = len(network)
    A = Candidate("A", 20, 1, n)
    B = Candidate("B", 30, 0, n)
    e = Election(data, [A, B], 30, opinion_attr, rand=False)
    e.update_network()
    h = np.array(e.calculate_homophily(e.theta))
    if h[0] < h[1]:
        mu = e.calculate_mean()/e.n
    else:
        mu = 1-e.calculate_mean()/e.n
    print(h, mu)
    return h, mu
    # ftpl(e, 5)
    # display(e, A, B)

def analyze(i, attr, count):
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)
    # print(data[attr])
    for i in range(len(data[attr])):
            a = data[attr][i][0]
            count[a] += 1
    return count


def display(e, A, B):
    print('i', '\t', 'p_A', '\t', 'p_B', '\t', 'alpha', '\t', 'X_A', '\t', 'X_B')
    for i in range(e.n):
        print(i, '\t', round(A.p[i], 2), '\t', round(B.p[i], 2), '\t', round(e.alpha[i], 2), '\t',\
         # round(A.u[i], 2), '\t', round(B.u[i], 2), '\t', \
         round(A.X[i], 2), '\t', round(B.X[i], 2))
    fig, axes = plt.subplots(nrows=3, ncols=1)
    for ax in axes.flat:
        ax.set(adjustable='box', aspect=0.5)
    fig.tight_layout(pad=0) 
    thetas = [e.theta, e.theta_0, e.theta_T]
    label_X = [False, True, False]
    # axes[-1, -1].axis('off')
    # draw_networks(fig, axes, e, thetas, label_X)

def main():
    minmax = np.zeros(2)
    homophily=np.zeros(2)
    total_mu = 0
    k=84
    for i in range(1, k+1):
        h, mu = run(i)
        homophily += h
        total_mu += mu
        minmax[0] += min(h)
        minmax[1] += max(h)
    print("AVERAGE", homophily/k, minmax/k, total_mu/k)

    # count = Counter()
    # k=84
    # attr = "race"
    # for i in range(1, k+1):
    #     print(i)
    #     count = analyze(i, attr, count)
    # print(count)
main()

strats = [[1,0,0]]
# , [0.33, 0, 0.67], #[0.3, 0, 0.7], [0.4, 0, 0.5]\
 # [0.5, 0, 0.5], [0.67, 0, 0.33], [0, 0, 1]]
n = len(strats)
game = [[None] * n for _ in range(n)]
thetas = [[None] * n for _ in range(n)]
for i, a in enumerate(strats):
    for j, b in enumerate(strats):
        A.X = a
        B.X = b
        e3.update_network()
        game[i][j] = calc_pov(e3)
        thetas[i][j] = e3.theta_0

for row in game:
    print(row)
for row in thetas:
    print(row)