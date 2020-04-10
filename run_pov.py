import json
from classes import Election, Candidate
import random
from visualize import *
import matplotlib.pyplot as plt
from optimize_pov import iterated_best_response

random.seed(410)
def run(i):
    with open ('data/json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    network = data['trustMatrix']
    opinion_attr = "sex"
    n = len(network)
    A = Candidate("A", 20, 1, n)
    B = Candidate("B", 30, 0, n)
    e = Election(data, [A, B], 10, opinion_attr, rand=False)

    i, r = iterated_best_response(e, 1e-4, 1e3)
    print(e.A.X)
    print(e.B.X)
    print('POV', e.calculate_pov_exact())
    print('iterations', i, r)

def main():
	run(3)

main()