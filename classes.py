import random
import numpy as np
from scipy.stats import norm
import math
from poibin import PoiBin

delta = 0.0001

class Election:
    def __init__(self, network, candidates, T, theta=[], random=False, model_type='linear'):
        self.network = network
        self.P = np.array(network["relationsMatrix"])
        self.n = len(self.P)
        self.model_type = model_type
        if len(theta):
            self.theta = np.array(theta)
        else: 
            [0.5] * self.n
            if model_type == 'linear':
                if random:
                    self.theta = [random.random() for _ in range(self.n)]
                else:
                    self.assign_opinions()
        self.A, self.B = candidates
        self.A.opp, self.B.opp = self.B, self.A
        # self.A.p = np.ones(self.n)
        # self.B.p = -1 * np.ones(self.n)
        self.P_T = np.linalg.matrix_power(self.P, T)
        self.alpha = np.dot(self.P_T, np.ones(self.n))

    def assign_opinions(self):
        for i in range(self.n):
            race = self.network["race"][i][0]
            if race == 0:
                self.theta[i] = random.random()
            elif race in [1, 3, 4, 5]:
                self.theta[i] = random.uniform(0, 0.8)
            elif race == 2:
                self.theta[i] = random.uniform(0.5, 1)

    def advertise(self):
        theta_0 = self.theta + (np.ones(self.n) - self.theta) * self.A.p * self.A.X - self.theta * self.B.p * self.B.X \
        + (2 * self.theta - np.ones(self.n)) * self.A.p * self.B.p * self.A.X * self.B.X
        return theta_0

    def update_network(self):
        self.theta_0 = self.advertise()
        self.theta_T = np.matmul(self.P_T, self.theta_0)

    def calculate_mean(self):
        self.update_network()
        mean = np.sum(self.theta_T)
        return mean

    def calculate_pov_approx(self):
        mu = self.calculate_mean()
        print(mu)
        print(self.theta_T)
        square_sum = 0
        for i in range(self.n):
            square_sum += self.theta_T[i] ** 2
        print(square_sum)
        print(((self.n + 1)/2 - mu)/(math.sqrt(mu-square_sum)))
        pov = 1 - norm.cdf(((self.n + 1)/2 - mu)/(math.sqrt(mu-square_sum)))
        return pov

    def calculate_pov_exact(self):
        self.theta_T = self.round_probabilities(self.theta_T)
        pb = PoiBin(self.theta_T)
        return 1 - pb.cdf(math.floor(self.n/2))

    def round_probabilities(self, lst):
        # round items in list if just above 1 or just below 0 due to float issues
        for i in range(len(lst)):
            if lst[i] > 1:
                if lst[i]-1 < delta:
                    lst[i] = 1
                else:
                    print(i, lst[i])
                    raise Exception("probability > 1 for item {}: {}".format(i, lst[i]))
            if lst[i] < 0:
                if -lst[i] < delta:
                    lst[i] = 0
                else:
                    raise Exception("probability < 0 for item {}: {}".format(i, lst[i]))
        return lst
class Candidate:
    def __init__(self, id_, k, goal, n, p=[]):
        self.id = id_
        self.goal = goal
        self.k = k
        if len(p):
            self.p = np.array(p) 
        else:
            numpy.multiply(1 if goal else -1, [random.random() for _ in range(n)])
        self.X = np.zeros(n)
        self.ftpl_history = []
        self.opp = None

    # u
    def marginal_payoff(self, e, X_opp):
        # set sign to negative for B
        sign = 1 if self.goal else -1
        return sign * e.alpha * ((np.array([self.goal] * e.n) - e.theta) * self.p + (2 * e.theta - np.ones(e.n)) * self.p * self.opp.p * X_opp)
    
    # eq. 3.11: expenditure required to convert node to 1
    def max_expenditure(self, e, X_opp, i):
        num = self.goal - e.theta[i] + (e.theta[i] - (self.opp.goal)) * self.opp.p[i] * X_opp[i]
        denom = (self.goal-e.theta[i]) * self.p[i] + (2 * e.theta[i] - 1) * self.p[i] * self.opp.p[i] * X_opp[i]
        return min(num/denom, 1/self.p[i])




