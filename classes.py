import random
import numpy as np
from scipy.stats import norm
import math
from poibin import PoiBin


class Election:
    def __init__(self, n, candidates, theta=None, model_type='linear'):
        self.n = n
        # self.alpha  = alpha
        self.theta = theta or [0.5] * self.n
        self.model_type = "linear"
        # if self.model_type == "linear":
        #   self.assign_opinions()
        self.A, self.B = candidates
        self.epsilon = 5

    def assign_network(self, network, T):
        self.P = np.array(network)
        self.P_T = np.linalg.matrix_power(self.P, T)
        self.alpha = np.dot(self.P_T, np.ones(self.n))

    def assign_opinions(self):
        self.theta = [random.random() for _ in range(self.n)]

    def advertise(self, X_a, m_a, X_b, m_b):
        theta_0 = self.theta + np.multiply(X_a, m_a) + np.multiply(X_b, m_b)
        return [self.restrict_range(theta_0[i]) for i in range(self.n)]

    def update_network(self):
        self.theta_0 = self.advertise(self.A.X, self.A.margin, self.B.X, self.B.margin)
        self.theta_T = np.matmul(self.P_T, self.theta_0)

    def restrict_range(self, x):
        if 0 <= x <= 1: return x
        if x < 0: return 0
        if x > 1: return 1

    def calculate_mean(self):
        mean = np.dot(self.alpha, self.theta_0)
        return mean

    def calculate_pov_approx(self):
        mu = self.calculate_mean()
        square_sum = 0
        for i in range(self.n):
            square_sum += self.theta_T[i] ** 2
        pov = 1 - norm.cdf(((self.n + 1)/2 - mu)/(math.sqrt(mu-square_sum)))
        return pov

    def calculate_pov_exact(self):
        pb = PoiBin(self.theta_T)
        return 1 - pb.cdf(math.floor(self.n/2))


class Candidate:
    def __init__(self, id_, budget, goal, n, margin):
        self.id = id_
        self.goal = goal
        self.budget = budget
        self.margin = margin 
        # numpy.multiply(1 if goal else -1, [random.random() for _ in range(n)])
        self.X = np.zeros(n)
        self.ftpl_history = []




