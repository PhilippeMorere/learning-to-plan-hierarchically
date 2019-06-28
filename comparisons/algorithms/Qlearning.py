import numpy as np


class QLearning:
    def __init__(self, all_a, epsilon=0.2, learning_rate=0.1, gamma=0.99):
        """
        Parameters.
        :param all_a: list of all actions
        :param learning_rate: stochasticOptimisers object
        :param epsilon: parameter for epsilon greedy policy
        :param learning_rate: Q-learning learning rate
        :param gamma: discount factor
        """
        self.all_a = all_a
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_values = {}

    def q(self, sa):
        if tuple(sa) in self.q_values:
            return self.q_values[tuple(sa)]
        else:
            return 0.0

    def policy(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.all_a)
        max_q = None
        max_a = None
        for a in self.all_a:
            q = self.q(np.hstack([s, [a]]))
            if max_q is None or q > max_q:
                max_q, max_a = q, a
        return max_a

    def update_q(self, sa, q):
        self.q_values[tuple(sa)] = (1 - self.learning_rate) * self.q(sa) + \
                                   self.learning_rate * q

    def update(self, s, a, r, sp):
        """
        Adds one data point at a time.
        """
        sa = np.hstack([s, [a]])
        ap = self.policy(sp)
        sap = np.hstack([sp, [ap]])
        self.update_q(sa, r + self.gamma * self.q(sap))

    def end_of_ep_update(self):
        pass
