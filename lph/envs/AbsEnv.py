from abc import abstractmethod
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np


class AbsEnv(gym.Env):
    def __init__(self, low_s, high_s, low_a, high_a, disc_s=False,
                 disc_a=False):
        self.low_s, self.high_s = low_s, high_s
        self.low_a, self.high_a = low_a, high_a
        self.d_s, self.d_a = len(low_s), len(low_a)
        self.disc_s, self.disc_a = disc_s, disc_a
        self.n_a = int(high_a[0]) if disc_a else -1
        self.n_s = int(high_s[0]) if disc_s else -1

        self.observation_space = MultiDiscrete(high_s)
        self.action_space = Discrete(self.n_a)

    @abstractmethod
    def step(self, a):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode='human', close=False):
        pass


class GraphEnv(AbsEnv):
    def __init__(self, conditions, curriculum, stochastic_reset=False,
                 noise_prob=0.0, goal=None):
        """
        Environment with graph transitions.
        :param conditions: A (list) of skill conditions, with one element per
        skill. Each element uses tuples to represent AND and lists to
        represent OR operators.
        Eg. [a,(b,c)] for "a or (b and c)"; (a,b) for "a and b"
        :param curriculum: A (list) of goals from which agents can learn to
        abstract skills
        :param stochastic_reset: (bool) whether to add randomness to starting
        :param noise_prob: transition noise probability
        state
        """
        self.conditions = conditions
        self.curriculum = curriculum
        self.stochastic_reset = stochastic_reset
        self.noise_prob = noise_prob
        disc_s = True
        disc_a = True
        n_s = n_a = len(conditions)
        low_s, high_s = [0] * n_s, [1] * n_s
        low_a, high_a = [0], [n_a]
        super().__init__(low_s, high_s, low_a, high_a, disc_s, disc_a)
        self.state = None
        self.goal = goal
        self.reset()

    def _check_cond(self, cond, s):
        if len(cond) == 0:
            return True
        is_and = isinstance(cond, tuple)  # Tuple means AND, list means OR
        # Decompose into OR and AND
        success_vec = [False] * len(cond)
        for i, c in enumerate(cond):
            if isinstance(c, (tuple, list)):
                success_vec[i] = self._check_cond(c, s)
            else:
                success_vec[i] = (s[c, :] == 1)
        if is_and:
            return np.all(success_vec)
        else:
            return np.any(success_vec)

    def step(self, a):
        required_cond = self.conditions[int(a)]
        self.state = self.state.copy()
        if self._check_cond(required_cond, self.state):
            self.state[a] = 1

        # Random perturbation
        if np.random.rand() < self.noise_prob:
            idx = a
            while idx == a:
                idx = np.random.randint(self.d_s)
            self.state[idx] = 1 - self.state[idx]

        done = False
        r = -1
        if self.goal is not None and self.goal.matches(self.state):
            done = True
            r = 0
        return self.state, r, done, {}

    def reset(self):
        self.state = np.zeros((len(self.high_s), 1))
        if self.stochastic_reset:
            n_rand_dim = np.random.poisson(2)
            idx = np.random.choice(len(self.high_s), n_rand_dim)
            self.state[idx, :] = 1
        return self.state

    def render(self, mode='human', close=False):
        pass
        # print("State: {}".format(self.state.T))


def test_check_cond():
    ge = GraphEnv([(0, 2)], [([0], [1])])

    # Testing single element
    assert not ge._check_cond([0], np.array([[0, 0, 0]]).T)
    assert not ge._check_cond([0], np.array([[0, 1, 0]]).T)
    assert not ge._check_cond([0], np.array([[0, 0, 1]]).T)
    assert not ge._check_cond([0], np.array([[0, 1, 1]]).T)
    assert ge._check_cond([0], np.array([[1, 0, 0]]).T)
    assert ge._check_cond([0], np.array([[1, 0, 1]]).T)
    assert ge._check_cond([0], np.array([[1, 1, 0]]).T)
    assert ge._check_cond([0], np.array([[1, 1, 1]]).T)

    # Testing AND
    assert not ge._check_cond((0, 2), np.array([[0, 0, 0]]).T)
    assert not ge._check_cond((0, 2), np.array([[0, 1, 0]]).T)
    assert not ge._check_cond((0, 2), np.array([[0, 0, 1]]).T)
    assert not ge._check_cond((0, 2), np.array([[0, 1, 1]]).T)
    assert not ge._check_cond((0, 2), np.array([[1, 0, 0]]).T)
    assert not ge._check_cond((0, 2), np.array([[1, 1, 0]]).T)
    assert ge._check_cond((0, 2), np.array([[1, 0, 1]]).T)
    assert ge._check_cond((0, 2), np.array([[1, 1, 1]]).T)

    # Testing OR
    assert not ge._check_cond([0, 1], np.array([[0, 0, 0]]).T)
    assert not ge._check_cond([0, 1], np.array([[0, 0, 1]]).T)
    assert ge._check_cond([0, 1], np.array([[1, 0, 1]]).T)
    assert ge._check_cond([0, 1], np.array([[1, 1, 1]]).T)
    assert ge._check_cond([0, 1], np.array([[0, 1, 0]]).T)
    assert ge._check_cond([0, 1], np.array([[0, 1, 1]]).T)
    assert ge._check_cond([0, 1], np.array([[1, 0, 0]]).T)
    assert ge._check_cond([0, 1], np.array([[1, 1, 0]]).T)

    # Testing compositions
    assert ge._check_cond([(0, 1), 2], np.array([[1, 1, 0]]).T)
    assert not ge._check_cond([(0, 1), 2], np.array([[1, 0, 0]]).T)
    assert not ge._check_cond([(0, 1), 2], np.array([[0, 1, 0]]).T)
    assert not ge._check_cond([(0, 1), 2], np.array([[0, 0, 0]]).T)
    assert ge._check_cond([(0, 1), 2], np.array([[1, 1, 1]]).T)
    assert ge._check_cond([(0, 1), 2], np.array([[1, 0, 1]]).T)
    assert ge._check_cond([(0, 1), 2], np.array([[0, 1, 1]]).T)
    assert ge._check_cond([(0, 1), 2], np.array([[0, 0, 1]]).T)

    # Testing compositions 2
    assert not ge._check_cond((0, [1, 2]), np.array([[0, 0, 0]]).T)
    assert not ge._check_cond((0, [1, 2]), np.array([[0, 1, 0]]).T)
    assert not ge._check_cond((0, [1, 2]), np.array([[0, 0, 1]]).T)
    assert not ge._check_cond((0, [1, 2]), np.array([[0, 1, 1]]).T)
    assert not ge._check_cond((0, [1, 2]), np.array([[1, 0, 0]]).T)
    assert ge._check_cond((0, [1, 2]), np.array([[1, 1, 0]]).T)
    assert ge._check_cond((0, [1, 2]), np.array([[1, 0, 1]]).T)
    assert ge._check_cond((0, [1, 2]), np.array([[1, 1, 1]]).T)
