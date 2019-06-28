import numpy as np
import os
from lph.envs.AbsEnv import GraphEnv


def _build_curriculum(conditions):
    nodes_n_parents = [0] * len(conditions)  # Number of parents for each node
    nodes_parents = [list(c) for c in conditions]
    # Count number of parents for each node
    for i in range(len(nodes_parents)):
        for parent in nodes_parents[i]:
            nodes_n_parents[i] += 1 + nodes_n_parents[parent]

    curriculum = [([idx], [1]) for idx in np.argsort(nodes_n_parents)
                  if nodes_n_parents[idx] > 0]
    return curriculum[:-1:2] + [curriculum[-1]]


class RandomGraphEnv(GraphEnv):
    """
    A randomly generated environment:
    >>> conditions = [[]]+[list(np.random.choice(np.arange(i),
    >>>    min(i,np.random.poisson(2)), replace=False)) for i in range(1,100)]

    """
    # Load conditions
    condition_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "random_graph_conditions.npz")
    conditions = list(np.load(condition_file)['arr_0'])

    # Suggested curriculum
    curriculum = _build_curriculum(conditions)

    def __init__(self, stochastic_reset=False, noise_prob=0.2, goal=None):
        super().__init__(RandomGraphEnv.conditions,
                         RandomGraphEnv.curriculum,
                         stochastic_reset, noise_prob, goal=goal)
