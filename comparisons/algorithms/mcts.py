import time
import math
import random


def random_policy(state):
    r = 0
    while not state.is_terminal():
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception(
                "Non-terminal state has no possible actions: " + str(state))
        state = state.take_action(action)
        r += state.get_reward()
    return r


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}


class Mcts:
    def __init__(self, time_limit=None, iteration_limit=None,
                 exploration_constant=1 / math.sqrt(2),
                 rollout_policy=random_policy):
        self.root = None
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError(
                    "Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = 'time'
        else:
            if iteration_limit is None:
                raise ValueError(
                    "Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limit
            self.limit_type = 'iterations'
        self.exploration_constant = exploration_constant
        self.rollout = rollout_policy

    def search(self, initial_state):
        self.root = TreeNode(initial_state, None)

        if self.limit_type == 'time':
            time_limit = time.time() + self.time_limit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        return self.get_action(self.root, best_child)

    def execute_round(self):
        node = self.select_node(self.root)
        reward = self.rollout(node.state)
        self.back_propagate(node, reward)

    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node

    @staticmethod
    def expand(node):
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children.keys():
                new_node = TreeNode(node.state.take_action(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node

        raise Exception("Should never reach here")

    @staticmethod
    def back_propagate(node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    @staticmethod
    def get_best_child(node, exploration_value):
        best_value = float("-inf")
        best_node = None
        for child in node.children.values():
            e = exploration_value * math.sqrt(2 * math.log(node.num_visits) /
                                              child.num_visits)
            node_value = child.total_reward / child.num_visits + e
            if node_value >= best_value:
                best_value = node_value
                best_node = child
        return best_node

    @staticmethod
    def get_action(root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action
