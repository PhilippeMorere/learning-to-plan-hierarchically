import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from .utils import DynamicArray


class RRT:
    def __init__(self, sample_fn, collision_fn, goal_check_fn=None,
                 action_list=None, expand_dis=1.0, goal_sample_rate=.1):
        """
        Rapidly-explorting Random Trees (RRT) Planning.
        :param sample_fn: sample function to select a random goal
        :param collision_fn: collision function, test whether it is possible
        to go from one node to another.
        :param goal_check_fn: function to check whether a state achieves the
        final goal or not
        :param action_list: tensor of actions (N_a, d), if using RRT with
        actions (default: None). Each action should be a state difference tensor
        :param expand_dis: If not using actions, distance of tree expansions
        :param goal_sample_rate: probability to sample goal
        """
        self.collision_fn = collision_fn
        self.sample_fn = sample_fn
        self.goal_check_fn = goal_check_fn
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.all_actions = action_list

        self.node_pos = None
        self.node_action = None
        self.node_parents = None

    def planning(self, start, end, max_iter=500, verbose=0):
        """
        Run path planning from start to end.
        :param start: starting node as (tensor)
        :param end: end node as (tensor)
        :param max_iter: number of planning iterations (int)
        :param verbose: verbose level (int)
        :return tuple of (path, path_acts, found_goal, iter_i)
            - path: list of nodes to reach goal
            - path_acts: list of actions to reach goal
            - found_goal: whether goal was found
            - iter_i: number of iterations to find successful plan
        """
        found_goal = False
        self.node_pos = DynamicArray(d=start.shape[1])
        self.node_parents = DynamicArray(d=1)
        if self.all_actions is not None:
            self.node_action = DynamicArray(d=1)
        self.node_pos.append(start)
        if self.all_actions is not None:
            self.node_action.append(torch.tensor([-1.]))
        self.node_parents.append(torch.tensor([-1]))
        gen = range(max_iter)
        if verbose > 0:
            gen = tqdm(gen)
        for iter_i in gen:
            # Random Sampling
            if random.random() > self.goal_sample_rate:
                rnd = self.sample_fn()
            else:
                rnd = end

            # Find nearest node
            nind = self._get_nearest_list_index(rnd)

            # expand tree
            nearest_node = self.node_pos[nind]
            if self.all_actions is not None:
                new_node, act_id = self._expand_tree_actions(rnd, nearest_node)
                if not self.collision_fn(nearest_node, act_id, new_node):
                    continue
            else:
                new_node = self._expand_tree_classic(rnd, nearest_node)
                if not self.collision_fn(nearest_node, new_node):
                    continue

            self.node_pos.append(new_node)
            self.node_parents.append(nind)
            if self.all_actions is not None:
                self.node_action.append(act_id)
            if verbose > 1:
                print("Node list:", len(self.node_pos))

            # check goal
            d = torch.norm(new_node - end, 2)
            if d <= self.expand_dis:
                found_goal = True
                if verbose > 0:
                    print("Goal found")
                break

        # Generate path from tree
        path = []
        path_acts = []
        last_index = len(self.node_pos) - 1
        n = 0
        while self.node_parents[last_index] != -1:
            n += 1
            path.append(self.node_pos[last_index].view(1, -1))
            if self.all_actions is not None:
                path_acts.append(self.node_action[last_index])
            last_index = int(self.node_parents[last_index])
        path.append(start)

        path = torch.cat(path[::-1], dim=0).view(n + 1, -1)
        if len(path_acts) == 0:
            path_acts = torch.tensor(path_acts)
        else:
            path_acts = torch.cat(path_acts[::-1], dim=0).view(n, -1)
        return path, path_acts, found_goal

    def _get_nearest_list_index(self, point):
        """
        Returns index of the closest node to given point
        :param point: given point
        :return: index
        """
        distances = ((self.node_pos.get() - point) ** 2).sum(1)
        return int(torch.argmin(distances))

    def _expand_tree_classic(self, node, nearest_node):
        unit_vec = node[0, :] - nearest_node
        unit_vec /= torch.norm(unit_vec, 2)
        new_node = torch.clone(nearest_node).view(1, -1)
        new_node[0, :] += self.expand_dis * unit_vec

        return new_node

    def _expand_tree_actions(self, node, nearest_node):
        diff = nearest_node + self.all_actions - node[0, :]
        dist = torch.norm(diff, 2, dim=1)
        action_id = torch.argmin(dist)
        new_node = torch.clone(nearest_node).view(1, -1)
        new_node[0, :] += self.all_actions[action_id, :]

        return new_node, action_id

    def draw_graph_2d(self, start=None, end=None, obstacle_list=None,
                      rnd=None, axis_lims=None):
        """
        Draw Graph. Only valid for planning in 2D.
        """
        # plt.clf()
        plt.figure()
        plt.grid(True)
        if axis_lims is not None:
            plt.axis(axis_lims)
        if rnd is not None:
            plt.plot(rnd[0, 0], rnd[0, 1], "^k")

        # Plot obstacles
        if obstacle_list is not None:
            for (ox, oy, size) in obstacle_list:
                plt.plot(ox, oy, "ok", ms=30 * size)

        # Start and end
        if start is not None:
            plt.plot(start[0, 0], start[0, 1], "xb")
        if end is not None:
            plt.plot(end[0, 0], end[0, 1], "xr")

        for i in range(len(self.node_pos)):
            if self.node_parents[i] != -1:
                parent_id = int(self.node_parents[i])
                from_node, to_node = self.node_pos[i], self.node_pos[parent_id]
                plt.plot([from_node[0], to_node[0]], [from_node[1], to_node[1]],
                         "-g")
                # plt.pause(0.001)
