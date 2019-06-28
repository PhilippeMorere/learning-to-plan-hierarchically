import numpy as np
from functools import reduce
from operator import add

from lph.utils import Effect


class Trajectory:
    class Node:
        def __init__(self, skill):
            self.skill = skill
            self.parents = []  # A skill requires parent skill first.
            self.children = []

        def add_parent(self, node):
            if node not in self.parents:
                self.parents.append(node)
            if self not in node.children:
                node.children.append(self)

        def remove_parent(self, node):
            if node in self.parents:
                self.parents.remove(node)
            if self in node.children:
                node.children.remove(self)

        def __str__(self):
            return str(self.skill)

    def __init__(self, initial_state, skill_base):
        """
        Creates a trajectory.
        :param initial_state: initial state as dense state
        :param skill_base: agent (SkillBase)
        """
        self.skill_base = skill_base
        self.initial_state = initial_state
        self.leaves = []  # Leaves are skills that are not reused (ie. goal)
        self.roots = []  # Roots a skills that should be executed first
        self.nodes = []

    def update(self, state, action, new_state):
        """
        Extends the trajectory with a state action pair.
        :param state: old state as dense state
        :param action: new action
        :param new_state: new state as dense state
        """
        # Select successful transitions only
        observed_effect = Effect.from_dense_start_goal(state, new_state)
        skill = self.skill_base.primitive_from_action(action)

        # Only add to graph is skill succeeded
        if skill.effect in observed_effect:
            node = Trajectory.Node(skill)
            # Attach node
            for n in self.nodes:
                if n.skill.effect.end_state.matches(skill.get_condition()):
                    node.add_parent(n)
                    if n in self.leaves:  # n is not a leaf anymore
                        self.leaves.remove(n)
            self.leaves.append(node)
            self.nodes.append(node)
            if len(node.parents) == 0:
                self.roots.append(node)

    def refine(self):
        """
        Finds higher-level skills within trajectory and replace them.
        """
        i = -1
        while i < len(self.nodes) - 1:
            i += 1
            n = self.nodes[i]
            skills = self.skill_base.skills_from_effect(n.skill.effect, True)
            if len(skills) == 0:
                continue
            # Compute condition for sub-graph up to node n
            cond = Trajectory.get_parent_conditions(n)
            # Filtering for skills that match the condition and were refined
            matching_skills = [s for s in skills if
                               s.get_condition().matches(cond) and
                               s.refined_seq is not None]
            if len(matching_skills) == 0:
                continue
            skills_len = [len(s) for s in matching_skills]
            s = matching_skills[int(np.argmin(skills_len))]

            # Replace sub-graph with a new node with found skill
            new_n = Trajectory.Node(s)

            del_list = [n for n in self.nodes if n.skill.effect in s.effect]
            self._replace_nodes(del_list, new_n)
            i = -1  # start procedure again

    def _replace_nodes(self, del_list, new_node):
        """
        Replaces the given list of nodes with a new given node.
        :param del_list: (list) of (Node)
        :param new_node: (Node)
        """
        # Make graph connections to new node, remove connections to del_list
        cond = new_node.skill.get_condition()
        for n in del_list:
            for i in range(len(n.parents) - 1, -1, -1):
                parent = n.parents[i]
                # Only add parent if some of its effect is required in
                # new node's conditions.
                req_effect = np.any([d in cond.dims for d in
                                     parent.skill.effect.dims])
                if parent not in new_node.parents and parent not in del_list \
                        and req_effect:
                    new_node.add_parent(parent)
                n.remove_parent(parent)
            for i in range(len(n.children) - 1, -1, -1):
                child = n.children[i]
                if child not in new_node.children and child not in del_list:
                    child.add_parent(new_node)
                child.remove_parent(n)

        # Remove n from node list, leaves, roots
        idx = self.nodes.index(del_list[-1])
        for n in del_list:
            self.nodes.remove(n)
            if n in self.leaves:
                self.leaves.remove(n)
            if n in self.roots:
                self.roots.remove(n)

        # Add new node to node list, and leaves and/or roots?
        idx = max(0, min(idx - len(del_list) - 1, len(del_list) - 1))
        self.nodes.insert(idx, new_node)
        if len(new_node.parents) == 0:
            self.roots.append(new_node)
        if len(new_node.children) == 0:
            self.leaves.append(new_node)

        if len(new_node.parents) > 0 and \
                np.all(new_node.skill.effect.dims == np.array([0, 1, 4, 7])):
            print("gonna blow!")

    @staticmethod
    def all_parents(node):
        """
        Return a list of all parents of given node
        :param node: (Node)
        :return: (list) of (Node)
        """
        remain_parents = list(node.parents)
        all_parents = [node]
        while len(remain_parents) > 0:
            n = remain_parents.pop(0)
            if n not in all_parents:
                all_parents.append(n)
                remain_parents.extend(n.parents)
        return all_parents

    @staticmethod
    def get_parent_conditions(node):
        """
        Return a the condition for given node skill, according to parent skills.
        :param node: (Node)
        :return: conditions as (SparseState)
        """
        if len(node.parents) == 0:
            return node.skill.get_condition()

        cond = node.skill.get_condition()
        if node in node.parents:
            print("problem!")
        for parent in node.parents:
            cond = cond.union(Trajectory.get_parent_conditions(parent))
            cond.remove(parent.skill.effect.end_state)
        return cond

    def to_skill_seq(self, effect):
        """
        Returns a skill sequence with given effect from this trajectory, and
        Note: effect should correspond to one of the trajectory leaf nodes.
        :param effect: trajectory (Effect).
        :return: (list) of (Skill), skill sequence (Effect)
        """
        for r in self.nodes:
            if effect not in r.skill.effect:  # Select node with desired effect
                continue
            node_seq = []
            all_parents = Trajectory.all_parents(r)
            remain_nodes = [n for n in self.roots if n in all_parents]
            while len(remain_nodes) > 0:
                n = remain_nodes.pop(0)
                if len(n.parents) > 0 and \
                        not set(n.parents).issubset(set(node_seq)):
                    remain_nodes.append(n)
                    continue
                node_seq.append(n)
                remain_nodes.extend([c for c in n.children
                                     if c in all_parents and c not in
                                     node_seq and c not in remain_nodes])
            # Compute skill sequence effect
            seq_effect = reduce(add, [p.skill.effect for p in node_seq])
            return [n.skill for n in node_seq], seq_effect

        raise ValueError("Could not find leaf with effect {}".format(effect))
