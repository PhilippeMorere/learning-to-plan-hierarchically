import numpy as np
from functools import reduce
from operator import add

from lph.Trajectory import Trajectory
from lph.utils import SparseState
from lph.Condition import LearnedCondition


class SkillBase:
    def __init__(self, primitive_skills):
        """
        Initialize skill base.
        :param primitive_skills: list of (PrimitiveSkill)
        """
        self.primitive_skills = primitive_skills
        self.all_skills = {}
        for skill in primitive_skills:
            self.add_skill(skill)

    def add_skill(self, skill):
        """
        Adds skill to skill base.
        :param skill: (Skill)
        :return: None
        """
        if isinstance(skill, PrimitiveSkill):
            k = skill.effect
            values = self.all_skills.get(k, [])
            values.append(skill)
            self.all_skills[k] = values
        else:
            lengths = [len(traj) for traj in skill.success_skill_seqs]
            l_skill = len(skill.success_skill_seqs[int(np.argmin(lengths))])
            for k in skill.effect.unit_effects():
                # If there's already an efficient skills for this effect,
                # don't add a new one.
                skills = self.skills_from_effect(k, True)
                # ref_skills = [s for s in skills if s.refined_seq is not None]
                ref_skills_lengths = [len(s.flatten(False)) for s in skills]
                if len(ref_skills_lengths) > 0 and \
                        l_skill > np.min(ref_skills_lengths):
                    continue
                values = self.all_skills.get(k, [])
                values.append(skill.copy())
                self.all_skills[k] = values

    def remove_skill(self, effect, skill):
        """
        Removes skill from skill base.
        :param effect: skill (Effect)
        :param skill: (Skill) to remove
        """
        values = self.all_skills.get(effect, [])
        values.remove(skill)
        self.all_skills[effect] = values

    def skills_from_effect(self, effect, non_primitive=False):
        """
        Returns a list of all skills with given effect.
        :param effect: (Effect)
        :param non_primitive: Whether to ignore primitive skills (bool)
        :return: list of skills (empty list if no match)
        """
        skills = self.all_skills.get(effect, [])
        if non_primitive:
            return [s for s in skills if not isinstance(s, PrimitiveSkill)]
        else:
            return skills

    def primitive_from_action(self, action_id):
        """
        Returns the primitive skill corresponding to action id.
        :param action_id: action id (int)
        :return: (PrimitiveSkill)
        """
        return self.primitive_skills[int(action_id)]

    def find_closest_skill(self, start, effect):
        """
        Returns skill with the closest effect to that specified,
        and starting conditions closest to given start state.
        :param effect: target (Effect)
        :param start: starting (SparseState)
        :return: closest (Skill) to given effect
        """
        # Effect similarity lookup
        all_effects = list(self.all_skills.keys())
        sims = [effect.similarity_to(e) for e in all_effects]
        close_skills = []
        for idx in np.flatnonzero(sims == np.max(sims)):
            close_skills.extend(self.all_skills[all_effects[idx]])
        close_skills = close_skills[::-1]  # Prefer higher-level skills

        # Start state similarity lookup
        dists = [sk.get_condition().distance_from(start) for sk in close_skills]
        closest_skill = close_skills[int(np.argmin(dists))]
        return closest_skill

    def random_skill(self):
        """
        Returns random skill.
        :return: random (Skill)
        """
        idx = np.random.randint(0, len(self.all_skills))
        skills = list(self.all_skills.values())[idx]
        return np.random.choice(skills)


class Skill:
    def __init__(self, effect, skill_seq, init_state):
        """
        Creates higher-level skill.
        :param effect: skill effect (Effect)
        :param skill_seq: (list) of (Skill)
        :param init_state: trajectory initial state as dense state
        """
        self.effect = effect
        self.success_skill_seqs = [skill_seq]
        self.success_init_states = [init_state]
        self.refined_seq = None

    def copy(self):
        s = Skill(self.effect.copy(), None, None)
        s.success_skill_seqs = list(self.success_skill_seqs)
        s.success_init_states = list(self.success_init_states)
        s.refined_seq = self.refined_seq
        return s

    def add_successful_skill_seq(self, seq_effect, skill_seq, init_state):
        """
        Adds successful trajectory to skill.
        :param seq_effect: skill sequence (Effect)
        :param skill_seq: (list) of (Skill)
        :param init_state: trajectory initial state as dense state
        """
        # If the new trajectory has a more specific effect, keep it instead
        # of previous ones.
        if len(seq_effect.dims) < len(self.effect.dims):
            self.effect = seq_effect
            self.success_skill_seqs = [skill_seq]
            self.success_init_states = [init_state]
        elif skill_seq not in self.success_skill_seqs:
            self.success_skill_seqs.append(skill_seq)
            self.success_init_states.append(init_state)

    def get_condition(self):
        """
        Returns the necessary conditions for skill to succeed.
        :return: conditions as (SparseState)
        """
        plan = self.flatten(randomness=False)
        cond = plan[-1].get_condition().copy()
        for i in range(len(plan) - 2, -1, -1):
            cond.remove(plan[i].effect.end_state)
            cond = cond.union(plan[i].get_condition())
        return cond

    def fails_in(self, state):
        """
        Returns whether the skill has previously failed in this state.
        :param state: start state (SparseState)
        :return: bool
        """
        condition = self.get_condition()
        return not condition.matches(state.to_dense().reshape(-1))

    def _extract_condition(self, start_state, skill_seq):
        """
        Extract skill condition using successful trajectory
        :param start_state: starting state as dense state
        :param skill_seq: sequence of skills as (list) of (Skill)
        """
        # For non-primitive skill, conditions of sub-skills in the successful
        # trajectory can be used to infer necessary conditions.
        conditions = SparseState([], [], len(self.effect.end_state))
        # Combine all sub-skill conditions
        for sub_skill in skill_seq:
            conditions = conditions.union(sub_skill.get_condition())
        # Remove all sub-skill effects (except last)
        for sub_skill in skill_seq[:-1]:
            conditions.remove(sub_skill.effect.end_state)
        conditions.values = np.clip(conditions.values, 0., 1.)  # Get rid of -1
        valid_state = start_state.copy()
        valid_state[conditions.dims] = conditions.values

    def flatten(self, randomness=True):
        """
        Returns a sequence of sub-skills to achieve skill.
        Note: at the moment, a skill just replays the shortest demonstration.
        :param randomness: Whether to enable randomly removing elements
        from trajectory (bool).
        :return: (list) of (Skill)
        """
        if self.refined_seq is not None:
            return self.refined_seq

        lengths = [len(trajectory) for trajectory in self.success_skill_seqs]
        shortest_traj= self.success_skill_seqs[int(np.argmin(lengths))]

        # Randomly removing elements from trajectory helps learning skill
        # conditions.
        if randomness and np.random.rand() < 0.5:
            # randomly remove one element from trajectory
            i = np.random.randint(0, len(shortest_traj))
            return shortest_traj[0:i] + shortest_traj[i + 1:len(shortest_traj)]
        else:
            return shortest_traj

    def refine(self, desired_effect, skill_base):
        """
        Try and refine skill to learn an optimal skill sequence using previous
        successful trajectories, once skill sequence conditions are learned.
        This is done by reasoning on graph conditions.
        :param desired_effect: desired skill (Effect)
        :param skill_base: (SkillBase)
        """
        # Start with shortest successful sequence
        lengths = [len(traj) for traj in self.success_skill_seqs]
        id_shortest = int(np.argmin(lengths))
        shortest_skill_seq = self.success_skill_seqs[id_shortest]

        # Don't refine skill if all conditions are not learned?
        for skill in shortest_skill_seq:
            if not skill.conditions.is_ready():
                return

        # Create a trajectory by simulating running the skill
        s = self.success_init_states[id_shortest]
        traj = Trajectory(s, skill_base)
        for skill in shortest_skill_seq:
            s_new = np.array(s)
            s_new[skill.effect.dims] = skill.effect.end_state.values
            traj.update(s, skill.action_id, s_new)
            s = s_new

        # Refine trajectory
        self.refined_seq = None
        traj.refine()
        new_seq, new_effect = traj.to_skill_seq(desired_effect)
        len_new_seq = reduce(add, [len(s) for s in new_seq])
        if len(new_seq) == 1:  # Useless skill
            skill_base.remove_skill(desired_effect, self)
        elif len_new_seq <= len(shortest_skill_seq):
            self.refined_seq, self.effect = new_seq, new_effect

    def __str__(self):
        return "Learned skill (effect={}, conditions=<disabled>)".format(
            self.effect)  # , self.get_condition())

    def __len__(self):
        """
        Return total skill trajectory length.
        :return: skill length as (int)
        """
        plan = self.flatten(False)
        return reduce(add, [len(s) for s in plan])


class PrimitiveSkill:
    def __init__(self, effect, action_id, conditions=None):
        """
        Creates primitive skill, bound to an action id.
        :param effect: skill effect (Effect)
        :param action_id: action id (int)
        :param conditions: Conditions for successful execution of skill. If
        conditions are None, learned from experience. (default: None)
        """
        self.effect = effect
        self.action_id = action_id
        if conditions is None:
            self.conditions = LearnedCondition()
        else:
            self.conditions = conditions

    def refine(self, start_state):
        """
        Refines skill using successful trajectory.
        :param start_state: starting state as dense state
        :return: None
        """
        self.update_conditions(start_state, True)

    def update_conditions(self, start, success):
        """
        Updates skill success conditions.
        :param start: start state as dense state
        :param success: whether the skill was successful (bool)
        :return: None
        """
        self.conditions.update(start.reshape(-1), success)

    def get_condition(self):
        """
        Returns the necessary conditions for skill to succeed.
        :return: conditions as (SparseState)
        """
        # Logistic regression weights reflect which dimensions are important
        ds = len(self.effect.end_state)
        dims, values = self.conditions.sample()
        return SparseState(dims, values, ds)

    def fails_in(self, state):
        """
        Returns whether the skill has previously failed in this state.
        :param state: start state (SparseState)
        :return: bool
        """
        return self.conditions.fails_in(state.to_dense().reshape(-1))

    def flatten(self):
        return [self]

    def __str__(self):
        return "PrimitiveSkill (action={})".format(self.action_id)

    def __len__(self):
        """
        Return total skill trajectory length (=1 for primitive skill).
        :return: skill length as (int)
        """
        return 1
