from functools import reduce
from operator import add

from lph.IntentPlanner import IntentPlanner
from lph.Skill import SkillBase, Skill, PrimitiveSkill
from lph.utils import SparseState, Effect


class IntentLearner:
    def __init__(self, primitive_skills, planner_max_rec=3, verbose=1):
        """
        Initialize intent learner.
        :param primitive_skills: list of (PrimitiveSkill)
        :param planner_max_rec: planner recursion order (default: 3). Tougher
        skill curriculum requires higher planner recursion order.
        :param verbose: verbose level (default: 1)
        """
        self.skill_base = SkillBase(primitive_skills)
        self.planner = IntentPlanner(self.skill_base, planner_max_rec, verbose)
        self.current_plan = None
        self.verbose = verbose

    def plan(self, state, goal):
        """
        Generates a plan from state to goal.
        :param state: Starting state (dense)
        :param goal: Goal as (SparseState)
        :return:
        """
        s_start = SparseState.from_dense_state(state)
        if self.current_plan is None:
            try:
                self.current_plan = self.planner.plan(s_start, goal)
            except RuntimeError:
                self.current_plan = [self.skill_base.random_skill()]
                if self.verbose > 1:
                    print("Planner failed (low rec), random action.")
                return self.current_plan

        # If next skill's effect is already satisfied, remove it
        plan = self.current_plan
        while len(plan) > 0:
            skill, new_plan = IntentPlanner.behead(plan, randomness=False)
            if not skill.effect.end_state.matches(s_start):
                break
            plan = new_plan
        self.current_plan = plan

        # # If the next skill can't be executed, execute random action
        # if len(self.current_plan) > 0:
        #     first_skill = IntentPlanner.behead(plan)[0]
        #     if first_skill.fails_in(s_start):
        #         # self.current_plan = []
        #         # Random action
        #         self.current_plan = [self.skill_base.random_skill()]
        #         if self.verbose > 1:
        #             print("Random action2")
        #         return self.current_plan

        # If no plan left, try to plan again
        if len(self.current_plan) == 0:
            self.current_plan = None
            return self.plan(state, goal)

        return self.current_plan

    def update(self, state, executed_skill, effect):
        """
        Updates agent with latest intent and transition.
        :param state: starting dense state
        :param executed_skill: (PrimitiveSkill) executed by the agent
        :param effect: observed transition effect as (Effect)
        :return: None
        """
        # Assess if last executed skill was successful
        successful_execution = (effect == executed_skill.effect)

        # Update executed skill conditions
        executed_skill.update_conditions(state, successful_execution)

        # If fail, execute a random action next time, then re-plan completely
        if not successful_execution:
            self.current_plan = [self.skill_base.random_skill()]
        else:  # Remove first action from plan
            _, self.current_plan = IntentPlanner.behead(self.current_plan)
            if len(self.current_plan) == 0:
                self.current_plan = None

    def learn_demonstration(self, trajectory, goal):
        """
        Learn from demonstration.
        :param trajectory: (Trajectory)
        :param goal: demonstration goal as (SparseState)
        :return: None
        """
        # Identify if existing skills match the trajectory and goal
        s_start = SparseState.from_dense_state(trajectory.initial_state)
        effect = Effect.from_sparse_start_goal(s_start, goal)
        skills = self.skill_base.skills_from_effect(effect)
        s_init = SparseState.from_dense_state(trajectory.initial_state)
        candidate_skills = [s for s in skills if not s.fails_in(s_init)]

        # trajectory.refine()
        # skill_seq, seq_effect = trajectory.to_skill_seq(effect)
        skill_seq = [n.skill for n in trajectory.nodes]
        seq_effect = reduce(add, [s.effect for s in skill_seq])

        # If none found, create a new skill
        if len(candidate_skills) == 0:
            # Learn new skill
            new_skill = Skill(seq_effect, skill_seq, trajectory.initial_state)
            self.skill_base.add_skill(new_skill)
        else:
            for skill in candidate_skills:
                if not isinstance(skill, PrimitiveSkill):
                    skill.add_successful_skill_seq(seq_effect, skill_seq,
                                                   trajectory.initial_state)

    def end_of_ep_update(self):
        """
        Performs updates at the end of each episode (such as refining skills).
        """
        for effect, skill_list in self.skill_base.all_skills.items():
            for skill in skill_list:
                if not isinstance(skill, PrimitiveSkill):
                    skill.refine(effect, self.skill_base)
