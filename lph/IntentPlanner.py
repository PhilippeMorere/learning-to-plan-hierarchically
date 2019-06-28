from lph.Skill import PrimitiveSkill
from lph.utils import Effect


class IntentPlanner:
    def __init__(self, skill_base, planner_max_rec, verbose=1):
        self.verbose = verbose
        self.skill_base = skill_base
        self.planner_max_rec = planner_max_rec

    def plan(self, start, goal):
        """
        Plan from start to goal using given skills.
        :param start: starting state (SparseState)
        :param goal: goal state (SparseState)
        :return: A lazy plan (as list) of skills.
        """
        return self._plan(start, goal, 0)

    def _plan(self, start, goal, rec):
        """
        Plan recursively from start to goal using given skills.
        :param start: starting state (SparseState)
        :param goal: goal state (SparseState)
        :param rec: recursion order
        :return: A lazy plan (as list) of skills.
        """
        if rec >= self.planner_max_rec:
            # This happens if goal is "too complicated".
            raise RuntimeError("Planning failed. Rec==3.")

        if self.verbose > 2:
            print("Planning from {} to {}".format(start, goal))

        if len(goal.dims) == 0 or goal.matches(start):
            # This happens when conditions are not yet learned or wrong.
            # print("Planning for random skill")
            main_skill = self.skill_base.random_skill()
        else:
            # Find the skill with the closest effect (start to goal); this is
            # the end skill of a sequence of skills.
            intended_eff = Effect.from_sparse_start_goal(start, goal)
            main_skill = self.skill_base.find_closest_skill(start, intended_eff)

        # If necessary, find a skill that meets the condition of the main skill
        if not main_skill.fails_in(start):
            return [main_skill]
        else:
            # Fine-tune before main skill
            plan = self._plan(start, main_skill.get_condition(), rec + 1)
            plan.append(main_skill)
            return plan

    @staticmethod
    def flatten_plan(plan):
        """
        Reduces plan to list of primitive skills.
        :param plan: list of (Skill)
        :return: list of (primitiveSkill)
        """
        # Is the plan already flattened: (PrimitiveSkill) only?
        is_flat = True
        for skill in plan:
            if not isinstance(skill, PrimitiveSkill):
                is_flat = False
                break
        if is_flat:
            return plan

        prim_skills = []
        for skill in plan:
            prim_skills_sub = IntentPlanner.flatten_plan(skill.flatten())
            prim_skills.extend(prim_skills_sub)

        return prim_skills

    @staticmethod
    def flat_plan_to_actions(flat_plan):
        """
        Converts a flat skill plan to action id sequence
        :param flat_plan: list of (PrimitiveSkill)
        :return: list of action id (int)
        """
        return [skill.action_id for skill in flat_plan]

    @staticmethod
    def behead(plan, randomness=False):
        """
        Separates the head (PrimitiveSkill) of the plan from the rest, returned
        as a tuple.
        :param plan: list of (Skill)
        :param randomness: Whether to enable randomly removing elements from
        trajectory when flattening plans (bool).
        :return: first (PrimitiveSkill), rest of plan as list of (Skill)
        """
        if isinstance(plan[0], PrimitiveSkill):
            return plan[0], plan[1:]
        else:
            plan = plan[0].flatten(randomness) + plan[1:]
            return IntentPlanner.behead(plan)
