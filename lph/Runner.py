from prettytable import PrettyTable
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.Condition import FixedCondition
from lph.IntentLearner import IntentLearner
from lph.IntentPlanner import IntentPlanner
from lph.Skill import PrimitiveSkill
from lph.Trajectory import Trajectory
from lph.utils import SparseState, Effect


class Runner:
    def __init__(self, env, primitive_skills, verbose=1):
        """
        Creates a runner object.
        :param env: a Gym environment
        :param primitive_skills: (list) of (PrimitiveSkill)
        :param verbose: verbose level (int)
        """
        self.il = IntentLearner(primitive_skills, verbose=verbose)
        self.env = env
        self.d_s = len(self.env.high_s)
        self.d_a = len(self.env.high_a)
        self.verbose = verbose

    def run(self, goal, n_steps=100):
        """
        Runs agent for one episode.
        :param goal: Goal as (SparseState)
        :param n_steps: Maximum number of step for episode (int)
        :return: Whether goal was successfully reached (bool)
        """
        state = goal
        while goal.matches(state):
            state = self.env.reset()
        trajectory = Trajectory(state.reshape(-1), self.il.skill_base)

        for i in range(n_steps):
            # Plan and get first action
            plan = self.il.plan(state, goal)
            skill, _ = IntentPlanner.behead(plan)
            a = IntentPlanner.flat_plan_to_actions([skill])[0]

            # Execute
            new_state, r, done, info = self.env.step(a)
            if self.verbose > 2:
                print("Transition: {} + {} -> {}".format(
                    state.T, a, new_state.T))

            # Update agent
            effect = Effect.from_dense_start_goal(state, new_state)
            self.il.update(state, skill, effect)

            # Save trajectory info
            trajectory.update(state.reshape(-1), a, new_state.reshape(-1))

            # Goal reached?
            state = new_state
            if goal.matches(state):
                break

        if goal.matches(state):
            # Learn from trajectory
            if self.verbose > 0:
                print("Learning from successful trajectory.")
            self.il.learn_demonstration(trajectory, goal)
            success = True
        else:
            # Learn from trajectory
            if self.verbose > 0:
                print("Failed trajectory.")
            success = False

        self.il.end_of_ep_update()
        return success


def print_skills(env, skill_base):
    # Primitive skills
    x1 = PrettyTable()
    x1.field_names = ["Type", "Effect", "Condition", "Real condition"]
    for skill in skill_base.primitive_skills:
        cond = skill.get_condition()
        cond_str = "d={}, v={}".format(cond.dims, cond.values)
        if len(cond.dims) == 0:
            cond_str = "None"
        real_cond = env.conditions[skill.action_id]
        real_str = "d={}, v={}".format(real_cond, [1] * len(real_cond))
        if len(real_cond) == 0:
            real_str = "None"
        x1.add_row(["Prim", "d=[{}], v=[1.]".format(skill.action_id), cond_str,
                    real_str])
    print("Primitive skills:")
    print(x1)

    # Higher skills
    x = PrettyTable()
    x.field_names = ["Type", "From", "To", "Condition", "Refined", "Policy",
                     "Flat policy"]
    all_skills = set([s for sl in skill_base.all_skills.values() for s in sl
                      if s not in skill_base.primitive_skills])
    ordered_skills = sorted(list(all_skills), key=lambda s: len(s.effect.dims))
    for skill in ordered_skills:
        cond = skill.get_condition()
        cond_str = "d={}, v={}".format(cond.dims, cond.values)
        if len(cond.dims) == 0:
            cond_str = "None"
        flat_pol = IntentPlanner.flat_plan_to_actions(
            IntentPlanner.flatten_plan(skill.flatten(randomness=False)))
        pol = ", ".join(["action={}".format(s.action_id)
                         if isinstance(s, PrimitiveSkill)
                         else "Skill:(d={})".format(s.effect.end_state.dims)
                         for s in skill.flatten(randomness=False)])
        x.add_row(["Abs", "d={}, v={}".format(skill.effect.start_state.dims,
                                              skill.effect.start_state.values),
                   "d={}, v={}".format(skill.effect.end_state.dims,
                                       skill.effect.end_state.values),
                   cond_str, skill.refined_seq is not None, pol, flat_pol])
    print("Learned skills:")
    print(x)


def main():
    env = MiningCraftingEnv(stochastic_reset=False)
    d_s = env.d_s
    n_a = env.high_a[0]
    given_conditions = False
    # Learning conditions from data  or fixed conditions?
    if given_conditions:
        primitive_skills = [
            PrimitiveSkill(Effect([i], [0], [1], n_a), i, FixedCondition(
                env.conditions[i], [1.] * len(env.conditions[i])))
            for i in range(n_a)]
    else:
        primitive_skills = [
            PrimitiveSkill(Effect([i], [0], [1], n_a), i, None)
            for i in range(n_a)]

    runner = Runner(env, primitive_skills, verbose=1)

    goals = [
        MiningCraftingEnv.goal_stick,
        MiningCraftingEnv.goal_stone_pick,
        MiningCraftingEnv.goal_coal,
        MiningCraftingEnv.goal_furnace,
        MiningCraftingEnv.goal_smelt_iron,
        MiningCraftingEnv.goal_iron_pick,
        MiningCraftingEnv.goal_gold_ore,
        MiningCraftingEnv.goal_goldware,
        MiningCraftingEnv.goal_necklace,
        MiningCraftingEnv.goal_earrings
    ]
    n_success = 8
    for i in range(len(goals)):
        success = 0
        for ite in range(20):
            goal = SparseState(goals[i][0], goals[i][1], d_s)
            if runner.run(goal, 20):
                success += 1
            if success >= n_success:
                print("####### Success at skill {} after {} "
                      "episodes\n\n".format(i, ite + 1))
                break
        if success < n_success:
            print("####### Failed at skill {}\n\n".format(i))

    print("\n\n\n\n%%%%%%% END")
    print_skills(env, runner.il.skill_base)


if __name__ == "__main__":
    import numpy as np

    seed = np.random.randint(0, 100000000)
    # seed = 11872354
    print("Seed: {}".format(seed))
    np.random.seed(seed)
    main()
