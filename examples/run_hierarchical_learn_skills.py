import time
from prettytable import PrettyTable
from tqdm import tqdm
import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lph.Condition import FixedCondition
from lph.IntentLearner import IntentLearner
from lph.IntentPlanner import IntentPlanner
from lph.Runner import Runner
from lph.Skill import PrimitiveSkill
from lph.envs.Baking import BakingEnv
from lph.envs.Drawer import DrawerEnv
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.envs.RandomGraph import RandomGraphEnv
from lph.utils import SparseState, Effect

"""
Example running "Learning to plan hierarchically" to learn skills from 
successfully completing the goals of a curriculum.
Success conditions for primitive skills are given.
"""


def train_with_curriculum(env, agent, curriculum, ep_length,
                          n_max_success_per_goal=5, n_max_ep_per_goal=20):
    runner = Runner(env, [], verbose=0)
    runner.il = agent
    n_train_ep = 0
    for i in range(len(curriculum)):
        success = 0
        for ite in range(n_max_ep_per_goal):
            goal = SparseState(curriculum[i][0], curriculum[i][1], env.d_s)
            n_train_ep += 1
            if runner.run(goal, ep_length):
                success += 1
            if success >= n_max_success_per_goal:
                print("Success at skill {} after {}  episodes".format(
                    i, ite + 1))
                break
        if success < n_max_success_per_goal:
            print("Failed at skill {}\n\n".format(i))

    return n_train_ep


def run_episode(env, agent, goal, n_steps=100):
    state = goal
    while goal.matches(state):
        state = env.reset()
    seq_a = []

    for i in range(n_steps):
        # Plan and get first action
        plan = agent.plan(state, goal)
        skill, left_plan = IntentPlanner.behead(plan, False)
        a = IntentPlanner.flat_plan_to_actions([skill])[0]
        seq_a.append(a)

        # Execute
        new_state, r, done, info = env.step(a)

        # Update agent
        effect = Effect.from_dense_start_goal(state, new_state)
        agent.update(state, skill, effect)

        # Goal reached?
        state = new_state
        if goal.matches(state):
            break

    if goal.matches(state):
        return True, seq_a

    return False, seq_a


def compute_stats(success_buffer, length_buffer, time_buffer, n_ep_buffer):
    time_ = 0.0
    length = 0.0
    n_ep = 0.0
    n = 0.0
    for i, success in enumerate(success_buffer):
        if success:
            n += 1.0
            length += length_buffer[i]
            time_ += time_buffer[i]
            n_ep += n_ep_buffer[i]
    if n == 0:
        raise RuntimeError("No success!")
    return length / n, time_ / n, n_ep / n


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


def run_n_episodes(env, primitive_skills, s_goal, n_eps, ep_length,
                   n_max_success_per_goal):
    success_buffer = [False] * n_eps
    length_buffer = [n_eps] * n_eps
    time_buffer = [1.0] * n_eps
    n_ep_buffer = [1.0] * n_eps
    for n_ep in tqdm(range(n_eps)):
        agent = IntentLearner(primitive_skills, planner_max_rec=3, verbose=0)
        n_train_ep = train_with_curriculum(
            env, agent, env.curriculum, ep_length, n_max_success_per_goal)

        t = time.time()
        success, seq_a = run_episode(env, agent, s_goal)
        print(seq_a)

        # Compute stats
        time_buffer = time_buffer[1:len(time_buffer)] + [time.time() - t]
        success_buffer = success_buffer[1:len(success_buffer)] + [success]
        length_buffer = length_buffer[1:len(length_buffer)] + [len(seq_a)]
        n_ep_buffer = n_ep_buffer[1:len(n_ep_buffer)] + [n_train_ep]

        # Print learned skills
        if n_ep == n_eps - 1:
            print_skills(env, agent.skill_base)

    stats = compute_stats(success_buffer, length_buffer, time_buffer,
                          n_ep_buffer)
    return stats[0], stats[1], stats[2]


if __name__ == "__main__":
    n_runs = 1
    env_name = "mining"
    if env_name == "mining":
        env, ep_length = MiningCraftingEnv(stochastic_reset=False), 20
    elif env_name == "baking":
        env, ep_length = BakingEnv(stochastic_reset=False), 40
    elif env_name == "random":
        env = RandomGraphEnv(stochastic_reset=False, noise_prob=0.2)
        ep_length = 100
    elif env_name == "drawer":
        env, ep_length = DrawerEnv(stochastic_reset=False), 10
    else:
        raise RuntimeError("Unknown environment name")

    goal = env.curriculum[-1]
    ds = env.d_s
    n_a = env.high_a[0]
    s_goal = SparseState(goal[0], goal[1], ds)

    n_max_success_per_goal = 1
    primitive_skills = [
        PrimitiveSkill(Effect([i], [0], [1], n_a), i, FixedCondition(
            env.conditions[i], [1.] * len(env.conditions[i])))
        for i in range(n_a)]
    trajectory_length, time_, n_train_ep = run_n_episodes(
        env, primitive_skills, s_goal, n_runs, ep_length,
        n_max_success_per_goal)

    print("#### FINAL ####")
    print("number of training episodes:", n_train_ep)
    print("number of runs:", n_runs)
    print("trajectory length:", trajectory_length)
    print("time:", time_)
