import time
from tqdm import tqdm
import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lph.Condition import FixedCondition
from lph.IntentLearner import IntentLearner
from lph.IntentPlanner import IntentPlanner
from lph.Skill import PrimitiveSkill, Skill
from lph.envs.Baking import BakingEnv
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.utils import SparseState, Effect

"""
Example running "Learning to plan hierarchically" to plan hierarchically to 
achieve goals of a curriculum.
Primitive skills (and their success conditions) AND abstract skills are given.
"""


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


def compute_stats(success_buffer, length_buffer, time_buffer):
    time_ = 0.0
    length = 0.0
    n = 0.0
    for i, success in enumerate(success_buffer):
        if success:
            n += 1.0
            length += length_buffer[i]
            time_ += time_buffer[i]
    if n == 0:
        raise RuntimeError("No success!")
    return length / n, time_ / n


def run_n_episodes(env, agent, s_goal, n_eps):
    success_buffer = [False] * n_eps
    length_buffer = [n_eps] * n_eps
    time_buffer = [1.0] * n_eps
    for n_ep in tqdm(range(n_eps)):
        t = time.time()
        success, seq_a = run_episode(env, agent, s_goal)
        print(seq_a)

        # Compute stats
        time_buffer = time_buffer[1:len(time_buffer)] + [time.time() - t]
        success_buffer = success_buffer[1:len(success_buffer)] + [success]
        length_buffer = length_buffer[1:len(length_buffer)] + [len(seq_a)]

    stats = compute_stats(success_buffer, length_buffer, time_buffer)
    return stats[0], stats[1]


if __name__ == "__main__":
    n_runs = 1
    env_name = "mining"
    if env_name == "mining":
        env, ep_length = MiningCraftingEnv(stochastic_reset=False), 20
    elif env_name == "baking":
        env, ep_length = BakingEnv(stochastic_reset=False), 40
    else:
        raise RuntimeError("Unknown environment name")
    goal = env.curriculum[-1]
    ds = env.d_s
    n_a = env.high_a[0]
    s_goal = SparseState(goal[0], goal[1], ds)
    primitive_skills = [
        PrimitiveSkill(Effect([i], [0], [1], n_a), i, FixedCondition(
            env.conditions[i], [1.] * len(env.conditions[i])))
        for i in range(n_a)]

    # Manually add skills
    given_skills = []

    if env_name == "mining":
        given_skills.append(
            Skill(Effect([0, 4], [0] * 2, [1] * 2, ds),
                  [primitive_skills[0], primitive_skills[4]], None))
        given_skills.append(
            Skill(Effect([0, 1, 4, 7], [0] * 4, [1] * 4, ds),
                  [given_skills[0], primitive_skills[1], primitive_skills[7]],
                  None))
        given_skills.append(
            Skill(Effect([0, 1, 4, 7, 8, 11], [0] * 6, [1] * 6, ds),
                  [given_skills[1], primitive_skills[8], primitive_skills[11]],
                  None))
        given_skills.append(
            Skill(Effect([0, 1, 4, 7, 8, 9, 11, 12, 14], [0] * 9, [1] * 9, ds),
                  [given_skills[2], primitive_skills[9], primitive_skills[12],
                   primitive_skills[14]],
                  None))
        given_skills.append(
            Skill(Effect([0, 1, 4, 7, 8, 9, 11, 12, 14, 16, 18], [0] * 11,
                         [1] * 11,
                         ds),
                  [given_skills[3], primitive_skills[16], primitive_skills[18]],
                  None))
        given_skills.append(
            Skill(
                Effect([0, 1, 4, 7, 8, 9, 11, 12, 14, 16, 17, 18, 21], [0] * 13,
                       [1] * 13, ds),
                [given_skills[4], primitive_skills[17], primitive_skills[21]],
                None))
    elif env_name == "baking":
        given_skills.append(
            Skill(Effect([20, 21], [0] * 2, [1] * 2, ds),
                  [primitive_skills[20], primitive_skills[21]], None))
        given_skills.append(
            Skill(Effect([12, 13], [0] * 2, [1] * 2, ds),
                  [primitive_skills[12], primitive_skills[13]], None))
        given_skills.append(
            Skill(Effect([0, 1], [0] * 2, [1] * 2, ds),
                  [primitive_skills[0], primitive_skills[1]], None))
        given_skills.append(
            Skill(Effect([10, 11], [0] * 2, [1] * 2, ds),
                  [primitive_skills[10], primitive_skills[11]], None))
        given_skills.append(
            Skill(Effect([8, 9], [0] * 2, [1] * 2, ds),
                  [primitive_skills[8], primitive_skills[9]], None))
        given_skills.append(
            Skill(Effect([6, 7], [0] * 2, [1] * 2, ds),
                  [primitive_skills[6], primitive_skills[7]], None))
        given_skills.append(
            Skill(Effect([4, 5], [0] * 2, [1] * 2, ds),
                  [primitive_skills[4], primitive_skills[5]], None))
        given_skills.append(
            Skill(Effect([17, 18, 19], [0] * 3, [1] * 3, ds),
                  [primitive_skills[17], primitive_skills[18],
                   primitive_skills[19]], None))
        given_skills.append(
            Skill(Effect([0, 1, 2, 3], [0] * 4, [1] * 4, ds),
                  [given_skills[2], primitive_skills[2], primitive_skills[3]],
                  None))
        given_skills.append(
            Skill(Effect([4, 5, 6, 7, 8, 9, 15], [0] * 7, [1] * 7, ds),
                  [given_skills[4], given_skills[5], given_skills[6],
                   primitive_skills[15]],
                  None))
        given_skills.append(
            Skill(Effect([0, 1, 2, 3, 10, 11, 12, 13, 14], [0] * 9,
                         [1] * 9, ds),
                  [given_skills[1], given_skills[3], given_skills[8],
                   primitive_skills[14]],
                  None))  # 10
        given_skills.append(
            Skill(Effect(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                [0] * 17, [1] * 17, ds),
                [given_skills[9], given_skills[10], primitive_skills[16]],
                None))
        given_skills.append(
            Skill(Effect(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 20, 21, 22],
                [0] * 20, [1] * 20, ds),
                [given_skills[0], given_skills[11], primitive_skills[22]],
                None))
        given_skills.append(
            Skill(Effect(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 26, 27, 28],
                [0] * 23, [1] * 23, ds),
                [primitive_skills[26], given_skills[7], given_skills[11],
                 primitive_skills[27], primitive_skills[28]],
                None))
        given_skills.append(
            Skill(Effect(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 21, 22, 23],
                [0] * 24, [1] * 24, ds),
                [given_skills[12], given_skills[7], primitive_skills[23]],
                None))
        given_skills.append(
            Skill(Effect(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 26, 27, 28, 29],
                [0] * 24, [1] * 24, ds),
                [given_skills[13], primitive_skills[29]],
                None))
        given_skills.append(
            Skill(Effect(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 21, 22, 23, 24, 25],
                [0] * 24, [1] * 24, ds),
                [primitive_skills[24], given_skills[14], primitive_skills[25]],
                None))

    agent = IntentLearner(primitive_skills, planner_max_rec=3, verbose=0)
    for s in given_skills:
        agent.skill_base.add_skill(s)
    trajectory_length, time_ = run_n_episodes(env, agent, s_goal, n_runs)

    print("#### FINAL ####")
    print("number of runs:", n_runs)
    print("trajectory length:", trajectory_length)
    print("time:", time_)
