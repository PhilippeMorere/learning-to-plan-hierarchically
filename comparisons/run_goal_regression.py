import time
from tqdm import tqdm
import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lph.Condition import FixedCondition
from lph.IntentLearner import IntentLearner
from lph.IntentPlanner import IntentPlanner
from lph.Skill import PrimitiveSkill
from lph.envs.Baking import BakingEnv
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.envs.RandomGraph import RandomGraphEnv
from lph.utils import SparseState, Effect


def run_episode(env, agent, goal, n_steps=100):
    state = goal
    while goal.matches(state):
        state = env.reset()
    seq_a = []

    for i in range(n_steps):
        # Plan and get first action
        plan = agent.plan(state, goal)
        skill, _ = IntentPlanner.behead(plan)
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
        env = MiningCraftingEnv(stochastic_reset=False)
    elif env_name == "baking":
        env = BakingEnv(stochastic_reset=False)
    elif env_name == "random":
        env = RandomGraphEnv(stochastic_reset=False, noise_prob=0.2)
    else:
        raise RuntimeError("Unknown environment name")

    goal = env.curriculum[-1]
    s_goal = SparseState(goal[0], goal[1], env.d_s)
    n_a = env.high_a[0]
    primitive_skills = [
        PrimitiveSkill(Effect([i], [0], [1], n_a), i, FixedCondition(
            env.conditions[i], [1.] * len(env.conditions[i])))
        for i in range(n_a)]
    agent = IntentLearner(primitive_skills, planner_max_rec=100, verbose=0)
    trajectory_length, time_ = run_n_episodes(env, agent, s_goal, n_runs)

    print("#### FINAL ####")
    print("number of runs:", n_runs)
    print("trajectory length:", trajectory_length)
    print("time:", time_)
