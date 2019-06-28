import random
import time
import torch
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.rrt import RRT
from lph.envs.Drawer import DrawerEnv
from lph.envs.Baking import BakingEnv
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.envs.RandomGraph import RandomGraphEnv
from lph.utils import SparseState


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


def run_episode(env, agent, goal, n_steps=100, n_rrt_iter=1000):
    seq_a = []
    success = False
    state = env.reset()
    iterator = tqdm(range(n_steps))
    for i in iterator:
        plan = agent.planning(
            torch.tensor(state, dtype=torch.float32).view(1, -1),
            torch.tensor(goal.to_dense(), dtype=torch.float32), n_rrt_iter)[1]
        if len(plan) == 0:
            a = random.randint(0, env.d_s - 1)
        else:
            a = int(plan.detach().cpu().numpy()[0])
        seq_a.append(a)

        # Execute
        env.state = state
        new_state, r, done, info = env.step(a)
        if goal.matches(new_state):
            r, done = 0, True
        else:
            r, done = -1, False
        state = new_state
        if done:
            success = True
            iterator.close()
            break
    return success, seq_a


def run_n_episodes(env, agent, s_goal, n_eps, n_rrt_iter):
    success_buffer = [False] * n_eps
    length_buffer = [n_eps] * n_eps
    time_buffer = [1.0] * n_eps
    for n_ep in tqdm(range(n_eps)):
        t = time.time()
        success, seq_a = run_episode(env, agent, s_goal, n_rrt_iter=n_rrt_iter)
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
        env2 = MiningCraftingEnv(stochastic_reset=False)
    elif env_name == "baking":
        env = BakingEnv(stochastic_reset=False)
        env2 = BakingEnv(stochastic_reset=False)
    elif env_name == "random":
        env = RandomGraphEnv(stochastic_reset=False, noise_prob=0.2)
        env2 = RandomGraphEnv(stochastic_reset=False, noise_prob=0.2)
    elif env_name == "drawer":
        env = DrawerEnv(stochastic_reset=False)
        env2 = DrawerEnv(stochastic_reset=False)
    else:
        raise RuntimeError("Unknown environment name")
    n_rrt_iter = 1000
    goal = env.curriculum[-1]
    s_goal = SparseState(goal[0], goal[1], env.d_s)
    all_a = torch.eye(env.d_s)


    def sample_fn():
        return torch.randint(2, (1, env.d_s)).to(dtype=torch.float32)


    def collision_fn(s, a, s2):
        env.state = s.detach().numpy().reshape(-1, 1)
        new_state = env.step(int(a.detach().numpy()))[0]
        return np.linalg.norm(new_state.T - s2.detach().numpy()) < 1e-5


    agent = RRT(sample_fn, collision_fn, action_list=all_a)

    trajectory_length, time_ = run_n_episodes(env, agent, s_goal, n_runs,
                                              n_rrt_iter)

    print("#### FINAL ####")
    print("number of runs:", n_runs)
    print("number mcts iter:", n_rrt_iter)
    print("trajectory length:", trajectory_length)
    print("time:", time_)
