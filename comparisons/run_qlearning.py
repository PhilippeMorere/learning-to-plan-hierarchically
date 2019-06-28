import time
from tqdm import tqdm
import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.Qlearning import QLearning
from lph.envs.Baking import BakingEnv
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.envs.RandomGraph import RandomGraphEnv
from lph.utils import SparseState


def run_episode(env, agent, goal, n_steps=100, verbose=1, train=True):
    """
    Runs agent for one episode.
    :param goal: Goal as (SparseState)
    :param n_steps: Maximum number of step for episode (int)
    :return: Whether goal was successfully reached (bool)
    """
    state = goal
    success = False
    while goal.matches(state):
        state = env.reset()

    seq_a = []
    for i in range(n_steps):
        a = agent.policy(state.reshape(-1))
        seq_a.append(a)

        # Execute
        new_state, r, done, info = env.step(a)
        if goal.matches(new_state):
            r, done = 0, True
        else:
            r, done = -1, False
        if verbose > 2:
            print("Transition: {} + {} -> {}".format(
                state.T, a, new_state.T))

        # Update agent
        if train:
            agent.update(state.reshape(-1), a, r, new_state.reshape(-1))

        # Goal reached?
        state = new_state
        if done:
            success = True
            break
    return success, seq_a


def compute_stats(success_buffer, length_buffer, time_buffer):
    time_ = 0.0
    length = 0.0
    n = 0.0
    for i, success in enumerate(success_buffer):
        if success:
            n += 1.0
            length += length_buffer[i]
            time_ += time_buffer[i]
    return length / n, time_ / n


def train(env, agent, s_goal, n_max_step, ratio_success=0.95):
    n_max_ep = 5000
    success_buffer = [False] * 20
    length_buffer = [n_max_step] * 20
    time_buffer = [1.0] * 20
    iterator = tqdm(range(n_max_ep))
    n_train_ep = n_max_ep
    for n_ep in iterator:
        t = time.time()
        success, seq_a = run_episode(env, agent, s_goal, n_max_step, verbose=1)

        # Compute stats
        time_buffer = time_buffer[1:len(time_buffer)] + [time.time() - t]
        success_buffer = success_buffer[1:len(success_buffer)] + [success]
        length_buffer = length_buffer[1:len(length_buffer)] + [len(seq_a)]

        if sum(success_buffer) >= ratio_success * len(success_buffer):
            print("####### Success after {} episodes\n\n".format(n_ep + 1))
            n_train_ep = n_ep
            iterator.close()
            break
    stats = compute_stats(success_buffer, length_buffer, time_buffer)
    return n_train_ep, stats[0], stats[1]


def run(env, goal):
    all_a = range(env.low_a[0], env.high_a[0])
    agent = QLearning(all_a, epsilon=0.1, gamma=0.99, learning_rate=0.3)
    n_max_step = 100
    n_eps, trajectory_length, time_ = train(env, agent, goal, n_max_step)
    print("training episodes:", n_eps)
    print("trajectory length:", trajectory_length)
    print("time:", time_)
    return n_eps, trajectory_length, time_


if __name__ == "__main__":
    n_runs = 1
    envs = {"mining": (MiningCraftingEnv, {"stochastic_reset": False}),
            "baking": (BakingEnv, {"stochastic_reset": False}),
            "random": (RandomGraphEnv, {"stochastic_reset": False,
                                        "noise_prob": 0.2})}

    # Create env
    env_name = "mining"
    curriculum = envs[env_name][0].curriculum
    d_s = len(envs[env_name][0].conditions)
    s_goal = SparseState(curriculum[-1][0], curriculum[-1][1], d_s)
    env = envs[env_name][0](**(envs[env_name][1]), goal=s_goal)

    n_eps = [0] * n_runs
    trajectory_length = [0] * n_runs
    time_ = [0] * n_runs
    for i in range(n_runs):
        n_eps[i], trajectory_length[i], time_[i] = run(env, s_goal)

    print("#### FINAL ####")
    print("training episodes:", sum(n_eps) / float(n_runs))
    print("trajectory length:", sum(trajectory_length) / float(n_runs))
    print("time:", sum(time_) / float(n_runs))
