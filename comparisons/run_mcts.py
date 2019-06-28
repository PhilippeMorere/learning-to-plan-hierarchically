import time
from tqdm import tqdm
import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.mcts import Mcts
from lph.envs.Baking import BakingEnv
from lph.envs.MiningCrafting import MiningCraftingEnv
from lph.envs.RandomGraph import RandomGraphEnv
from lph.utils import SparseState


class MiningState:
    def __init__(self, env, goal):
        self.env = env
        self.goal = goal
        self.state = env.state.copy()

    def get_possible_actions(self):
        return range(self.env.low_a[0], self.env.high_a[0])

    def take_action(self, action):
        self.env.state = self.state
        new_state = self.env.step(action)[0]
        s = MiningState(self.env, self.goal)
        s.state = new_state
        return s

    def is_terminal(self):
        return self.goal.matches(self.state)

    def get_reward(self):
        return 1.0 * self.is_terminal() - 1.0


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


def run_episode(env, agent, goal, n_steps=100):
    seq_a = []
    success = False
    state = env.reset()
    initial_state = MiningState(env, goal)
    iterator = tqdm(range(n_steps))
    for i in iterator:
        initial_state.state = state
        a = agent.search(initial_state=initial_state)
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


def main():
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
    n_mcts_iter = 100
    goal = env.curriculum[-1]
    s_goal = SparseState(goal[0], goal[1], env.d_s)
    agent = Mcts(iteration_limit=n_mcts_iter)
    trajectory_length, time_ = run_n_episodes(env, agent, s_goal, n_runs)

    print("#### FINAL ####")
    print("number of runs:", n_runs)
    print("number mcts iter:", n_mcts_iter)
    print("trajectory length:", trajectory_length)
    print("time:", time_)


if __name__ == "__main__":
    main()
