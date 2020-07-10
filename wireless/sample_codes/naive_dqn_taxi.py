"""
© 2020 UniPd
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import logging
import argparse
from os.path import join
import gym
import numpy as np
import matplotlib.pyplot as plt
from wireless.agents.naive_dqn import NaiveDQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_episodes', type=int, default=1000)
parser.add_argument('-t', '--max_steps', type=int, default=1000)
parser.add_argument('-p', '--save_path', type=str, default='../../models/toy_models/')


class BasicWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(BasicWrapper, self).__init__(env)
        self.n = self.env.observation_space.n

    def observation(self, observation):
        new_obs = np.zeros(self.n)
        new_obs[observation] = 1
        return new_obs


if __name__ == '__main__':
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    env = BasicWrapper(gym.make('Taxi-v3'))

    agent = NaiveDQNAgent(
        env,
        epsilon=0.1,
        min_epsilon=0.01,
        gamma=0.95,
    )

    rewards = agent.train(n_episodes=args.n_episodes, max_steps=args.max_steps, save_path=args.save_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rewards, label='Rewards', linewidth=1)
    ax.set_ylabel('Accumulated rewards')
    ax.set_xlabel('Episodes')
    plt.savefig(join(args.save_path, 'final_rewards.png'))

    rwds = list()
    stn = 100
    for i in range(stn):
        st, rwd, done = env.reset(), 0, False
        while not done:
            action = agent.act(st)
            st, r, done, _ = env.step(action)
            rwd += r
        rwds.append(rwd)
    print(f'Average rewards for {stn}: {np.mean(rwds)}')
