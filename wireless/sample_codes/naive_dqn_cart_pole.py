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
parser.add_argument('-e', '--n_episodes', type=int, default=100)
parser.add_argument('-t', '--max_steps', type=int, default=200)
parser.add_argument('-p', '--save_path', type=str, default='../../models/toy_models/')


if __name__ == '__main__':
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('CartPole-v0')

    agent = NaiveDQNAgent(env)
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
