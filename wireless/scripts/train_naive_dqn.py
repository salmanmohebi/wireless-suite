"""
© 2020 UniPd
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import json
import logging
import argparse
from os.path import join
import gym
import matplotlib.pyplot as plt
from wireless.agents.naive_dqn import NaiveDQNAgent

# Load environment parameters
with open('../../config/config_environment.json') as f:
    ec = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_episodes', type=int, default=300)
parser.add_argument('-t', '--max_steps', type=int, default=1000)
parser.add_argument('-p', '--save_path', type=str, default=None)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    env = gym.make('TimeFreqResourceAllocation-v1', n_ues=ec['env']['n_ues'],
                   n_prbs=ec['env']['n_prbs'], buffer_max_size=ec['env']['buffer_max_size'],
                   eirp_dbm=ec['env']['eirp_dbm'], f_carrier_mhz=ec['env']['f_carrier_mhz'],
                   max_pkt_size_bits=ec['env']['max_pkt_size_bits'],
                   it=ec['env']['non_gbr_traffic_mean_interarrival_time_ttis'])  # Init environment

    agent = NaiveDQNAgent(env)
    rewards = agent.train(n_episodes=args.n_episodes, max_steps=args.max_steps, save_path=args.save_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rewards, label='Rewards', linewidth=1)
    ax.set_ylabel('Accumulated rewards')
    ax.set_xlabel('Episodes')
    plt.savefig(join(args.save_path, 'final_rewards.png'))
