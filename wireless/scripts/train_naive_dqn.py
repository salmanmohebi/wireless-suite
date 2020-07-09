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
import numpy as np
import matplotlib.pyplot as plt
from wireless.agents.naive_dqn import NaiveDQNAgent

# Load environment parameters
with open('../../config/config_environment.json') as f:
    ec = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_episodes', type=int, default=10)
parser.add_argument('-t', '--max_steps', type=int, default=1000)
parser.add_argument('-p', '--save_path', type=str, default=None)


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)

    def observation(self, observation):
        # Channel Quality Indicator (CQI) normalized to [0, 1]
        cqi = observation[0:self.K] / 15
        # Normalized sizes in bits of packets in UEs' buffers
        s = np.reshape(observation[self.K:self.K * (1 + self.L)], (self.K, self.L))
        ue_buffers_size = np.sum(s, axis=1) / (self.max_pkt_size_bits * self.L)

        # Normalized age of oldest packet size for each UE
        e = np.reshape(observation[self.K * (1 + self.L):self.K * (1 + 2 * self.L)], (self.K, self.L))  # Packet ages in TTIs
        oldest_packet = np.max(e, axis=1) / self.t_max

        bi = [300, 30, 150, 100]  # delay budget index
        qi_ohe = np.reshape(observation[self.K + 2 * self.K * self.L:5 * self.K + 2 * self.K * self.L], (self.K, 4))
        b = np.array([bi[np.where(r == 1)[0][0]] for r in qi_ohe]) / np.max(bi)  # Convert QI to delay budget

        new_obs = np.concatenate((cqi, ue_buffers_size, oldest_packet, b))
        return new_obs


if __name__ == '__main__':
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    env = ObservationWrapper(
        gym.make('TimeFreqResourceAllocation-v0', n_ues=ec['env']['n_ues'],
                 n_prbs=ec['env']['n_prbs'], buffer_max_size=ec['env']['buffer_max_size'],
                 eirp_dbm=ec['env']['eirp_dbm'], f_carrier_mhz=ec['env']['f_carrier_mhz'],
                 max_pkt_size_bits=ec['env']['max_pkt_size_bits'],
                 it=ec['env']['non_gbr_traffic_mean_interarrival_time_ttis']))  # Init environment

    agent = NaiveDQNAgent(env)
    rewards = agent.train(n_episodes=args.n_episodes, max_steps=args.max_steps, save_path=args.save_path)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rewards, label='Rewards', linewidth=1)
    ax.set_ylabel('Accumulated rewards')
    ax.set_xlabel('Episodes')
    plt.savefig(join(args.save_path, 'final_rewards.png'))
