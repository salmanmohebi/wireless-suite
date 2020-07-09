import logging
import datetime
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from wireless.utils.experience_replay import ExperienceReplay


class NaiveDQNAgent:
    def __init__(self, env, memory_size=10**5, learning_rate=1e-3, epsilon=5e-2, min_epsilon=1e-4,
                 gamma=0.99, batch_size=128, target_update_interval=10):

        self.env = env

        self.lr = learning_rate
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        self.experience_replay = ExperienceReplay(memory_size)
        self.q_network = self.build_network(len(self.env.reset()), self.env.action_space.n)
        self.target_network = self.build_network(len(self.env.reset()), self.env.action_space.n)

        self.q_network.compile(
            optimizer=Adam(learning_rate=learning_rate), loss='mse'
        )
        self.update_target_network()

    def build_network(self, input_size, output_size):
        model = Sequential(
            [
                Dense(128, input_shape=(input_size,), name='dense1', activation='relu',
                      kernel_initializer='he_uniform'),
                Dense(128, name='dense2', activation='relu', kernel_initializer='he_uniform'),
                Dense(output_size, name='logits')
            ]
        )
        return model

    def train_network(self):
        #  sample from replay memory
        state_b, action_b, reward_b, next_state_b, done_b = self.experience_replay.sample(self.batch_size)

        target_q = reward_b + self.gamma * np.amax(self.target_network.predict(next_state_b.astype(float)), axis=1) * (1 - done_b)
        target_v = self.q_network.predict(state_b)
        target_v[range(self.batch_size), action_b] = target_q
        losses = self.q_network.train_on_batch(state_b, target_v)
        return losses

    def epsilon_greedy(self, obs):
        return self.env.action_space.sample() if np.random.rand() < self.epsilon else self.act(obs)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, obs, reward=None, done=None):  # compatible with launch agent script
        return np.argmax(self.q_network.predict(obs[None].astype(float)), axis=-1)[0]

    def train(self, n_episodes, max_steps, save_path=None):
        logging.info(f'Start to train agent at {datetime.datetime.now()}')
        init_epsilon, rewards = self.epsilon, list()
        for ep in range(1, n_episodes+1):
            state, ep_reward = self.env.reset(), 0
            for step in range(max_steps):
                action = self.epsilon_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                self.experience_replay.store([state, action, reward, next_state, done])
                state = next_state
                if self.experience_replay.size > 2 * self.batch_size:
                    self.train_network()
                if done:
                    break
            rewards.append(ep_reward)
            if ep % self.target_update_interval:
                self.update_target_network()

            #  Decay epsilon
            decay_factor = max((n_episodes - ep) / n_episodes, 0)
            self.epsilon = (init_epsilon - self.min_epsilon) * decay_factor + self.min_epsilon

            logging.info(f'Episode: {ep}/{n_episodes}, reward: {ep_reward} '
                         f'(mean: {ep_reward / max_steps}, average mean: {np.mean(rewards) / max_steps:2f}), '
                         f'Epsilon: {self.epsilon:2f}'
                         )
            if save_path and ep % 10 == 0:
                self.target_network.save_weights(join(save_path, f'model_{ep}.h5'))
                np.save(join(save_path, f'reward.npy'), rewards)

                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(rewards, label='Rewards', linewidth=1)
                ax.set_ylabel('Accumulated rewards')
                ax.set_xlabel('Episodes')
                plt.savefig(join(save_path, 'fig.png'))
        return rewards
