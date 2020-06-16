import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

np.random.seed(1)
tf.random.set_seed(1)


class MyModel(Model):
    def __init__(self, num_actions):
        super().__init__(name='simple_dqn')
        self.hidden1 = Dense(32, name='dense1', activation='relu', kernel_initializer='he_uniform')
        self.hidden2 = Dense(32, name='dense2', activation='relu', kernel_initializer='he_uniform')
        self.logits = Dense(num_actions, name='logits')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.logits(x)
        return x


class ExperienceReplay:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.experiences = deque(maxlen=self.memory_size)

    def sample(self, batch_size):
        assert batch_size < len(self.experiences)
        minibatch = np.array(random.sample(self.experiences, batch_size))
        return minibatch

    def store(self, experience):
        self.experiences.append(experience)

    @property
    def size(self):
        return len(self.experiences)


class BasicDQNAgent:
    def __init__(
            self,
            online_model, target_network, action_space,
            memory_size=500,
            learning_rate=0.015,
            epsilon=0.05,
            gamma=0.95,
            batch_size=32,
            target_update_interval=400,
            max_episodes=200,
    ):
        self.q_network = online_model
        self.target_network = target_network

        self.q_network.compile(
            optimizer=Adam(learning_rate=learning_rate), loss='mse'
        )

        self.action_space = action_space
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.max_episodes = max_episodes

        self.experience_replay = ExperienceReplay(memory_size=memory_size)

    def train_network(self):
        state_b, action_b, reward_b, next_state_b, done_b = self.experience_replay.sample(self.batch_size).T

        state_b, action_b, next_state_b = np.stack(state_b), np.stack(action_b), np.stack(next_state_b)

        target_q = reward_b + self.gamma * np.amax(self.target_network.predict(next_state_b), axis=1) * (1 - done_b)
        target_v = self.q_network.predict(state_b)
        target_v[range(self.batch_size), action_b] = target_q
        losses = self.q_network.train_on_batch(state_b, target_v)
        return losses

    def epsilon_greedy(self, obs):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.q_network.predict(obs[None]), axis=-1)[0])

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
