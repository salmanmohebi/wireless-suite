import random
from collections import deque

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


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
            state_size,
            action_space,
            memory_size=10**5,
            learning_rate=1e-3,
            learning_rate_decay=1e-4,
            min_learning_rate=1e-4,
            epsilon=1e-1,
            epsilon_decay=1e-5,
            min_epsilon=1e-4,
            gamma=0.99,
            batch_size=128,
            target_update_interval=600,
            max_episodes=200,
    ):
        self.state_size = state_size
        self.action_space = action_space

        self.lr = learning_rate
        self.lr_decay = learning_rate_decay
        self.min_lr = min_learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.max_episodes = max_episodes

        self.experience_replay = ExperienceReplay(memory_size=memory_size)
        self.q_network = self.build_network()
        self.target_network = self.build_network()

        self.q_network.compile(
            optimizer=Adam(learning_rate=learning_rate), loss='mse'
        )
        self.update_target_network()

    def build_network(self):
        model = Sequential(
            [
                Dense(128, input_shape=(self.state_size,), name='dense1', activation='relu',
                      kernel_initializer='he_uniform'),
                Dense(128, name='dense2', activation='relu', kernel_initializer='he_uniform'),
                Dense(self.action_space.n, name='logits')
            ]
        )
        return model

    def train_network(self):
        state_b, action_b, reward_b, next_state_b, done_b = self.experience_replay.sample(self.batch_size).T

        state_b, action_b, next_state_b = np.stack(state_b), np.stack(action_b), np.stack(next_state_b)

        target_q = reward_b + self.gamma * np.amax(self.target_network.predict(next_state_b.astype(float)), axis=1) * (1 - done_b)
        target_v = self.q_network.predict(state_b)
        target_v[range(self.batch_size), action_b] = target_q
        losses = self.q_network.train_on_batch(state_b, target_v)
        return losses

    def epsilon_greedy(self, obs):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return self.act(obs)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)

    def update_lr(self):
        self.lr = max(self.lr - self.lr_decay, self.min_lr)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, obs, reward=None, done=None):
        return np.argmax(self.q_network.predict(obs[None].astype(float)), axis=-1)[0]
