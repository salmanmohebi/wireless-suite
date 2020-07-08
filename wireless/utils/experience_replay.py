import random
import numpy as np


class ExperienceReplay:
    def __init__(self, memory_size):
        self.states, self.actions, self.rewards, self.next_states, self.terminates = [], [], [], [], []
        self.memory_size = memory_size
        self.index = 0

    def sample(self, batch_size):
        assert batch_size < self.index
        idx = random.sample(range(self.index), batch_size)
        return np.array(self.states)[idx], np.array(self.actions)[idx], np.array(self.rewards)[idx], np.array(self.next_states)[idx], np.array(self.terminates)[idx]

    def store(self, experience):
        if self.index == self.memory_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.terminates.pop(0)
            self.index -= 1

        state, action, reward, next_state, done = experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminates.append(done)
        self.index += 1

    @property
    def size(self):
        return self.index
