import random
import pytest
import numpy as np
from wireless.utils.experience_replay import ExperienceReplay


class TestReplyMemory:
    def test_store(self):
        size = 10
        memory = ExperienceReplay(size)
        for i in range(1, 20):
            memory.store(np.random.randint(1, 10, 5))
            assert memory.size == min(i, size)

    def test_sample(self):
        random.seed(123)
        np.random.seed(123)
        size = 10
        memory = ExperienceReplay(size)
        for i in range(1, size+1):
            memory.store(np.random.randint(1, 10, 5))
            assert memory.size <= min(i, size)

        state, action, reward, next_state, done = memory.sample(3)
        assert (state == np.array([3, 5, 7])).all()
        assert (action == np.array([3, 8, 2])).all()
        assert (reward == np.array([7, 3, 1])).all()
        assert (next_state == np.array([2, 5, 2])).all()
        assert (done == np.array([4, 9, 1])).all()
