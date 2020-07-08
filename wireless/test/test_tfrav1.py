"""
© 2020 UniPd
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import pytest
import random

import gym
import numpy as np


@pytest.fixture
def env():
    env = gym.make('TimeFreqResourceAllocation-v1')  # Init environment
    yield env


@pytest.fixture
def env64():
    env = gym.make('TimeFreqResourceAllocation-v1', n_ues=64)  # Init environment
    yield env


class TestTfraV1:
    def test_state_features(self):
        n_ues = 64
        n_steps = 512
        env = gym.make('TimeFreqResourceAllocation-v1', n_ues=n_ues, eirp_dbm=7)  # Low power to have some CQI=0
        env.seed(seed=1234)

        state, _, _, _ = env.step(0)  # Get state to measure its length
        states = np.zeros((n_steps, len(state)), dtype=np.uint32)  # Memory pre-allocation
        for t in range(n_steps):
            action = random.randint(0, n_ues-1)
            state, _, _, _ = env.step(action)
            states[t, :] = state

        assert (states <= 1).all() and (states >= 0).all()
