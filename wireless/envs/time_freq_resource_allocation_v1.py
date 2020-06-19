import numpy as np

from wireless.envs.time_freq_resource_allocation_v0 import TimeFreqResourceAllocationV0


class TimeFreqResourceAllocationV1(TimeFreqResourceAllocationV0):
    def _update_state(self):
        state = np.concatenate((self.cqi, self.s.flatten(), self.e.flatten(), self.qi.flatten(), [self.p]))
        cqi = state[0:self.K] / 15  # Channel Quality Indicator (CQI) normalized into [0, 1]
        # Sizes in bits of packets in UEs' buffers
        s = np.reshape(state[self.K:self.K * (1 + self.L)], (self.K, self.L))
        # TODO: use number of packet instead of buffer size? maybe!
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[self.K * (1 + self.L):self.K * (1 + 2 * self.L)], (self.K, self.L))  # Packet ages in TTIs
        oldest_packet = np.max(e, axis=1)/self.t_max  # Age of oldest packet for each UE

        qi_ohe = np.reshape(state[self.K + 2 * self.K * self.L:5 * self.K + 2 * self.K * self.L], (self.K, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  # Decode One-Hot-Encoded QIs

        # Extract packet delay budget for all UEs
        # TODO: no need to know about delay budget or maybe its better to used normalized b instead to get more sensible information about the delay not just an index
        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        new_state = np.concatenate((cqi, buffer_size_per_ue, oldest_packet, b/300))

        self.state = new_state
