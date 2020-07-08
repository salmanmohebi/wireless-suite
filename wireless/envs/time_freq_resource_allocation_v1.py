import numpy as np

from wireless.envs.time_freq_resource_allocation_v0 import TimeFreqResourceAllocationV0


class TimeFreqResourceAllocationV1(TimeFreqResourceAllocationV0):
    def _update_state(self):

        # Channel Quality Indicator (CQI) normalized to [0, 1]
        cqi = self.cqi / 15

        # Normalized sizes in bits of packets in UEs' buffers
        ue_buffers_size = np.sum(self.s, axis=1) / (self.max_pkt_size_bits * self.L)

        # Normalized age of oldest packet size for each UE
        oldest_packet = np.max(self.e, axis=1) / self.t_max

        bi = [300, 30, 150, 100]  # delay budget index
        b = np.array([bi[np.where(r == 1)[0][0]] for r in self.qi]) / np.max(bi)  # Convert QI to delay budget

        self.state = np.concatenate((cqi, ue_buffers_size, oldest_packet, b))
