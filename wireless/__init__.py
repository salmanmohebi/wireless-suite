from gym.envs.registration import register

register(
    id='TimeFreqResourceAllocation-v0',
    entry_point='wireless.envs:TimeFreqResourceAllocationV0',
)

register(
    id='TimeFreqResourceAllocation-v1',
    entry_point='wireless.envs:TimeFreqResourceAllocationV1',
)
