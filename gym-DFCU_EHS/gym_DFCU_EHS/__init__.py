from gym.envs.registration import register

register(
    id='DFCU_EHS-v0',
    entry_point='gym_DFCU_EHS.envs:DFCU_EHSEnv',
)