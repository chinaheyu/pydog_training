from gym.envs.registration import register

register(
	id='OdorEnvA-v0',
	entry_point='odor_env.odor_env:OdorEnvA'
)

register(
	id='OdorEnvB-v0',
	entry_point='odor_env.odor_env:OdorEnvB'
)
