import gym
import numpy as np
from gym.spaces import Box, Discrete, Tuple
from gym.error import DependencyNotInstalled
import json

all_env_details = dict()

all_envs = gym.envs.registry.all()

for i, env_spec in enumerate(all_envs):
	try:
		env = gym.make(env_spec.id)
		print()
		print(env_spec.id)
		for space_type, space in dict(zip(['ActSpace', 'ObsSpace'], [env.action_space, env.observation_space])).items():
			if isinstance(space, Box):
				print('{} Box: Low {} // High {}'.format(space_type, space.low, space.high))
			elif isinstance(space, Discrete):
				print('{} Discrete: {}'.format(space_type, space.n))
			elif isinstance(space, Tuple):
				print('{} Tuple: {}'.format(space_type, space))
			else:
				print('ERROR')
				quit()
		env.close()
	except DependencyNotInstalled as exception: # Raised for MuJoCo if not installed
		print()
		print(env_spec.id)
		print('MuJoCo')
	if i == 30:
		break
