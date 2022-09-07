import numpy as np

from rlba.environment_loop import EnvironmentLoop
from rlba.environments import RandomEnvironment 
from rlba.agents import RandomAgent
from rlba.types import ArraySpec, DiscreteArraySpec

if __name__ == "__main__":

  n_action = 3
  action_dtype = np.int32
  obs_shape = (5, 7)
  obs_dtype = np.float32

  observation_spec = ArraySpec(
      shape=obs_shape,
      dtype=obs_dtype)
  action_spec = DiscreteArraySpec(n_action, dtype=action_dtype)
  env = RandomEnvironment(
    action_spec=action_spec,
    observation_spec=observation_spec,
    seed=0,
  )
  agent = RandomAgent(action_spec, observation_spec, 0)

  loop = EnvironmentLoop(env, agent)
  loop.run(100)
