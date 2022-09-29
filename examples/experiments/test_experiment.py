# Copyright 2022 Morteza Ibrahimi. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""A simple test experiment with random environment  and agent."""


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

    observation_spec = ArraySpec(shape=obs_shape, dtype=obs_dtype)
    action_spec = DiscreteArraySpec(n_action, dtype=action_dtype)
    env = RandomEnvironment(
        action_spec=action_spec,
        observation_spec=observation_spec,
        seed=0,
    )
    agent = RandomAgent(action_spec, observation_spec, 0)

    loop = EnvironmentLoop(env, agent)
    loop.run(100)
