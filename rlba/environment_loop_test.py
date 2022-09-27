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

"""Tests for the environment loop."""


from rlba import environment_loop
from rlba import types
from rlba.testing import fakes
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

EPISODE_LENGTH = 10

# Action specs
I32_NVAL_1 = types.DiscreteArraySpec(num_values=1, dtype=np.int32)
I32_NVAL_5 = types.DiscreteArraySpec(num_values=5, dtype=np.int32)
TREE_DAS = {"a": I32_NVAL_1, "b": I32_NVAL_5}

# Observation specs
F32 = types.ArraySpec(dtype=np.float32, shape=())
F32_1x3 = types.ArraySpec(dtype=np.float32, shape=(1, 3))
TREE_AS = {"a": F32, "b": F32_1x3}

TEST_CASES = (
    ("single_action_scalar_reward", I32_NVAL_1, F32),
    ("multiple_action_scalar_reward", I32_NVAL_5, F32),
    ("tree_action_scalar_reward", TREE_DAS, F32),
    ("single_action_matrix_reward", I32_NVAL_1, F32_1x3),
    ("multiple_action_matrix_reward", I32_NVAL_5, F32_1x3),
    ("tree_action_matrix_reward", TREE_DAS, F32_1x3),
    ("single_action_tree_reward", I32_NVAL_1, TREE_AS),
    ("multiple_action_tree_reward", I32_NVAL_5, TREE_AS),
    ("tree_action_tree_reward", TREE_DAS, TREE_AS),
)


class EnvironmentLoopTest(parameterized.TestCase):
    @parameterized.named_parameters(*TEST_CASES)
    def test_run_steps(self, action_spec, observation_spec):
        env, agent, loop = _parameterized_setup(action_spec, observation_spec)

        # Run the loop. This will run a total number of steps of EPISODE_LENGTH.
        loop.run(n_step=EPISODE_LENGTH)
        self.assertEqual(agent.num_updates, EPISODE_LENGTH)


def _parameterized_setup(
    action_spec: types.NestedArraySpec, observation_spec: types.NestedArraySpec
):
    """Common setup code that, unlike self.setUp, takes arguments.

    Args:
      action_spec: a (nested) types.ArraySpec.
      observation_spec: a (nested) types.ArraySpec.
    Returns:
      environment, agent, loop
    """
    env = fakes.FakeEnvironment(action_spec, observation_spec, 0)
    agent = fakes.RandomAgent(action_spec, observation_spec, 0)
    loop = environment_loop.EnvironmentLoop(env, agent)
    return env, agent, loop


if __name__ == "__main__":
    absltest.main()
