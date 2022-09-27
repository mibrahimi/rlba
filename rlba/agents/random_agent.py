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

""" A random agent that acts randomly.
"""

from dataclasses import dataclass

from rlba import types
from rlba.utils import metrics
from numpy.random import default_rng


@metrics.record_class_usage
@dataclass
class RandomAgent:
    """A random agent that select actions randomly"""

    def __init__(
        self,
        action_spec: types.ActionSpec,
        observation_spec: types.ObservationSpec,
        seed: int,
    ):
        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self._rng = default_rng(seed)

    def select_action(self) -> types.NestedArray:
        """Samples from the policy and returns an action."""
        return self._rng.choice(self._action_spec.num_values)

    def reward_spec(self) -> types.Array:
        """Describes the reward returned by the environment.

        By default this is assumed to be a single float.

        Returns:
          An `Array` spec.
        """
        return types.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> types.BoundedArraySpec:
        """Describes the discount considered by the agent for planning.

        By default this is assumed to be a single float between 0 and 1.

        Returns:
          An `Array` spec.
        """
        return types.BoundedArraySpec(
            shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )

    def observe(
        self,
        action: types.NestedArray,
        obs: types.NestedArray,
    ):
        """Make an observation from the environment.

        Args:
          action: action taken in the environment.
          obs: observation produced by the environment given the action.
        """
        return 0

    def update(self, wait: bool = False):
        """Perform an update of the actor parameters from past observations.

        Args:
          wait: if True, the update will be blocking.
        """
        pass
