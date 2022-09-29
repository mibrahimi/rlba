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

"""Core interfaces.

This file specifies and documents the notions of `Agent` and `Learner`.
"""

import itertools
from typing import List, Optional, Sequence
from typing_extensions import Protocol

from rlba import types


class Agent(Protocol):
    """Interface for an agent that can act.

    This interface defines an API for an Agent to interact with an EnvironmentLoop
    , e.g. a simple RL loop where each step is of the
    form:

      # Take two step and observe.
      action = agent.select_action(timestep.observation)
      obs = env.step()
      actor.observe(obs)
    """

    def select_action(self) -> int:
        """Samples from the policy and returns an action."""

    def reward_spec(self) -> types.ArraySpec:
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
        action: int,
        obs: types.NestedArray,
    ):
        """Make a first observation from the environment.

        Note that this need not be an initial state, it is merely beginning the
        recording of a trajectory.

        Args:
          action: action taken in the environment.
          obs: observation produced by the environment given the action.
        """


class Learner(Protocol):
    """Abstract learner object.

    This corresponds to an object which implements a learning loop. A single step
    of learning should be implemented via the `step` method and this step
    is generally interacted with via the `run` method which runs update
    continuously.
    """

    def step(self) -> None:
        """Perform an update step of the learner's parameters."""

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""

        iterator = range(num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.step()

    """Abstract source of variables.

  Objects which implement this interface provide a source of variables, returned
  as a collection of (nested) numpy arrays. Generally this will be used to
  provide variables to some learned policy/etc.
  """

    def get_variables(self, names: Sequence[str]) -> types.NestedArray:
        """Return the named variables as a collection of (nested) numpy arrays.

        Provides a source of variables. Generally this will be used to
        provide variables to some learned policy/etc.

        Args:
          names: args where each name is a string identifying a predefined subset of
            the variables.

        Returns:
          A list of (nested) numpy arrays `variables` such that `variables[i]`
          corresponds to the collection named by `names[i]`.
        """

    def save(self) -> types.NestedArray:
        """Returns the state from the object to be saved."""

    def restore(self, state: types.NestedArray) -> None:
        """Given the state, restores the object."""
