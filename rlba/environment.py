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

"""Environment interface and related utility functions.

This module exposes the environment interface as well as providing an
additional `EnvironmentSpec` class which collects all of the specs for a given
environment. An `EnvironmentSpec` instance can be created directly or by using
the `make_environment_spec` helper given a `dm_env.Environment` instance.
"""

from typing import Any, Iterable, Mapping, Union
from typing_extensions import Protocol

from dataclasses import dataclass
from rlba.types import NestedArray, NestedArraySpec, NestedDiscreteArraySpec


class Environment(Protocol):
  """Interface for RL environments.

  Observations and valid actions are described with `Array` specs, defined in
  the `dm_env` repository.
  """

  def step(self, action: NestedArray) -> NestedArray:
    """Updates the environment according to the action and returns an `observation`.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called.

    Args:
      action: A DiscreteArray corresponding to `action_spec()`.

    Returns:
      An `Observation` A NumPy array, or a nested dict, list or tuple of arrays.
          Scalar values that can be cast to NumPy arrays (e.g. Python floats)
          are also valid in place of a scalar array. Must conform to the
          specification returned by `observation_spec()`.
    """

  def observation_spec(self):
    """Defines the observations provided by the environment.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """

  def action_spec(self):
    """Defines the actions that should be provided to `step`.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      A `DiscereteArray` spec, or a nested dict, list or tuple of `Array` specs.
    """

  def close(self):
    """Frees any resources used by the environment.

    Implement this method for an environment backed by an external process.

    This method can be used directly

    ```python
    env = Env(...)
    # Use env.
    env.close()
    ```

    or via a context manager

    ```python
    with Env(...) as env:
      # Use env.
    ```
    """
    pass

  def __enter__(self):
    """Allows the environment to be used in a with-statement context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Allows the environment to be used in a with-statement context."""
    del exc_type, exc_value, traceback  # Unused.
    self.close()


@dataclass
class EnvironmentSpec:
  """Full specification of the domains used by a given environment."""
  observation_spec: NestedArraySpec
  action_spec: NestedDiscreteArraySpec


def make_environment_spec(environment: Environment) -> EnvironmentSpec:
  """Returns an `EnvironmentSpec` describing values used by an environment."""
  return EnvironmentSpec(
      observation_spec=environment.observation_spec(),
      action_spec=environment.action_spec())
