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

"""Fake (mock) components.

Minimal implementations of fake Agent and Environment which can be instantiated in
order to test or interact with other components.
"""

import threading
from typing import List, Mapping, Optional, Sequence, Callable, Iterator

from rlba import agent
from rlba.environment import Environment
from rlba import types
# from rlba import wrappers
import numpy as np
import tree


class RandomAgent:
  """Fake agent which generates random actions and validates specs."""

  def __init__(
    self,
    action_spec: types.ActionSpec,
    observation_spec: types.ObservationSpec,
    seed: int,
  ):
    self._action_spec: types.ActionSpec = action_spec
    self._observation_spec: types.ObservationSpec = observation_spec

    self.num_updates = 0

  def select_action(self) -> types.NestedArray:
    action = generate_from_spec(self._action_spec)
    _validate_spec(self._action_spec, action)

    return action

  def observe(
    self,
    action: types.ActionSpec,
    obs: types.ObservationSpec,
    ):
    _validate_spec(self._action_spec, action)
    _validate_spec(self._observation_spec, obs)
    return 0.0

  def update(self, wait: bool = False):
    self.num_updates += 1


class VariableSource:
  """Fake variable source."""

  def __init__(self,
               variables: Optional[types.NestedArray] = None,
               barrier: Optional[threading.Barrier] = None,
               use_default_key: bool = True):
    # Add dummy variables so we can expose them in get_variables.
    if use_default_key:
      self._variables = {'policy': [] if variables is None else variables}
    else:
      self._variables = variables
    self._barrier = barrier

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    if self._barrier is not None:
      self._barrier.wait()
    return [self._variables[name] for name in names]


class Learner(agent.Learner, VariableSource):
  """Fake Learner."""

  def __init__(self,
               variables: Optional[types.NestedArray] = None,
               barrier: Optional[threading.Barrier] = None):
    super().__init__(variables=variables, barrier=barrier)
    self.step_counter = 0

  def step(self):
    self.step_counter += 1


class FakeEnvironment:
  """A fake environment with a given spec."""

  def __init__(
      self,
      action_spec: types.ActionSpec,
      observation_spec: types.ObservationSpec,
      seed: int,
  ):

    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._step = 0

  def _generate_fake_observation(self):
    return generate_from_spec(self._observation_spec)

  def step(self, action) -> types.NestedArray:
    _validate_spec(self._action_spec, action)

    return self._generate_fake_observation()

  @property
  def action_spec(self):
    return self._action_spec

  @property
  def observation_spec(self):
    return self._observation_spec


def make_discrete_env(
               *,
               num_actions: int = 1,
               num_observations: int = 1,
               action_dtype=np.int32,
               obs_dtype=np.int32,
               obs_shape: Sequence[int] = (),
               **kwargs):
  """Discrete state and action fake environment."""
  observation_spec = types.BoundedArraySpec(
      shape=obs_shape,
      dtype=obs_dtype,
      minimum=obs_dtype(0),
      maximum=obs_dtype(num_observations - 1))
  action_spec = types.DiscreteArraySpec(num_actions, dtype=action_dtype)
  return FakeEnvironment(
    action_spec=action_spec,
    observation_spec=observation_spec
  )


def _validate_spec(spec: types.NestedArraySpec, value: types.NestedArray):
  """Validate a value from a potentially nested spec."""
  tree.assert_same_structure(value, spec)
  tree.map_structure(lambda s, v: s.validate(v), spec, value)


def _normalize_array(array: types.ArraySpec) -> types.ArraySpec:
  """Converts bounded arrays with (-inf,+inf) bounds to unbounded arrays.

  The returned array should be mostly equivalent to the input, except that
  `generate_value()` returns -infs on arrays bounded to (-inf,+inf) and zeros
  on unbounded arrays.

  Args:
    array: the array to be normalized.

  Returns:
    normalized array.
  """
  if isinstance(array, types.DiscreteArraySpec):
    return array
  if not isinstance(array, types.BoundedArraySpec):
    return array
  if not (array.minimum == float('-inf')).all():
    return array
  if not (array.maximum == float('+inf')).all():
    return array
  return types.ArraySpec(array.shape, array.dtype, array.name)


def generate_from_spec(spec: types.NestedArraySpec) -> types.NestedArray:
  """Generate a value from a potentially nested spec."""
  return tree.map_structure(lambda s: _normalize_array(s).generate_value(),
                            spec)


def fake_atari_wrapped(episode_length: int = 10,
                       oar_wrapper: bool = False) -> Environment:
  """Builds fake version of the environment to be used by tests.

  Args:
    episode_length: The length of episodes produced by this environment.
    oar_wrapper: Should ObservationActionRewardWrapper be applied.

  Returns:
    Fake version of the environment equivalent to the one returned by
    env_loader.load_atari_wrapped
  """
  env = make_discrete_env(
      num_actions=18,
      num_observations=2,
      obs_shape=(84, 84, 4),
      obs_dtype=np.float32)

  if oar_wrapper:
    raise ValueError
    # env = wrappers.ObservationActionRewardWrapper(env)
  return env