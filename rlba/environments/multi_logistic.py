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

"""A logistic bandit environment with multiple independent heads.
"""


from typing import Iterator

import numpy as np
from numpy.random import default_rng
from rlba.types import Array, ArraySpec, BoundedArraySpec, DiscreteArraySpec, NestedArraySpec, NestedDiscreteArraySpec


class MultipleLogisticEnv(object):
  """Environment representing a logistic model with multiple concurrent binary outputs."""

  def __init__(
      self,
      action_feature_dim: int,
      embedding_dim: int,
      output_dim: int,
      num_action: int,
      sigma_p: float = 1.0,
      seed: int = 0) -> None:
    self._action_feature_dim = action_feature_dim
    self._embedding_dim = embedding_dim
    self._output_dim = output_dim
    self._num_action = num_action
    self._sigma_p = sigma_p
    self._theta = None
    self._output_features = None 
    self.action_features = None 

    self._action_spec = DiscreteArraySpec(num_action, name='action spec')
    self._observation_spec: ArraySpec = BoundedArraySpec(
        shape=(output_dim,), dtype=bool, minimum=0.0, maximum=1.0,
        name='observation spec')

    self._reset(seed)


  def step(self, action: int) -> Array:
    """Step the environment according to the action and returns an `observation`.

    Args:
      action: an integer corresponding to the arm index.

    Returns:
      An `Observation` A NumPy array of bools. Must conform to the
          specification returned by `observation_spec()`.
    """
    try:
        action = int(action)
    except TypeError:
        TypeError('Action does not seem to be convertable to an int')
    if action >= self._action_spec.num_values:
        raise ValueError('action is larger than number of available arms.')

    probs = self._compute_output_probs(action)
    return (self._rng.random(self._output_dim) <= probs).astype(np.float32)

  def _reset(self, seed: int):
    """"Create a new instance of the environment and initialize the features."""
    self._rng = np.random.default_rng(seed)
    self._theta = self._sigma_p * self._rng.normal(size=(self._action_feature_dim - 1, self._embedding_dim))
    self._theta = np.r_[self._theta, -np.ones((1, self._embedding_dim))]
    self._output_features = np.abs(self._rng.normal(size=(self._embedding_dim, self._output_dim)))
    self._action_features = self._rng.normal(size=(self._action_feature_dim - 1, self._num_action))
    self._action_features /= np.linalg.norm(self._action_features, axis=0, keepdims=True)
    self._action_features = np.r_[self._action_features, np.ones((1, self._num_action))]

  def _get_action_feature(self, action):
    """Validate the action and return corresponding feature vector."""
    assert self._action_features is not None, 'Please first reset the environment.'
    action = np.array(action)
    action_feature = self._action_features[:, action]
    return action_feature
 
  def _compute_output_probs(self, action):
    action_feature = self._get_action_feature(action)
    embedding = action_feature.T @ self._theta
    logits = embedding @ self._output_features
    exp_logits = np.exp(-logits)
    probs = 1 / (1 + exp_logits)
    return probs

  @property
  def observation_spec(self) -> NestedArraySpec:
    """Defines the observations provided by the environment.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return self._observation_spec

  @property
  def action_spec(self) -> NestedDiscreteArraySpec:
    """Defines the actions that should be provided to `step`.

    Returns:
      A `DiscereteArray` spec, or a nested dict, list or tuple of `DiscreteArray` specs.
    """
    return self._action_spec

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