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


from typing import Tuple

import numpy as np
from numpy.random import default_rng
from rlba.types import (
    Array,
    ArraySpec,
    BoundedArraySpec,
    DiscreteArraySpec,
    NestedArray,
    NestedArraySpec,
    NestedDiscreteArraySpec,
)


class MultipleLogisticEnv(object):
    """Environment representing a logistic model with multiple concurrent binary outputs.

    The first element of the observat
    """

    def __init__(
        self,
        action_feature_dim: int,
        input_dim: int,
        embedding_dim: int,
        output_dim: int,
        sigma_p: float = 1.0,
        seed: int = 0,
    ) -> None:
        self._action_feature_dim = action_feature_dim
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._output_dim = output_dim
        self._sigma_p = sigma_p
        self._theta = None
        self._output_features = None
        self.action_features = None

        self._action_spec = DiscreteArraySpec(input_dim, name="action spec")
        self._observation_spec: ArraySpec = BoundedArraySpec(
            shape=(output_dim,),
            dtype=bool,
            minimum=0.0,
            maximum=1.0,
            name="observation spec",
        )

        self._reset(seed)

    def _reset(self, seed: int):
        """ "Create a new instance of the environment and initialize the features."""
        self._rng = np.random.default_rng(seed)
        self._theta = self._sigma_p * self._rng.normal(
            size=(self._action_feature_dim - 1, self._embedding_dim)
        )
        self._theta = np.r_[self._theta, -np.ones((1, self._embedding_dim))]
        self._output_features = np.abs(
            self._rng.normal(size=(self._embedding_dim, self._output_dim))
        )
        self._input_features = self._rng.normal(
            size=(self._action_feature_dim - 1, self._input_dim)
        )
        self._input_features /= np.linalg.norm(
            self._input_features, axis=0, keepdims=True
        )
        self._input_features = np.r_[
            self._input_features, np.ones((1, self._input_dim))
        ]
        self._output_probs = np.array(
            [
                self._compute_output_probs(iidx, np.arange(self._output_dim))
                for iidx in range(self._input_dim)
            ],
        ).T
        self._optimal_expected_reward = self._output_probs[0, :].max()

    def _get_input_feature(self, input_idx):
        """Returns corresponding input feature vector for a given action."""
        return self._input_features[:, input_idx]

    def _get_output_feature(self, output_idxs):
        """Returns corresponding input feature vector for a given action."""
        output_idxs = np.array(output_idxs)
        return self._output_features[:, output_idxs]

    def _compute_output_probs(self, input_idx: int, output_idxs: Array):
        input_feature = self._get_input_feature(input_idx)
        output_features = self._get_output_feature(output_idxs)
        embedding = input_feature.T @ self._theta
        logits = embedding @ output_features
        exp_logits = np.exp(-logits)
        probs = 1 / (1 + exp_logits)
        return probs

    def _validate_action(self, action: NestedArray) -> Tuple[int, Array]:
        try:  # action only specifies the input (curriculum in AI tutor example)
            input_idx = int(action)
            output_idxs = np.arange(self._output_dim)  # Assume all outputs are returned
        except TypeError:
            input_idx, output_idxs = action
            input_idx = int(input_idx)
            output_idxs = np.insert(np.array(output_idxs), 0, 0)
        except TypeError:
            TypeError("Action does not seem to be convertable to an int")
        if input_idx >= self._input_dim:
            raise ValueError("input index is larger than number of available arms.")
        if output_idxs.max() >= self._output_dim:
            raise ValueError("output index is larger than number of available arms.")

        return input_idx, output_idxs

    def step(self, action: NestedArray) -> Array:
        """Step the environment according to the action and returns an `observation`.

        Args:
          action: an integer corresponding to the arm index.

        Returns:
          An `Observation` A NumPy array of bools. Must conform to the
              specification returned by `observation_spec()`.
        """

        input_idx, output_idxs = self._validate_action(action)
        probs = self._compute_output_probs(input_idx, output_idxs)
        return (self._rng.random(len(output_idxs)) <= probs).astype(np.float32)

    def expected_reward(self, action):
        return self._output_probs[0, action]

    def optimal_expected_reward(self):
        return self._optimal_expected_reward

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
