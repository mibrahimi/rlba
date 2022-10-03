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

"""An assortment recommendation environment.
"""


import math
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


def _choose_optimal_assortment(theta, n_slot, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_item = len(theta)
    theta_sorted_idx = np.argsort(-theta)
    theta_thr = theta[theta_sorted_idx[n_slot - 1]]
    theta_thr_idx = np.nonzero(theta == theta_thr)[0]
    theta_opt_idx = np.setdiff1d(theta_sorted_idx[:n_slot], theta_thr_idx, True)
    theta_opt_idx = np.concatenate(
        (
            theta_opt_idx,
            rng.choice(theta_thr_idx, size=n_slot - len(theta_opt_idx), replace=False),
        )
    )
    selected_items = np.zeros(n_item)
    selected_items[theta_opt_idx] = 1
    return selected_items


class AssortmentRecommendationEnv:
    """Environment representing a multi-item recommendation environment."""

    def __init__(self, n_item: int, n_slot, seed: int, sigma_p=1.0):
        self._n_item = n_item  # Total number of items that can be recommended
        self._n_slot = (
            n_slot  # Number of available sltos. #selected items should be <= n_slot
        )
        self._sigma_p = sigma_p
        self._reset(seed)

        n_action = int(
            math.factorial(n_item)
            / math.factorial(n_slot)
            / math.factorial(n_item - n_slot)
        )
        self._action_spec = DiscreteArraySpec(n_action, name="action spec")
        self._observation_spec: ArraySpec = BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0.0,
            maximum=n_item,
            name="observation spec",
        )

    def _reset(self, seed):
        # reset the environment - will generate a new environment
        self._rng = np.random.default_rng(seed)
        self._theta = np.minimum(
            self._rng.normal(size=self._n_item) * self._sigma_p - 1.0 * self._sigma_p,
            0.0,
        )
        self._optimal_expected_reward = 1.0 - self._compute_opt_selection_probs()[0]

    def _validate_action(self, action):
        action = np.array(action)
        assert len(action) == self._n_item, f"action has wrong dimension {len(action)}."
        action = (action > 0).astype(np.float32)  # Convert action to 0/1
        assert (
            action.sum() <= self._n_slot
        ), f"action has wrong number of non zero elements: {action.sum()}."
        return action

    def step(self, action: NestedArray) -> Array:
        """Step the environment according to the action and returns an `observation`.

        Args:
          action: an integer corresponding to the arm index.

        Returns:
          An `Observation`: an integer 0 <= i <= n_item representing the item
          selected with 0 meaning no item selected."""

        action = self._validate_action(action)
        probs = self._compute_selection_probs(action)
        return self._rng.choice(self._n_item + 1, p=probs)

    def _compute_selection_probs(self, action):
        logits = self._theta * action
        logits[action <= 0.0] = -np.inf  # Set the logit for non selected item to -Inf
        logits = np.insert(
            logits, 0, 0.0
        )  # Add logits for the no-item-selected observation
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    def _compute_opt_selection_probs(self):
        action = _choose_optimal_assortment(self._theta, self._n_slot, self._rng)
        return self._compute_selection_probs(action)

    def expected_reward(self, action):
        action = self._validate_action(action)
        return 1.0 - self._compute_selection_probs(action)[0]

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
