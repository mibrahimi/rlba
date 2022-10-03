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

"""An observer that tracks regret in a bandit setting."""

from typing import Any, Callable, Iterable, List, Mapping, Optional
import numpy as np

from rlba.environment import Environment
from rlba.types import Array, NestedArray
from rlba.utils.observers import base
import numbers


class RegretObserver:
    """Observer the expected and cumulative regret in a bandit environment.

    This observer assumes the expected reward for each step can be computed
    from the environment and depends only on the selected action.

    """

    def __init__(
        self,
        expected_reward_fn: Optional[
            Callable[[Environment, NestedArray], float]
        ] = None,
        opt_expected_reward_fn: Optional[Callable[[Environment], float]] = None,
        cache_expected_reward: bool = True,
        action_key_fn: Callable[[NestedArray], Any] = lambda a: a,
    ):
        """Observe instantanous and cumulative expected regret.
        Args:
            expected_reward_fn: a function that given (env, action) returns the
                expected reward. If None we assume env provides a method
                `expected_reward` with the same signature.
            opt_expected_reward_fn: a function that given (env) returns the
                optimal expected reward. If None we assume env provides a method
                `opt_expected_reward` with the same signature.
            cache_expected_reward: if True assume expected rewards are static and cache
                the computed values.
            action_key_fn: a function to transform action into an action key for
                to be used for caching the results. This is useful when action is not
                hashable, e.g., a numpy array.

        """
        self._expected_reward_fn = expected_reward_fn
        self._opt_expected_reward_fn = opt_expected_reward_fn
        self._cache_expected_reward = cache_expected_reward
        self._action_key_fn = action_key_fn

        self._cumulative_regret = 0.0
        self._observer_step = 0
        self._metrics = {}

    def observe(
        self,
        env: Environment,
        action: NestedArray,
        observation: NestedArray,
        reward: float,
    ) -> None:
        """Records one environment step."""
        if self._observer_step == 0:
            if self._expected_reward_fn is None:
                self._expected_reward_fn = lambda e, a: e.expected_reward(a)
            if self._opt_expected_reward_fn is None:
                self._opt_expected_reward_fn = lambda e: e.optimal_expected_reward()
            if self._cache_expected_reward:
                self._exp_rewards = {}
                self._opt_exp_reward = self._opt_expected_reward_fn(env)

        if self._cache_expected_reward:
            action_key = self._action_key_fn(action)
            if action_key in self._exp_rewards:
                exp_reward = self._exp_rewards[action_key]
            else:
                exp_reward = self._expected_reward_fn(env, action)
                self._exp_rewards[action_key] = exp_reward
            opt_exp_reward = self._opt_exp_reward
        else:
            opt_exp_reward = self._opt_expected_reward_fn(env)
            exp_reward = self._expected_reward_fn(env, action)

        regret = opt_exp_reward - exp_reward
        self._cumulative_regret += regret
        self._observer_step += 1
        self._metrics = {
            "observer_step": self._observer_step,
            "action": action,
            "reward": reward,
            "exp_reward": exp_reward,
            "regret": regret,
            "cumulative_regret": self._cumulative_regret,
        }

    def get_metrics(self) -> Mapping[str, List[base.Number]]:
        """Returns metrics collected for the last step."""
        return self._metrics
