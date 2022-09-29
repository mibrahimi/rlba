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

from typing import Callable, Mapping, List

from rlba.environment import Environment
from rlba.types import Array, NestedArray
from rlba.utils.observers import base
import numbers
import numpy as np


class RegretObserver:
    """Observer the expected and cumulative regret in a bandit environment.

    This observer assumes the expected reward for each step can be computed
    from the environment and depends only on the selected action.

    """

    def __init__(self, exp_reward_fn: Callable[[Environment], Array]):
        """
        Args:
            exp_reward_fn: a callable that given the environment returns an
            Array representing the expected reward for taking each action.
        """
        self._exp_reward_fn = exp_reward_fn
        self._cumulative_regret = 0.0
        self._observer_step = 0
        self._metrics = {}

    def observe(
        self, env: Environment, action: NestedArray, observation: NestedArray
    ) -> None:
        """Records one environment step."""
        assert isinstance(
            action, numbers.Number
        ), "RegretObserver only works with scalar actions"
        assert action == int(action), "RegretObserver only works with integer actions"
        action = int(action)
        exp_rewards = self._exp_reward_fn(env)
        opt_reward = exp_rewards.max()
        exp_reward = exp_rewards[action]
        regret = opt_reward - exp_reward
        self._cumulative_regret += regret
        self._observer_step += 1
        self._metrics = {
            "observer_step": self._observer_step,
            "action": action,
            "exp_reward": exp_reward,
            "regret": regret,
            "cumulative_regret": self._cumulative_regret,
        }

    def get_metrics(self) -> Mapping[str, List[base.Number]]:
        """Returns metrics collected for the last step."""
        return self._metrics
