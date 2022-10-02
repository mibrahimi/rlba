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


class RegretObserver:
    """Observer the expected and cumulative regret in a bandit environment.

    This observer assumes the expected reward for each step can be computed
    from the environment and depends only on the selected action.

    """

    def __init__(self):
        """Observe instantanous and cumulative expected regret.

        This observer expects the environment to have exposed two methods: (i)
        env.expected_reward(action) and (ii) env.optimal_expected_reward() which
        returns the current expected and optimal expected rewards respectively.
        """
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
        exp_reward = env.expected_reward(action)
        opt_reward = env.optimal_expected_reward()
        regret = opt_reward - exp_reward
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
