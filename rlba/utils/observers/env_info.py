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

"""An observer that returns env's info.
"""
from typing import Dict

from rlba.environment import Environment
from rlba.types import NestedArray
from rlba.utils.observers import base
import numpy as np


class EnvInfoObserver:
    """An observer that collects and accumulates scalars from env's info."""

    def __init__(self):
        self._metrics = {}

    def _accumulate_metrics(self, env: Environment) -> None:
        if not hasattr(env, "get_info"):
            return
        info = getattr(env, "get_info")()
        if not info:
            return
        for k, v in info.items():
            if np.isscalar(v):
                self._metrics[k] = self._metrics.get(k, 0) + v

    def observe(
        self, env: Environment, action: NestedArray, observation: NestedArray
    ) -> None:
        """Records one environment step."""
        self._accumulate_metrics(env)

    def get_metrics(self) -> Dict[str, base.Number]:
        """Returns metrics collected for the current episode."""
        return self._metrics
