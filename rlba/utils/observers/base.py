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

"""Metrics observers."""

from typing import Dict, Union
from typing_extensions import Protocol


from rlba.environment import Environment
from rlba.types import NestedArray


Number = Union[int, float]


class EnvLoopObserver(Protocol):
  """An interface for collecting metrics/counters in EnvironmentLoop."""

  def observe(self, env: Environment, action: NestedArray,
              observation: NestedArray, reward: float) -> None:
    """Records one environment step."""

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
