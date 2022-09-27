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

"""Common types used throughout RLBA."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union
from dm_env import specs
from numpy.typing import ArrayLike


ArraySpec = specs.Array
BoundedArraySpec = specs.BoundedArray
DiscreteArraySpec = specs.DiscreteArray

# pytype: disable=not-supported-yet
NestedArraySpec = Union[
    specs.Array,
    Iterable["NestedArraySpec"],
    Mapping[Any, "NestedArraySpec"],
]

NestedDiscreteArraySpec = Union[
    specs.DiscreteArray,
    Iterable["NestedDiscreteArraySpec"],
    Mapping[Any, "NestedDiscreteArraySpec"],
]
# pytype: enable=not-supported-yet


ActionSpec = NestedDiscreteArraySpec
ObservationSpec = NestedArraySpec


@dataclass
class EnvironmentSpec:
    action_spec: ActionSpec
    observation_spec: ObservationSpec


Array = ArrayLike
# pytype: disable=not-supported-yet
NestedArray = Union[
    ArrayLike,
    Iterable["NestedArray"],
    Mapping[Any, "NestedArray"],
]
# pytype: enable=not-supported-yet
