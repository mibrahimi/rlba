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

"""Tests for action_metrics_observers."""


from rlba.environment import Environment
from rlba.testing import fakes
from rlba.types import ArraySpec, DiscreteArraySpec
from rlba.utils.observers import action_metrics
import numpy as np

from absl.testing import absltest


def _make_fake_env() -> Environment:
    action_spec = DiscreteArraySpec(10, dtype=np.int32)
    observation_spec = ArraySpec(shape=(10, 5), dtype=np.float32)
    return fakes.FakeEnvironment(action_spec, observation_spec, 0)


_FAKE_ENV = _make_fake_env()
_ACTION = fakes.generate_from_spec(_FAKE_ENV.action_spec)
_OBSERVATION = fakes.generate_from_spec(_FAKE_ENV.observation_spec)


class ActionMetricsTest(absltest.TestCase):
    def test_observe_nothing(self):
        observer = action_metrics.ActionStatsObserver()
        self.assertEqual({}, observer.get_metrics())

    def test_observe_single_step(self):
        observer = action_metrics.ActionStatsObserver()
        action = np.array([1])
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        self.assertEqual(
            {
                "action[0]_max": 1,
                "action[0]_min": 1,
                "action[0]_mean": 1,
                "action[0]_p50": 1,
            },
            observer.get_metrics(),
        )

    def test_observe_multiple_step(self):
        observer = action_metrics.ActionStatsObserver()
        action = np.array([1])
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        action = np.array([4])
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        action = np.array([5])
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        self.assertEqual(
            {
                "action[0]_max": 5,
                "action[0]_min": 1,
                "action[0]_mean": 10 / 3,
                "action[0]_p50": 4,
            },
            observer.get_metrics(),
        )

    def test_observe_zero_dimensions(self):
        observer = action_metrics.ActionStatsObserver()
        action = np.array([1])
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        self.assertEqual(
            {
                "action[]_max": 1,
                "action[]_min": 1,
                "action[]_mean": 1,
                "action[]_p50": 1,
            },
            observer.get_metrics(),
        )

    def test_observe_multiple_dimensions(self):
        observer = action_metrics.ActionStatsObserver()
        action = np.array([[1, 2], [3, 4]])
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        np.testing.assert_equal(
            {
                "action[0, 0]_max": 1,
                "action[0, 0]_min": 1,
                "action[0, 0]_mean": 1,
                "action[0, 0]_p50": 1,
                "action[0, 1]_max": 2,
                "action[0, 1]_min": 2,
                "action[0, 1]_mean": 2,
                "action[0, 1]_p50": 2,
                "action[1, 0]_max": 3,
                "action[1, 0]_min": 3,
                "action[1, 0]_mean": 3,
                "action[1, 0]_p50": 3,
                "action[1, 1]_max": 4,
                "action[1, 1]_min": 4,
                "action[1, 1]_mean": 4,
                "action[1, 1]_p50": 4,
            },
            observer.get_metrics(),
        )


if __name__ == "__main__":
    absltest.main()
