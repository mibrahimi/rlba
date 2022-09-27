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

"""Tests for measurement_metrics."""


from rlba.environment import Environment
from rlba.testing import fakes
from rlba.types import ArraySpec, DiscreteArraySpec
from rlba.utils.observers import measurement_metrics
import numpy as np

from absl.testing import absltest


def _make_fake_env() -> Environment:
    action_spec = DiscreteArraySpec(10, dtype=np.int32)
    observation_spec = ArraySpec(shape=(10, 5), dtype=np.float32)
    return fakes.FakeEnvironment(action_spec, observation_spec, 0)


_FAKE_ENV = _make_fake_env()
_ACTION = fakes.generate_from_spec(_FAKE_ENV.action_spec)
_OBSERVATION = fakes.generate_from_spec(_FAKE_ENV.observation_spec)


class MeasurementMetricsTest(absltest.TestCase):
    def test_observe_nothing(self):
        observer = measurement_metrics.MeasurementObserver()
        self.assertEqual({}, observer.get_metrics())

    def test_observe_single_step(self):
        observer = measurement_metrics.MeasurementObserver()
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        self.assertEqual(
            {
                "measurement[0]_max": 1.0,
                "measurement[0]_mean": 1.0,
                "measurement[0]_p25": 1.0,
                "measurement[0]_p50": 1.0,
                "measurement[0]_p75": 1.0,
                "measurement[1]_max": -2.0,
                "measurement[1]_mean": -2.0,
                "measurement[1]_p25": -2.0,
                "measurement[1]_p50": -2.0,
                "measurement[1]_p75": -2.0,
                "measurement[0]_min": 1.0,
                "measurement[1]_min": -2.0,
            },
            observer.get_metrics(),
        )

    def test_observe_multiple_step_same_observation(self):
        observer = measurement_metrics.MeasurementObserver()
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        self.assertEqual(
            {
                "measurement[0]_max": 1.0,
                "measurement[0]_mean": 1.0,
                "measurement[0]_p25": 1.0,
                "measurement[0]_p50": 1.0,
                "measurement[0]_p75": 1.0,
                "measurement[1]_max": -2.0,
                "measurement[1]_mean": -2.0,
                "measurement[1]_p25": -2.0,
                "measurement[1]_p50": -2.0,
                "measurement[1]_p75": -2.0,
                "measurement[0]_min": 1.0,
                "measurement[1]_min": -2.0,
            },
            observer.get_metrics(),
        )

    def test_observe_empty_observation(self):
        observer = measurement_metrics.MeasurementObserver()
        observation = {}
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=observation)
        self.assertEqual({}, observer.get_metrics())

    def test_observe_single_dimensions(self):
        observer = measurement_metrics.MeasurementObserver()
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        observation = [1000.0, -50.0]

        observer.observe(
            env=_FAKE_ENV, action=np.array([[1, 2], [3, 4]]), observation=observation
        )

        np.testing.assert_equal(
            {
                "measurement[0]_max": 1000.0,
                "measurement[0]_min": 1000.0,
                "measurement[0]_mean": 1000.0,
                "measurement[0]_p25": 1000.0,
                "measurement[0]_p50": 1000.0,
                "measurement[0]_p75": 1000.0,
                "measurement[1]_max": -50.0,
                "measurement[1]_mean": -50.0,
                "measurement[1]_p25": -50.0,
                "measurement[1]_p50": -50.0,
                "measurement[1]_p75": -50.0,
                "measurement[1]_min": -50.0,
            },
            observer.get_metrics(),
        )


if __name__ == "__main__":
    absltest.main()
