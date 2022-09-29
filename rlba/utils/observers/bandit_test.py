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
from rlba.utils.observers import bandit
import numpy as np

from absl.testing import absltest, parameterized


def _make_fake_env(num_action: int) -> Environment:
    action_spec = DiscreteArraySpec(num_action, dtype=np.int32)
    observation_spec = ArraySpec(shape=(10, 5), dtype=np.float32)
    return fakes.FakeEnvironment(action_spec, observation_spec, 0)


_NUM_ACTION = 10
_FAKE_ENV = _make_fake_env(_NUM_ACTION)
_ACTION = fakes.generate_from_spec(_FAKE_ENV.action_spec)
_OBSERVATION = fakes.generate_from_spec(_FAKE_ENV.observation_spec)
_ZEROS_REWARD_FN = lambda e: np.zeros(e.action_spec.num_values)
_ONES_REWARD_FN = lambda e: np.ones(e.action_spec.num_values)
_RANGE_REWARD_FN = lambda e: np.arange(e.action_spec.num_values)


class BanditRegretTest(absltest.TestCase):
    def test_observe_nothing(self):
        observer = bandit.RegretObserver(_ZEROS_REWARD_FN)
        self.assertEqual({}, observer.get_metrics())

    def test_observe_reward_zero(self):
        observer = bandit.RegretObserver(_ZEROS_REWARD_FN)
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        self.assertEqual(
            {
                "observer_step": 1,
                "action": _ACTION,
                "exp_reward": 0.0,
                "regret": 0.0,
                "cumulative_regret": 0.0,
            },
            observer.get_metrics(),
        )

    def test_observe_reward_one(self):
        observer = bandit.RegretObserver(_ONES_REWARD_FN)
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        self.assertEqual(
            {
                "observer_step": 1,
                "action": _ACTION,
                "exp_reward": 1.0,
                "regret": 0.0,
                "cumulative_regret": 0.0,
            },
            observer.get_metrics(),
        )

    def test_observe_reward_range(self):
        observer = bandit.RegretObserver(_RANGE_REWARD_FN)
        observer.observe(env=_FAKE_ENV, action=_ACTION, observation=_OBSERVATION)
        self.assertEqual(
            {
                "observer_step": 1,
                "action": _ACTION,
                "exp_reward": 0.0,
                "regret": _NUM_ACTION - 1.0,
                "cumulative_regret": _NUM_ACTION - 1.0,
            },
            observer.get_metrics(),
        )

    def test_observe_multiple_step(self):
        observer = bandit.RegretObserver(_RANGE_REWARD_FN)
        action = 1
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        cumulative_regret = _NUM_ACTION - action - 1.0
        self.assertEqual(
            {
                "observer_step": 1,
                "action": action,
                "exp_reward": action,
                "regret": _NUM_ACTION - action - 1.0,
                "cumulative_regret": cumulative_regret,
            },
            observer.get_metrics(),
        )
        action = 3
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        cumulative_regret += _NUM_ACTION - action - 1.0
        self.assertEqual(
            {
                "observer_step": 2,
                "action": action,
                "exp_reward": action,
                "regret": _NUM_ACTION - action - 1.0,
                "cumulative_regret": cumulative_regret,
            },
            observer.get_metrics(),
        )

        action = 5
        observer.observe(env=_FAKE_ENV, action=action, observation=_OBSERVATION)
        cumulative_regret += _NUM_ACTION - action - 1.0
        self.assertEqual(
            {
                "observer_step": 3,
                "action": action,
                "exp_reward": action,
                "regret": _NUM_ACTION - action - 1.0,
                "cumulative_regret": cumulative_regret,
            },
            observer.get_metrics(),
        )


if __name__ == "__main__":
    absltest.main()
