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


from typing import Callable
from rlba.environment import Environment
from rlba.testing import fakes
from rlba.types import Array, ArraySpec, DiscreteArraySpec
from rlba.utils.observers import bandit
import numpy as np

from absl.testing import absltest, parameterized


def _make_fake_env(num_action: int, reward_fn: Callable[[int], Array]) -> Environment:
    action_spec = DiscreteArraySpec(num_action, dtype=np.int32)
    observation_spec = ArraySpec(shape=(10, 5), dtype=np.float32)
    EnvClass = fakes.FakeEnvironment
    reward_array = reward_fn(num_action)
    EnvClass.expected_reward = lambda s, a: reward_array[a]
    EnvClass.optimal_expected_reward = lambda s: reward_array.max()
    return EnvClass(action_spec, observation_spec, 0)


class BanditRegretTest(parameterized.TestCase):
    def test_observe_nothing(self):
        observer = bandit.RegretObserver()
        self.assertEqual({}, observer.get_metrics())

    @parameterized.parameters([1, 3, 5])
    def test_observe_reward_zero(self, n_action: int):
        fake_env = _make_fake_env(n_action, np.zeros)
        action = fakes.generate_from_spec(fake_env.action_spec)
        observation = fakes.generate_from_spec(fake_env.observation_spec)
        observer = bandit.RegretObserver()
        reward = np.random.random()
        observer.observe(
            env=fake_env, action=action, observation=observation, reward=reward
        )
        self.assertEqual(
            {
                "observer_step": 1,
                "action": action,
                "reward": reward,
                "exp_reward": 0.0,
                "regret": 0.0,
                "cumulative_regret": 0.0,
            },
            observer.get_metrics(),
        )

    @parameterized.parameters([1, 3, 5])
    def test_observe_reward_one(self, n_action: int):
        fake_env = _make_fake_env(n_action, np.ones)
        action = fakes.generate_from_spec(fake_env.action_spec)
        observation = fakes.generate_from_spec(fake_env.observation_spec)
        observer = bandit.RegretObserver()
        reward = np.random.random()
        observer.observe(
            env=fake_env, action=action, observation=observation, reward=reward
        )
        self.assertEqual(
            {
                "observer_step": 1,
                "action": action,
                "reward": reward,
                "exp_reward": 1.0,
                "regret": 0.0,
                "cumulative_regret": 0.0,
            },
            observer.get_metrics(),
        )

    @parameterized.parameters([1, 3, 5])
    def test_observe_reward_range(self, n_action: int):
        fake_env = _make_fake_env(n_action, np.arange)
        action = fakes.generate_from_spec(fake_env.action_spec)
        observation = fakes.generate_from_spec(fake_env.observation_spec)
        observer = bandit.RegretObserver()
        reward = np.random.random()
        observer.observe(
            env=fake_env, action=action, observation=observation, reward=reward
        )
        self.assertEqual(
            {
                "observer_step": 1,
                "action": action,
                "reward": reward,
                "exp_reward": 0.0,
                "regret": n_action - 1.0,
                "cumulative_regret": n_action - 1.0,
            },
            observer.get_metrics(),
        )

    @parameterized.parameters([6, 8, 13])
    def test_observe_multiple_step(self, n_action: int):
        fake_env = _make_fake_env(n_action, np.arange)
        observation = fakes.generate_from_spec(fake_env.observation_spec)
        observer = bandit.RegretObserver()
        reward = np.random.random()
        action = 1
        observer.observe(
            env=fake_env, action=action, observation=observation, reward=reward
        )
        cumulative_regret = n_action - action - 1.0
        self.assertEqual(
            {
                "observer_step": 1,
                "action": action,
                "reward": reward,
                "exp_reward": action,
                "regret": n_action - action - 1.0,
                "cumulative_regret": cumulative_regret,
            },
            observer.get_metrics(),
        )
        action = 3
        observer.observe(
            env=fake_env, action=action, observation=observation, reward=reward
        )
        cumulative_regret += n_action - action - 1.0
        self.assertEqual(
            {
                "observer_step": 2,
                "action": action,
                "reward": reward,
                "exp_reward": action,
                "regret": n_action - action - 1.0,
                "cumulative_regret": cumulative_regret,
            },
            observer.get_metrics(),
        )

        action = 5
        observer.observe(
            env=fake_env, action=action, observation=observation, reward=reward
        )
        cumulative_regret += n_action - action - 1.0
        self.assertEqual(
            {
                "observer_step": 3,
                "action": action,
                "reward": reward,
                "exp_reward": action,
                "regret": n_action - action - 1.0,
                "cumulative_regret": cumulative_regret,
            },
            observer.get_metrics(),
        )


if __name__ == "__main__":
    absltest.main()
