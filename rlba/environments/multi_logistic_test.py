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

"""Tests for multiple logistic environment."""


from absl.testing import absltest


import numpy as np
from rlba.environments.multi_logistic import MultipleLogisticEnv
from rlba.types import ArraySpec, DiscreteArraySpec


class MultipleLogisticEnvTest(absltest.TestCase):

    env = MultipleLogisticEnv(
        action_feature_dim=5, embedding_dim=3, output_dim=2, num_action=7, seed=0
    )

    def test_correct_dimensions(self):
        action_feature_dim = 5
        embedding_dim = 3
        output_dim = 2
        num_action = 7
        seed = 0
        env = MultipleLogisticEnv(
            action_feature_dim=action_feature_dim,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_action=num_action,
            seed=seed,
        )
        self.assertEqual(env._action_features.shape, (action_feature_dim, num_action))
        self.assertEqual(env._output_features.shape, (embedding_dim, output_dim))
        self.assertEqual(env._theta.shape, (action_feature_dim, embedding_dim))

    def test_unbiased_arm(self):
        action_feature_dim = 5
        embedding_dim = 3
        output_dim = 2
        num_action = 7
        seed = 0
        env = MultipleLogisticEnv(
            action_feature_dim=action_feature_dim,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_action=num_action,
            seed=seed,
        )

        env._action_features = np.zeros_like(env._action_features)
        obs = np.array([env.step(i % num_action) for i in range(1000)])
        self.assertLess(np.abs(np.sum(obs) / obs.size - 0.5), 0.05)

        env._reset(seed=1)
        env._theta = np.zeros_like(env._theta)
        obs = np.array([env.step(i % num_action) for i in range(1000)])
        self.assertLess(np.abs(np.sum(obs) / obs.size - 0.5), 0.05)


if __name__ == "__main__":
    absltest.main()
