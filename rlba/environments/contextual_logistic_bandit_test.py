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
from rlba.environments.contextual_logistic_bandit import ContextualLogisticBandit
from rlba.types import ArraySpec, DiscreteArraySpec


class ContextualLogisticBanditTest(absltest.TestCase):

    env = ContextualLogisticBandit(n_actions=5, 
                                   n_contexts=3,
                                   dim=2,
                                   seed=0)

    def test_correct_dimensions(self):
        n_actions = 5
        n_contexts = 3
        dim = 7
        seed = 0
        env = ContextualLogisticBandit(n_actions=n_actions, 
                                       n_contexts=n_contexts,
                                       dim=dim,
                                       seed=seed)
        self.assertEqual(env._rewards.shape, (n_contexts, n_actions))
        self.assertEqual(env._values.shape, (n_contexts, 1))
        self.assertEqual(env._feature.shape, (n_contexts, n_actions, dim))

    def test_unbiased_arm(self):
        n_actions = 5
        n_contexts = 1
        dim = 7
        seed = 0
        env = ContextualLogisticBandit(n_actions=n_actions, 
                                       n_contexts=n_contexts,
                                       dim=dim,
                                       seed=seed)


        reward_probs = env._rewards
        obs = np.array([env.step(0)[0] for i in range(1000)])
        self.assertLess(np.abs(np.sum(obs) / obs.size - reward_probs[0,0]), 0.05)


if __name__ == "__main__":
    absltest.main()
