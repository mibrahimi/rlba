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

"""Tests for assortment recommendation environment."""


from absl.testing import absltest, parameterized


import numpy as np
from rlba.environments.assortment import (
    AssortmentRecommendationEnv,
    choose_optimal_assortment,
)
from rlba.types import ArraySpec, DiscreteArraySpec


class OptimalAssortmentTest(parameterized.TestCase):
    @parameterized.parameters([5, 1], [13, 4])
    def test_optimal_assortment_zeros(self, n_item: int, n_slot: int):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            theta = np.append(np.zeros(n_item - 1), 1)
            selected_items = choose_optimal_assortment(theta, n_slot, rng)
            self.assertEqual(selected_items.sum(), n_slot)
            self.assertTrue(selected_items[n_item - 1])

    @parameterized.parameters([5, 3], [13, 4])
    def test_optimal_assortment_range(self, n_item: int, n_slot: int):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            theta = np.arange(n_item)
            selected_items = choose_optimal_assortment(theta, n_slot, rng)
            self.assertEqual(selected_items.sum(), n_slot)
            self.assertTrue((selected_items[-n_slot:] == np.ones(n_slot)).all())
            self.assertTrue(
                (selected_items[:-n_slot] == np.zeros(n_item - n_slot)).all()
            )


class AssortmentRecommendationEnvTest(parameterized.TestCase):
    @parameterized.parameters([5, 3], [13, 4])
    def test_correct_dimensions(self, n_item: int, n_slot: int):
        for seed in range(10):
            env = AssortmentRecommendationEnv(n_item, n_slot, seed, sigma_p=1)
            self.assertEqual(env._theta.shape, (n_item,))

    @parameterized.parameters([5, 3], [13, 4])
    def test_uniform_item_probs(self, n_item: int, n_slot: int):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            env = AssortmentRecommendationEnv(n_item, n_slot, seed, sigma_p=1)
            env._theta = np.zeros_like(env._theta)
            for _ in range(5):
                action = np.zeros(n_item)
                action[rng.choice(range(n_item), size=n_slot, replace=False)] = 1
                self.assertEqual(env.expected_reward(action), 1.0 - 1.0 / (n_slot + 1))


if __name__ == "__main__":
    absltest.main()
