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

"""Tests for bandit environment."""


from absl.testing import absltest


import numpy as np
from rlba.environments.bernoulli_bandit import BernoulliBanditEnv
from rlba.types import ArraySpec, DiscreteArraySpec


class BernoulliBanditEnvTest(absltest.TestCase):

    env = BernoulliBanditEnv([0, 1, 0.5], seed=0)

    def test_all_zero_arm(self):
        obs = np.array([self.env.step(0) for _ in range(100)])
        self.assertTrue(np.all(obs == 0))

    def test_all_one_arm(self):
        obs = np.array([self.env.step(1) for _ in range(100)])
        self.assertTrue(np.all(obs == 1))

    def test_unbiased_arm(self):
        obs = np.array([self.env.step(2) for _ in range(1000)])
        self.assertLess(np.abs(np.sum(obs) / len(obs) - 0.5), 0.05)


if __name__ == "__main__":
    absltest.main()
