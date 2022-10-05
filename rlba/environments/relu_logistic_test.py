from absl.testing import absltest, parameterized


import numpy as np
from rlba.environments.relu_logistic_bandit import ReLULogisticBandit
from rlba.types import ArraySpec, DiscreteArraySpec

class ReLULogisticBanditTest(absltest.TestCase):
    
    def test_correct_dimensions(self):
        n_action = 5
        n_context = 3
        seed = 0
        layer_dims = [3,4]
        env = ReLULogisticBandit(seed = seed, 
                             n_context = n_context,
                             n_action = n_action,
                             layer_dims = layer_dims)
        self.assertEqual(env._exp_reward.shape, (n_context, n_action))
        self.assertEqual(env._optimal_exp_reward.shape, (n_context, 1))
        self.assertEqual(env.get_feature().shape, (n_context, n_action, 
                                                   layer_dims[0]))
        
    def test_unbiased_arm(self):
        n_action = 5
        n_context = 3
        seed = 0
        layer_dims = [3,4]
        env = ReLULogisticBandit(seed = seed, 
                             n_context = n_context,
                             n_action = n_action,
                             layer_dims = layer_dims)
        
        fixed_action = 0

        reward_probs = env._rewards
        obs = np.array([env.step(fixed_action)[0] for i in range(1000)])
        # choosing same action across contexts should give an average over contexts 
        self.assertLess(np.abs(np.mean(obs) - 
                               np.mean(reward_probs[:,fixed_action])), 0.05)

if __name__ == "__main__":
    absltest.main()