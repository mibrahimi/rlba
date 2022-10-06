from absl.testing import absltest, parameterized


import numpy as np
from rlba.environments.relu_logistic_bandit import ReLULogisticBandit
from rlba.types import ArraySpec, DiscreteArraySpec

class ReLULogisticBanditTest(absltest.TestCase):
    
    def test_correct_dimensions(self):
        n_actions = 5
        n_contexts = 3
        seed = 0
        layer_dims = [3,4]
        env = ReLULogisticBandit(seed = seed, 
                             n_contexts = n_contexts,
                             n_actions = n_actions,
                             layer_dims = layer_dims)
        self.assertEqual(env._exp_reward.shape, (n_contexts, n_actions))
        self.assertEqual(env._optimal_exp_reward.shape, (n_contexts, 1))
        self.assertEqual(env.get_feature().shape, (n_contexts, n_actions, 
                                                   layer_dims[0]))
        
    def test_unbiased_arm(self):
        n_actions = 5
        n_contexts = 3
        seed = 0
        layer_dims = [3,4]
        env = ReLULogisticBandit(seed = seed, 
                             n_contexts = n_contexts,
                             n_actions = n_actions,
                             layer_dims = layer_dims)
        
        fixed_action = 0

        reward_probs = env._exp_reward
        obs = np.array([env.step(fixed_action)[0] for i in range(1000)])
        # choosing same action across contexts should give an average over contexts 
        self.assertLess(np.abs(np.mean(obs) - 
                               np.mean(reward_probs[:,fixed_action])), 0.05)

if __name__ == "__main__":
    absltest.main()