import unittest.mock as mock
import unittest
import pytest

import numpy as np
import tensorflow as tf
import yaml

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.trainer import PPOTrainer, discount_rewards
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.rl_trainer import AllRewardsOutput
from mlagents.trainers.components.reward_signals import RewardSignalResult
from mlagents.envs.brain import BrainParameters
from mlagents.envs.environment import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters
from mlagents.trainers.custom_layer_specs import CustomConvLayerSpecs

@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        trainer: ppo
        batch_size: 32
        beta: 5.0e-3
        buffer_size: 512
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 3.0e-4
        max_steps: 5.0e4
        normalize: true
        num_epoch: 5
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        use_recurrent: false
        memory_size: 8
        vis_encode_type: custom
        curiosity_strength: 0.0
        curiosity_enc_size: 1
        summary_path: test
        model_path: test
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


class TestStringMethods(unittest.TestCase):

    def test_custom_layer_specs(self):

        customConvLayerSpecs = CustomConvLayerSpecs(32,
            kernel_shape=(3,3), strides=(1,1), 
            activation='elu', 
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            use_bias=True, maxPool=False)

    @mock.patch("mlagents.envs.environment.UnityEnvironment.executable_launcher")
    @mock.patch("mlagents.envs.environment.UnityEnvironment.get_communicator")
    def test_ppo_model_cc_visual(self, mock_communicator, mock_launcher):

        tf.reset_default_graph()
        mock_communicator.return_value = MockCommunicator(
            discrete_action=False, visual_inputs=0
        )
        env = UnityEnvironment(" ")
        
        brain_infos = env.reset()
        brain_info = brain_infos[env.external_brain_names[0]]

        trainer_parameters = dummy_config()

        policy = PPOPolicy(
            0, env.brains[env.external_brain_names[0]], trainer_parameters, False, False
        )
        run_out = policy.evaluate(brain_info)
        assert run_out["action"].shape == (3, 2)
        print("----")
        print("boe")
        env.close()
                # model = PPOModel(
                #     make_brain_parameters(discrete_action=False, visual_inputs=2)
                # )
                # init = tf.global_variables_initializer()
                # sess.run(init)

                # run_list = [
                #     model.output,
                #     model.log_probs,
                #     model.value,
                #     model.entropy,
                #     model.learning_rate,
                # ]
                # feed_dict = {
                #     model.batch_size: 2,
                #     model.sequence_length: 1,
                #     model.vector_in: np.array([[1, 2, 3, 1, 2, 3], [3, 4, 5, 3, 4, 5]]),
                #     model.visual_in[0]: np.ones([2, 40, 30, 3]),
                #     model.visual_in[1]: np.ones([2, 40, 30, 3]),
                #     model.epsilon: np.array([[0, 1], [2, 3]]),
                # }
                # sess.run(run_list, feed_dict=feed_dict)

if __name__ == '__main__':
    unittest.main()