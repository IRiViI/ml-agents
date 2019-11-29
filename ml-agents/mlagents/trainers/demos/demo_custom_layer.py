import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.trainer import PPOTrainer, discount_rewards
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.rl_trainer import AllRewardsOutput
from mlagents.trainers.components.reward_signals import RewardSignalResult
from mlagents.envs.brain import BrainParameters, CameraResolution
from mlagents.envs.environment import UnityEnvironment
from mlagents.envs.mock_communicator import MockCommunicator
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import make_brain_parameters
from mlagents.trainers.custom_layer_specs import CustomConvLayerSpecs

env_name = "../../../../../../ml-agents-master/envs/first_try_conv/Unity Environment"

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
        vis_encode_type: custom
        max_steps: 5.0e4
        normalize: true
        num_epoch: 5
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        use_recurrent: false
        memory_size: 8
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

def main():

    # Check if it is the correct folder
    import os
    environment_dir = "/".join(env_name.split("/")[:-1])
    for folder in os.listdir(environment_dir):
        print(folder)

    env = UnityEnvironment(env_name)

    brain_infos = env.reset(train_mode=True)
    default_brain = env.external_brain_names[0]
    brain_info = brain_infos[default_brain]

    trainer_parameters = dummy_config()

    trainer_parameters["layers_specs"] = [
        {
            "type": "conv2D",
            "filters": 32,
            "activation": "relu",
            "use_bias": True,
            "maxPool": False,
            "kernel_shape": (8,8),
            "strides": (4,4),
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        },
        {
            "type": "conv2D",
            "filters": 32,
            "activation": "relu",
            "use_bias": True,
            "maxPool": False,
            "kernel_shape": (4,4),
            "strides": (2,2),
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
        }
    ]

    camera_resolution = CameraResolution(84,84,3)
    brain_params = BrainParameters(
        "test_brain",        # brain_name
        0,                   # vector_observation_space_size
        0,                   # num_stacked_vector_observations
        [camera_resolution], # camera_resolutions
        [4],                 # vector_action_space_size
        [],                  # vector_action_descriptions
        1                    # vector_action_space_type
    )
    trainer = PPOTrainer(brain_params, 0, trainer_parameters, True, False, 0, "0", False)
    policy = trainer.policy
    model = policy.model

    brain_info = brain_infos[default_brain]

    run_out = policy.evaluate(brain_info)

    for key, val in run_out.items():
        print(key, ":", val)

    # trainer.

if __name__ == '__main__':
    main()
