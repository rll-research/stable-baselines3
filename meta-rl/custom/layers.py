from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


""" extending from stable_baselines3.common.torch_layers.py """

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_block0 = nn.Sequential(
            *[
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ])
        )
       self.res_block1 = nn.Sequential(
            *[
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ])
        )

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out0 = self.res_block0(x) 
        inp1 = out0 + x
        out1 = self.res_block1(inp1)
        return out1 + inp1

class ImpalaCNN(BaseFeaturesExtractor):
    """
    for procgen: use depths=[16,32,32], emb_size=256

    torch reference:
    https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
    openai tf:
    https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(ImpalaCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        assert n_input_channels == 3, 'Must have 3 channels'

        residual_blocks = []
        out_channels_list = [16, 32, 32]
        for i, out_channels in enumerate(out_channels_list):
            residual_blocks.append(
                ImpalaBlock(
                    in_channels=(n_input_channels if i == 0 else out_channels_list[i - 1]),
                    out_channels=out_channels,
                    )
                )
        residual_blocks.append(nn.Flatten())
        self.cnn = nn.Sequential(*residual_blocks)
        # Compute shape by doing one forward pass
        with th.no_grad():
            dummy_inp = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_inp).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # /255 ???
        print('forwarding impala cnn', observations.max(), observations.min())
        return self.linear(self.cnn(observations))

