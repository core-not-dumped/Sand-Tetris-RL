
import torch
import torch.nn as nn
import gym
from variables.global_variables import *
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import make_atari_env

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            print(f'{n_flatten = }')

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observation normalization
        observations = (observations - env_mean) / env_std
        return self.linear(self.cnn(observations))
    

# large receptive field
class SandTetrisCNN_V1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2, padding=3),  # RF: 7
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # RF: 15
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),  # RF: 31
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # RF: 63 (>45)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # RF 유지
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature dimension 계산
        self._feature_size = self._get_conv_output(observation_space.shape)
        print(self._feature_size)

        # Final linear layer
        self.fc = nn.Linear(self._feature_size, features_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He로 변경
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers(input)
            return int(torch.numel(output) / output.shape[0])
    
    def forward(self, observations):
        x = (observations - env_mean) / env_std
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    


class ResidualConvDW(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=8, dilation=1):
        super().__init__()
        if isinstance(kernel, int): kernel_size = (kernel, kernel)
        else:                       kernel_size = kernel
        padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)

        # Depthwise conv
        self.dw_conv1 = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, dilation=dilation,
            padding=padding, groups=in_ch, bias=False
        )
        # Pointwise conv
        self.pw_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU()

        # skip conv if channel/stride mismatch
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.dw_conv1(x)
        out = self.pw_conv1(out)
        out = self.gn1(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        out = self.act(out)
        return out

class SandTetrisCNN_V2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        self.block1 = ResidualConvDW(32, 64, kernel=5, stride=2, groups=8)
        self.block2 = ResidualConvDW(64, 64, kernel=5, stride=2, groups=8)
        self.block3 = ResidualConvDW(64, 128, kernel=3, stride=1, groups=8)
        self.block4 = ResidualConvDW(128, 128, kernel=3, stride=1, groups=8)
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 4, features_dim),
            nn.ReLU()
        )

        # 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        x = (observations - env_mean) / env_std
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

class SandTetrisCNN_V3(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(13, 5), stride=2, padding=(6, 2), bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        self.block1 = ResidualConvDW(32, 64, kernel=(7, 3), stride=2, groups=8)
        self.block2 = ResidualConvDW(64, 64, kernel=(7, 3), stride=1, groups=8)
        self.block3 = ResidualConvDW(64, 128, kernel=(5, 3), stride=1, groups=8)
        self.block4 = ResidualConvDW(128, 128, kernel=(5, 3), stride=1, groups=8)
        pooling_width, pooling_height = self.get_pooling_size(observation_space.shape)
        #self.pool = nn.Conv2d(128, 128, (pooling_width,5), (pooling_width,3), bias=False)
        #self.pooling_norm = nn.GroupNorm(8, 128)
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, features_dim),
            nn.ReLU()
        )

        # 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_pooling_size(self, obs_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            dummy = self.stem(dummy)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            dummy = self.block4(dummy)
            return dummy.shape[2], dummy.shape[3]

    def forward(self, observations):
        x = (observations - env_mean) / env_std
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        #x = self.pooling_norm(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x