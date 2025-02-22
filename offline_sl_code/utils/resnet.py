"""
    Resnet Policy Net drawed by the paper 
    "RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning"
"""
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# from o0_global_const import PartialProduct
from ipdb import set_trace
# resnet
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)
            # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False)
                # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x, kernel_size=(x.shape[2], x.shape[3])
        )

# TODO: DeepQ 没有选动作这个函数，需要拷贝过来；
class DeepQPolicy(nn.Module):
    def __init__(
        self, block, input_channels=2, num_block=[2, 2, 2, 2], 
        num_classes=(8*2)*4, EPS_START = 0.9, 
        EPS_END = 0.10, EPS_DECAY = 500, MAX_STAGE_NUM=4,
        bit_width='8_bits', int_bit_width=8, str_bit_width=8, device='cpu',
        is_rnd_predictor=False, pool_is_stochastic=True, is_column_mask=False, is_factor=False
    ):
        super(DeepQPolicy, self).__init__()
        self.input_channels = input_channels
        self.in_channels = 64
        # EPS Hyperparameter
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        # stage num
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.bit_width = bit_width
        self.int_bit_width = int_bit_width
        self.str_bit_width = str_bit_width
        self.device = device
        self.is_rnd_predictor = is_rnd_predictor
        self.pool_is_stochastic = pool_is_stochastic
        self.num_classes = num_classes
        self.is_factor = is_factor

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 1)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)

        if pool_is_stochastic:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = GlobalAvgPool2d()


        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(x, dim=-1)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return output

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

class MBRLPPAModel(nn.Module):
    def __init__(
        self, block, model_hidden_dim=256, input_channels=2,
        num_block=[2, 2, 2, 2], pool_is_stochastic=True, device="cuda:0"
    ):
        super(MBRLPPAModel, self).__init__()
        self.input_channels = input_channels
        self.in_channels = 64
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 1)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)

        if pool_is_stochastic:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = GlobalAvgPool2d()

        # area model 
        self.area_model = nn.Sequential(
                nn.Linear(512 * block.expansion, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, 1)
        )
        # delay model
        self.delay_model = nn.Sequential(
                nn.Linear(512 * block.expansion, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, 1)
        )

    def forward(self, x):
        x = x.to(self.device)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        area = self.area_model(output)
        delay = self.delay_model(output)

        return area, delay

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)