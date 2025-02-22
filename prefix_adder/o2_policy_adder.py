import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import random
import math
from typing import Tuple, Dict

from o1_environment_adder import State


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
                # nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))


class DeepQPolicy(nn.Module):
    def __init__(
        self,
        block,
        num_block=[2, 2, 2, 2],
        num_classes=(8 * 2) * 4,
        EPS_START=0.9,
        legal_num_column=200,
        EPS_END=0.10,
        EPS_DECAY=500,
        task_index=0,
        bit_width=8,
        action_num=4,
        device="cpu",
        is_rnd_predictor=False,
        pool_is_stochastic=True,
        is_column_mask=False,
        is_factor=False,
        is_multi_obj=False,
    ):
        super(DeepQPolicy, self).__init__()
        self.in_channels = 64
        # EPS Hyperparameter
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        # stage num
        self.bit_width = bit_width
        self.action_num = action_num
        self.device = device
        self.is_rnd_predictor = is_rnd_predictor
        self.pool_is_stochastic = pool_is_stochastic
        self.num_classes = num_classes
        self.is_factor = is_factor
        self.is_multi_obj = is_multi_obj
        self.legal_num_column = legal_num_column
        self.task_index = task_index

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 1)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)

        if pool_is_stochastic:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = GlobalAvgPool2d()

        if self.is_rnd_predictor:
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, bit_width**2 * 2),
            )
        elif not self.is_factor and not self.is_multi_obj:
            self.fc = nn.Linear(512 * block.expansion, bit_width**2 * 2)

        self.is_column_mask = is_column_mask
        if is_column_mask:
            num_column_mask = int(self.bit_width * 2 * 4)
            self.column_mask = np.ones(num_column_mask)
            for i in range(16):
                self.column_mask[i * 4 : (i + 1) * 4] = 0
                self.column_mask[
                    num_column_mask - (i + 1) * 4 : num_column_mask - i * 4
                ] = 0
            self.column_mask = self.column_mask != 0
            print("\r", self.column_mask)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def partially_reset(self, reset_type="xavier"):
        if reset_type == "xavier":
            nn.init.xavier_uniform_(self.fc.weight)
        else:
            raise NotImplementedError

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

    def forward(self, x_: State, tensor_x=None, is_target=False, state_mask=None):
        if tensor_x is None:
            x = torch.tensor(x_.cell_map).unsqueeze(0).unsqueeze(0).double()
        else:
            x = tensor_x
        if state_mask is not None:
            mask = state_mask
        else:
            if is_target:
                mask = self.mask_with_legality(x_)
            else:
                mask = self.mask(x_)
        x = x.float().to(self.device)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = output.masked_fill(~mask.to(self.device), -1000)

        return output

    def mask(self, state: State) -> torch.Tensor:
        choise_list = state.available_choice_list
        mask = np.zeros(2 * state.input_bit**2)

        for action in choise_list:
            mask[action] = 1

        mask = mask != 0
        return torch.from_numpy(mask)

    def mask_with_legality(self, state: State) -> torch.Tensor:
        choise_list = state.available_choice_list
        mask = np.zeros(2 * state.input_bit**2)

        for action in choise_list:
            mask[action] = 1
        mask = mask != 0
        return torch.from_numpy(mask)

    def select_action(
        self, state: State, steps_done: int, deterministic=False, is_softmax=False
    ) -> Tuple[torch.Tensor, Dict]:
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * steps_done / self.EPS_DECAY
        )

        if deterministic:
            eps_threshold = 0.0

        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                mask = self.mask_with_legality(state).to(self.device)
                state_numpy = np.asanyarray(state.cell_map)
                state_tensor = torch.from_numpy(state_numpy)
                state_tensor = state_tensor.unsqueeze(0)

                simple_mask = self.mask(state)
                q = self(state, state_mask=mask)
                neg_inf = torch.tensor(float("-inf"), device=self.device)
                q = q.masked_fill(~mask, neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q

                if is_softmax:
                    q_distribution = Categorical(logits=q)
                    action = q_distribution.sample()
                else:
                    action = q.max(1)[1]
                return action.view(1, 1), info
        else:
            # 随机
            mask = self.mask_with_legality(state)
            simple_mask = self.mask(state)
            index = torch.zeros((int(self.bit_width**2)) * 2)
            for i in range(0, (int(self.bit_width**2)) * 2):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0

            return torch.tensor(
                [[int(random.choice(index))]], device=self.device, dtype=torch.long
            ), info

class FactorDeepQPolicy(DeepQPolicy):
    def __init__(
        self, block, **policy_kwargs
    ):
        # action space 
        # action_type = action_value / 2
        # action_x = (action_value % bit**2) / bit_width
        # action_y = (action_value % bit**2) % bit_width
        super(FactorDeepQPolicy, self).__init__(
            block, is_factor=True, **policy_kwargs
        )
        assert self.is_rnd_predictor != True
        num_action_x = self.bit_width 
        num_action_y = self.bit_width 
        num_action_type = 2
        self.fc_x = nn.Linear(512 * block.expansion, num_action_x)
        self.fc_y = nn.Linear(512 * block.expansion, num_action_y)
        self.fc_type = nn.Linear(512 * block.expansion, num_action_type)

    def _combine(self, output_x, output_y, output_type):
        batch_size = output_x.shape[0]
        num_classes = output_x.shape[1] * output_y.shape[1] * output_type.shape[1]
        output = torch.zeros(
            (batch_size, num_classes),
            dtype=torch.float,
            device=output_x.device
        )
        for k in range(output_type.shape[1]):
            for i in range(output_x.shape[1]):
                for j in range(output_y.shape[1]):        
                    output[:,k*(self.bit_width**2)+i*self.bit_width+j] = output_x[:,i] + output_y[:,j] + output_type[:,k]
        return output

    def forward(self, x_: State, tensor_x=None, is_target=False, state_mask=None):
        if tensor_x is None:
            x = torch.tensor(x_.cell_map).unsqueeze(0).unsqueeze(0).double()
        else:
            x = tensor_x
        if state_mask is not None:
            mask = state_mask
        else:
            if is_target:
                mask = self.mask_with_legality(x_)
            else:
                mask = self.mask(x_)
        x = x.float().to(self.device)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        # factor action value compute
        output_x = self.fc_x(output) # (batch_size, num_column)
        output_y = self.fc_y(output)
        output_type = self.fc_type(output) # (batch_size, num_type)
        output = self._combine(output_x, output_y, output_type)
        output = output.masked_fill(~mask.to(self.device),-1000)

        return output