import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torchvision
from ipdb import set_trace
import os
from o0_state import State

# resnet


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
        return torch.nn.functional.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))


class DeepQPolicy(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        num_block=[2, 2, 2, 2],
        num_classes=(8 * 2) * 4,
        action_type_num=4,  # 动作的种类数量，用于确定通道数
        EPS_START=0.9,
        EPS_END=0.10,
        EPS_DECAY=500,
        device="cpu",
        is_rnd_predictor=False,
        pool_is_stochastic=True,
        is_factor=False,
        is_multi_obj=False,
    ):
        super(DeepQPolicy, self).__init__()
        self.in_channels = 64
        self.action_type_num = action_type_num
        # EPS Hyperparameter
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        # stage num
        self.device = device
        self.is_rnd_predictor = is_rnd_predictor
        self.pool_is_stochastic = pool_is_stochastic
        self.num_classes = num_classes
        self.is_factor = is_factor
        self.is_multi_obj = is_multi_obj

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
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
                nn.Linear(512, num_classes),
            )
        elif not self.is_factor and not self.is_multi_obj:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x, is_target=False, state_mask=None):
        # set_trace()
        x = x.float().to(self.device)
        if state_mask is not None:
            mask = state_mask
        else:
            if is_target:
                mask = self.mask_with_legality(x)
            else:
                mask = self.mask(x)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = output.masked_fill(~mask.to(self.device), -1000)

        # actor: choses action to take from state s_t
        # by returning probability of each action
        # action_prob = F.softmax(x, dim=-1)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return output

    # fmt: off
    def select_action(self, state: State, steps_done, deterministic=False, is_softmax=False):
        info = {}
        sample = np.random.uniform(0, 1)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * steps_done / self.EPS_DECAY)

        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = state.get_stage_num()
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(state.shape)

                tensor_state = torch.tensor(state.archive(), device=self.device)
                mask = torch.tensor(state.mask_with_legality(), device=self.device)
                simple_mask = torch.tensor(state.mask())
                tensor_state = tensor_state.unsqueeze(0)
                q: torch.Tensor = self(tensor_state, state_mask=mask)

                neg_inf = torch.tensor(float('-inf'), device=self.device)
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
            mask = torch.tensor(state.mask_with_legality())
            simple_mask = torch.tensor(state.mask())
            index = torch.zeros(state.get_pp_len() * 4)
            for i in range(0, state.get_pp_len() * 4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info
    # fmt: on


class FactorDeepQPolicy(DeepQPolicy):
    def __init__(self, block, **policy_kwargs):
        super(FactorDeepQPolicy, self).__init__(block, is_factor=True, **policy_kwargs)
        assert self.is_rnd_predictor != True
        num_action_column = int(self.num_classes / 4)
        num_action_type = 4
        self.fc_column = nn.Linear(512 * block.expansion, num_action_column)
        self.fc_type = nn.Linear(512 * block.expansion, num_action_type)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def _combine(self, output_column, output_type):
        batch_size = output_column.shape[0]
        num_classes = output_column.shape[1] * output_type.shape[1]
        output = torch.zeros(
            (batch_size, num_classes), dtype=torch.float, device=output_column.device
        )
        for i in range(output_column.shape[1]):
            for j in range(output_type.shape[1]):
                output[:, i * 4 + j] = output_column[:, i] + output_type[:, j]
        return output

    def forward(self, x, is_target=False, state_mask=None):
        x = x.to(self.device).float()
        if state_mask is not None:
            mask = state_mask
        else:
            if is_target:
                mask = self.mask_with_legality(x)
            else:
                mask = self.mask(x)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output_column = self.fc_column(output)  # (batch_size, num_column)
        output_type = self.fc_type(output)  # (batch_size, num_type)
        output = self._combine(output_column, output_type)
        output = output.masked_fill(~mask.to(self.device), -1000)

        # actor: choses action to take from state s_t
        # by returning probability of each action
        # action_prob = F.softmax(x, dim=-1)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return output


class Resnet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        num_block=[2, 2, 2, 2],
        num_mask=90,
        input_mask_channels=4,
        pool_is_stochastic=True,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_mask_channels, 64, kernel_size=3, padding=1, bias=False),
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
        self.fc = nn.Linear(512 * block.expansion, num_mask)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class MaskDeepQPolicy(nn.Module):
    """
    仿照 maskplace
    """

    # fmt: off
    def __init__(
        self,
        block=BasicBlock,
        device="cpu",
        num_classes=60, # pp_width x action_type_num
        num_mask=90,  # max_stage_num x pp_width
        num_block=[2, 2, 2, 2],
        input_state_channels=2, # e.g. ct 的通道数 (+ comap, = 4)
        input_feature_channels=2, # e.g. power mask 的通道数
        action_type_num=4, # 动作的种类数量，用于确定通道数
        EPS_START=0.9,
        EPS_END=0.10,
        EPS_DECAY=500,
        is_rnd_predictor=False,
        pool_is_stochastic=True,
        is_factor=False,
        is_multi_obj=False,
        pretrained_path=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_mask = num_mask
        self.action_type_num = action_type_num

        self.input_state_channels = input_state_channels
        self.input_feature_channels = input_feature_channels

        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY

        self.is_rnd_predictor = is_rnd_predictor
        self.is_factor = is_factor
        self.is_multi_obj = is_multi_obj

        local_channels = action_type_num + input_feature_channels
        self.local_fusion_net = nn.Sequential(
            nn.Conv2d(local_channels, 2 * local_channels, 1), # action mask + power mask
            nn.ReLU(),
            nn.Conv2d(2 * local_channels, 2 * local_channels, 1),
            nn.ReLU(),
            nn.Conv2d(2 * local_channels, 1, 1),
        )

        self.global_encoder_net = Resnet(
            block, num_block, num_mask, input_feature_channels + input_state_channels, pool_is_stochastic
        )
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.global_encoder_net.load_state_dict(torch.load(pretrained_path))
        self.merge = nn.Conv2d(2, 1, 1)
        if self.is_rnd_predictor:
            self.global_encoder_net.fc = nn.Sequential(
                nn.Linear(block.expansion * 512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_mask),
            )
            self.fc = nn.Linear(num_mask, num_classes)
        elif not self.is_factor and not self.is_multi_obj:
            self.fc = nn.Linear(num_mask, num_classes)

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: shape [批次大小, 通道数量, max_stage_num, pp width]
                通道 0, 1 : 压缩树
                通道 2, 3 : power mask
                通道 4 - 7: action_mask
        """
        x = x.float().to(self.device)
        max_stage_num = x.shape[2]
        pp_width = x.shape[3]
        # $$DEBUG
        # assert self.num_classes == self.action_type_num * pp_width
        state_mask = x[:, :self.input_state_channels, :, :]
        feature_mask = x[:, self.input_state_channels : self.input_state_channels + self.input_feature_channels, :, :]
        action_mask = x[:, self.input_state_channels + self.input_feature_channels:, :]  # channel = self.action_type_num

        local_mask_input = torch.concat([feature_mask, action_mask], 1)  # channel = 6
        local_mask = self.local_fusion_net(local_mask_input) # channel = 1

        global_mask_input = torch.concat([state_mask, feature_mask], 1)  # channel = 4
        global_mask = self.global_encoder_net(global_mask_input).reshape([-1, 1, max_stage_num, pp_width]) # channel = 1

        merge_input = local_mask_input = torch.concat([local_mask, global_mask], 1) # channel = 2
        merge_mask = self.merge(merge_input) # channel = 1
        merge_mask = merge_mask.reshape([-1, max_stage_num * pp_width])

        action_value:torch.Tensor = self.fc(merge_mask)

        action_mask_bool = action_mask > 0.0
        batch_indices, action_type, _, action_column = torch.where(action_mask_bool)
        action_indices = action_column * self.action_type_num + action_type
        mask = torch.zeros((action_mask_bool.shape[0], self.num_classes), dtype=torch.bool)
        mask[batch_indices, action_indices] = True

        action_value = action_value.masked_fill(~mask.to(self.device), -1e3)
        return action_value
    
    # fmt: off
    def select_action(self, state: State, steps_done, deterministic=False, is_softmax=False):
        info = {}
        sample = np.random.uniform(0, 1)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * steps_done / self.EPS_DECAY)

        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = state.get_stage_num()
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():

                mask = torch.tensor(state.mask_with_legality(), device=self.device)
                simple_mask = torch.tensor(state.mask())
                tensor_state = torch.tensor(state.archive(return_mask=True), device=self.device)
                tensor_state = tensor_state.unsqueeze(0)
                q: torch.Tensor = self(tensor_state)

                neg_inf = torch.tensor(float('-inf'), device=self.device)
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
            mask = torch.tensor(state.mask_with_legality())
            simple_mask = torch.tensor(state.mask())
            index = torch.zeros(state.get_pp_len() * self.action_type_num)
            for i in range(0, state.get_pp_len() * self.action_type_num):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info
    # fmt: on


class MaskFactorDeepQPolicy(MaskDeepQPolicy):
    def __init__(self, block, **policy_kwargs) -> None:
        super().__init__(block, is_factor=True, **policy_kwargs)
        assert self.is_rnd_predictor != True
        num_action_column = int(self.num_classes / self.action_type_num)
        self.fc_column = nn.Linear(self.num_mask, num_action_column)
        self.fc_type = nn.Linear(self.num_mask, self.action_type_num)

    def _combine(self, output_column, output_type):
        batch_size = output_column.shape[0]
        num_classes = output_column.shape[1] * output_type.shape[1]
        output = torch.zeros(
            (batch_size, num_classes), dtype=torch.float, device=output_column.device
        )
        for i in range(output_column.shape[1]):
            for j in range(output_type.shape[1]):
                output[:, i * self.action_type_num + j] = (
                    output_column[:, i] + output_type[:, j]
                )
        return output

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: shape [批次大小, 通道数量, max_stage_num, pp width]
                通道 0, 1 : 压缩树
                通道 2, 3 : power mask
                通道 4 - 7: action_mask
        """
        x = x.float().to(self.device)
        max_stage_num = x.shape[2]
        pp_width = x.shape[3]
        # $$DEBUG
        # assert self.num_classes == self.action_type_num * pp_width
        state_mask = x[:, :self.input_state_channels, :, :]
        feature_mask = x[:, self.input_state_channels : self.input_state_channels + self.input_feature_channels, :, :]
        action_mask = x[:, self.input_state_channels + self.input_feature_channels:, :]  # channel = self.action_type_num

        local_mask_input = torch.concat([feature_mask, action_mask], 1)  # channel = 6
        local_mask = self.local_fusion_net(local_mask_input) # channel = 1

        global_mask_input = torch.concat([state_mask, feature_mask], 1)  # channel = 4
        global_mask = self.global_encoder_net(global_mask_input).reshape([-1, 1, max_stage_num, pp_width]) # channel = 1

        merge_input = local_mask_input = torch.concat([local_mask, global_mask], 1) # channel = 2
        merge_mask = self.merge(merge_input) # channel = 1
        merge_mask = merge_mask.reshape([-1, max_stage_num * pp_width])
        
        # factor !
        action_value_column = self.fc_column(merge_mask)
        action_value_type = self.fc_type(merge_mask)
        action_value = self._combine(action_value_column, action_value_type)

        action_mask_bool = action_mask > 0.0
        batch_indices, action_type, _, action_column = torch.where(action_mask_bool)
        action_indices = action_column * self.action_type_num + action_type
        mask = torch.zeros((action_mask_bool.shape[0], self.num_classes), dtype=torch.bool)
        mask[batch_indices, action_indices] = True

        action_value = action_value.masked_fill(~mask.to(self.device), -1e3)
        return action_value


class PrefixDeepQPolicy(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        num_block=[2, 2, 2, 2],
        input_feature_channels=4,
        action_mask_loc=2,
        input_width=8,
        EPS_START=0.9,
        EPS_END=0.10,
        EPS_DECAY=500,
        device="cpu",
        pool_is_stochastic=True,
        is_factor=False,
        is_rnd_predictor=False,
    ) -> None:
        super().__init__()

        self.input_feature_channels = input_feature_channels
        self.input_width = input_width
        self.action_mask_loc = action_mask_loc

        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY

        self.device = device
        self.is_factor = is_factor
        self.is_rnd_predictor = is_rnd_predictor

        self.conv = Resnet(
            block,
            num_block,
            2 * input_width * input_width,
            input_feature_channels,
            pool_is_stochastic,
        )

    # fmt: off
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            channel 0: cell map
            channel 1: level map
            channel 2: action_mask_0
            channel 3: action_mask_1
        """
        x = x.float().to(self.device)
        action_mask = x[:, self.action_mask_loc : self.action_mask_loc + 2, :, :].flatten()
        action_mask_bool = action_mask > 0

        action_value: torch.Tensor = self.conv(x)
        action_value = action_value.masked_fill(~action_mask_bool.to(self.device), 1e-3)

        return action_value

    # fmt: off
    def select_action(self, state: State, steps_done, deterministic=False, is_softmax=False):
        """
            action_type = action // (input_bit ** 2)
            x = (action % (input_bit ** 2)) // input_bit
            y = (action % (input_bit ** 2)) % input_bit
            
            action = action_type * input_bit ** 2 + x * input_bit + y
        """
        info = {}
        sample = np.random.uniform(0, 1)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * steps_done / self.EPS_DECAY)

        if deterministic:
            eps_threshold = 0.

        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                if self.input_feature_channels == 4:
                    tensor_state = torch.tensor(state.archive_cell_map(), device=self.device)
                else:
                    tensor_state = torch.tensor(state.archive_cell_map(True), device=self.device)
                mask = torch.tensor(state.mask_cell_map(), device=self.device)
                tensor_state = tensor_state.unsqueeze(0)
                q: torch.Tensor = self(tensor_state)
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask, neg_inf)

                info["mask"] = mask.cpu()
                info["q_value"] = q

                if is_softmax:
                    q_distribution = Categorical(logits=q)
                    action = q_distribution.sample()
                else:
                    action = q.max(1)[1]
                return action.view(1, 1), info
        else:
            mask = state.mask_cell_map()
            mask = mask > 0
            indices = np.where(mask)[0]
            index = np.random.choice(indices)

            info["mask"] = mask
            info["q_value"] = 0

            return torch.tensor([[int(index)]], device=self.device, dtype=torch.long), info


class FactorPrefixDeepQPolicy(PrefixDeepQPolicy):
    def __init__(self, block, **policy_kwargs) -> None:
        super().__init__(block, is_factor=True, **policy_kwargs)
        assert self.is_rnd_predictor != True

        self.conv.fc = nn.Identity()
        self.fc_type =  nn.Linear(512 * block.expansion, 2)
        self.fc_x =  nn.Linear(512 * block.expansion, self.input_width)
        self.fc_y =  nn.Linear(512 * block.expansion, self.input_width)

    def __combine(self, output_type, output_x, output_y):
        batch_size = output_type.shape[0]
        action_value = torch.zeros((batch_size, 2 * self.input_width ** 2), dtype=torch.float, device=output_type.device)

        for i in range(output_type.shape[1]):
            for j in range(output_x.shape[1]):
                for k in range(output_y.shape[1]):
                    action_value[:, i * (self.input_width ** 2) + j * self.input_width + k] = output_type[:, i] + output_x[:, j] + output_y[:, k]
        
        return action_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            channel 0: cell map
            channel 1: level map
            channel 2: action_mask_0
            channel 3: action_mask_1
        """
        x = x.float().to(self.device)
        action_mask = x[:, self.action_mask_loc : self.action_mask_loc + 2, :, :].flatten()
        action_mask_bool = action_mask > 0

        output: torch.Tensor = self.conv(x)

        output_type = self.fc_type(output)
        output_x = self.fc_x(output)
        output_y = self.fc_y(output)

        action_value = self.__combine(output_type, output_x, output_y)
        action_value = action_value.masked_fill(~action_mask_bool.to(self.device), 1e-3)

        return action_value



if __name__ == "__main__":
    pass
