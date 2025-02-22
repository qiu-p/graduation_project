"""
    Resnet Policy Net drawed by the paper 
    "RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning"
"""
import math
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from o0_global_const import PartialProduct, MacPartialProduct
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
        self, block, num_block=[2, 2, 2, 2], 
        num_classes=(8*2)*4, EPS_START = 0.9, legal_num_column=200,
        EPS_END = 0.10, EPS_DECAY = 500, MAX_STAGE_NUM=4, task_index=0,
        bit_width='8_bits_booth', int_bit_width=8, str_bit_width=8, action_num=4, device='cpu',
        is_rnd_predictor=False, pool_is_stochastic=True, 
        is_column_mask=False, is_factor=False, is_multi_obj=False, is_mac=False
    ):
        super(DeepQPolicy, self).__init__()
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
        self.action_num = action_num
        self.device = device
        self.is_rnd_predictor = is_rnd_predictor
        self.pool_is_stochastic = pool_is_stochastic
        self.num_classes = num_classes
        self.is_factor = is_factor
        self.is_multi_obj = is_multi_obj
        self.legal_num_column = legal_num_column
        self.task_index = task_index
        self.is_mac = is_mac

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
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

        if self.is_rnd_predictor:
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        elif not self.is_factor and not self.is_multi_obj:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.is_column_mask = is_column_mask
        if is_column_mask:
            num_column_mask = int(self.int_bit_width*2*4)
            self.column_mask = np.ones(
                num_column_mask
            )
            for i in range(16):
                self.column_mask[i*4:(i+1)*4] = 0
                self.column_mask[num_column_mask-(i+1)*4:num_column_mask-i*4] = 0
            self.column_mask = (self.column_mask!=0)
            print(self.column_mask)
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
        x = x.to(self.device)
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
        output = output.masked_fill(~mask.to(self.device),-1000)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(x, dim=-1)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return output

    def mask(self, state):
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32 = state[0]
        ct22 = state[1]
        ct32 = ct32.sum(axis=0).unsqueeze(0)
        ct22 = ct22.sum(axis=0).unsqueeze(0)
        state = torch.cat((ct32, ct22), dim=0)
        #state = torch.reshape(state, (2,int(self.int_bit_width*2)))
        state_np = state.cpu().numpy()
        np.savetxt(f'./build/state_{self.task_index}.txt', state_np, fmt="%d", delimiter=" ")
        legal_act = []
        mask = np.zeros((int(self.int_bit_width*2))*4)
        #initial_state = state
        pp = np.zeros(int(self.int_bit_width*2))
        for i in range(int(self.int_bit_width*2)):
            if self.is_mac:
                pp[i] = MacPartialProduct[self.bit_width][i]
            else:
                pp[i] = PartialProduct[self.bit_width][i]
        for i in range(2,int(self.int_bit_width*2)):
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #initial_pp = pp
        for i in range(2,int(self.int_bit_width*2)):
            if (pp[i] == 2):
                legal_act.append((i,0))
                if (state[1][i] >= 1):
                    legal_act.append((i,3))
            if (pp[i] == 1):
                if (state[0][i] >= 1):
                    legal_act.append((i,2))
                if (state[1][i] >= 1):
                    legal_act.append((i,1))
        for act_col, action in legal_act:
            mask[act_col * 4 + action] = 1

        mask = (mask!=0)
        return torch.from_numpy(mask)

    def mask_with_legality(self, state):
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32 = state[0]
        ct22 = state[1]
        ct32 = ct32.sum(axis=0).unsqueeze(0)
        ct22 = ct22.sum(axis=0).unsqueeze(0)
        state = torch.cat((ct32, ct22), dim=0)

        #state = torch.reshape(state, (2,int(self.int_bit_width*2)))
        try:
            state_np = state.numpy()
        except:
            state_np = state.cpu().numpy()
        np.savetxt(f'./build/state_{self.task_index}.txt', state_np, fmt="%d", delimiter=" ")
        legal_act = []
        mask = np.zeros((int(self.int_bit_width*2))*4)
        #initial_state = state
        pp = np.zeros(int(self.int_bit_width*2)+1)
        for i in range(int(self.int_bit_width*2)+1):
            if self.is_mac:
                pp[i] = MacPartialProduct[self.bit_width][i]
            else:
                pp[i] = PartialProduct[self.bit_width][i]

        for i in range(2,int(self.int_bit_width*2)):
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #initial_pp = pp
        for i in range(2,int(self.int_bit_width*2)):
            if (pp[i] == 2):
                legal_act.append((i,0))
                if (state[1][i] >= 1):
                    legal_act.append((i,3))
            if (pp[i] == 1):
                if (state[0][i] >= 1):
                    legal_act.append((i,2))
                if (state[1][i] >= 1):
                    legal_act.append((i,1))
        
        for act_col, action in legal_act:
            #state = initial_state
            df = pd.read_csv(f"./build/state_{self.task_index}.txt", header=None, sep=' ')
            df = df.to_numpy()
            state = torch.tensor(df)
            #pp = initial_pp
            #total column number cannot exceed 31
            pp = np.zeros(int(self.int_bit_width*2)+1)
            for i in range(int(self.int_bit_width*2)+1):
                if self.is_mac:
                    pp[i] = MacPartialProduct[self.bit_width][i]
                else:
                    pp[i] = PartialProduct[self.bit_width][i]

            for i in range(2,int(self.int_bit_width*2)):
                pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

            #change the CT structure
            if action == 0:
                state[1][act_col] = state[1][act_col] + 1
                pp[act_col] = pp[act_col] - 1
                pp[act_col+1] = pp[act_col+1] + 1
            elif action == 1:
                state[1][act_col] = state[1][act_col] - 1
                pp[act_col] = pp[act_col] + 1
                pp[act_col+1] = pp[act_col+1] - 1
            elif action == 2:
                state[1][act_col] = state[1][act_col] + 1
                state[0][act_col] = state[0][act_col] - 1
                pp[act_col] = pp[act_col] + 1
            elif action == 3:
                state[1][act_col] = state[1][act_col] - 1
                state[0][act_col] = state[0][act_col] + 1
                pp[act_col] = pp[act_col] - 1

            #legalization
            # mask 值为1 代表这个动作合法，为0代表不合法；
            for i in range(act_col+1,int(self.int_bit_width*2)+1):
                if (pp[i] == 1 or pp[i] == 2):
                    mask[act_col * 4 + action] = 1
                    break
                #column number restriction
                elif (i == int(self.int_bit_width*2)):
                    mask[act_col * 4 + action] = 1
                    break
                elif (pp[i] == 3):
                    state[0][i] = state[0][i] + 1
                    pp[i+1] = pp[i+1] + 1
                    pp[i] = 1           
                elif (pp[i] == 0):
                    if (state[1][i] >= 1):
                        state[1][i] = state[1][i] - 1
                        pp[i+1] = pp[i+1] -1
                        pp[i] = 1
                    else:
                        state[0][i] = state[0][i] - 1
                        pp[i+1] = pp[i+1] -1
                        pp[i] = 2
        #state = torch.tensor(state)
        index = torch.zeros((int(self.int_bit_width*2))*4)
        mask = (mask!=0)

        for i in range (0,(int(self.int_bit_width*2))*4):
            index[i] = i
        index = torch.masked_select(index, torch.from_numpy(mask))
        df = pd.read_csv(f"./build/state_{self.task_index}.txt", header=None, sep=' ')
        df = df.to_numpy()
        state = torch.tensor(df)
        for action in index:
            next_state = self.transition(np.array(state), action)
            next_state = np.reshape(next_state, (2,int(self.int_bit_width*2)))
            ct32, ct22, pp, stage_num = self.merge(next_state, 0)
            # 真实的最大阶段数为 6
            # 一个 6 个阶段的乘法器
            # len(ct32) = 6
            # stage_num = 5
            # MAX_STAGE_NUM:16:6
            if stage_num >= self.MAX_STAGE_NUM: # 大于 max stage num 的动作也会被mask掉啊，约束这么多
                mask[int(action)] = 0
        mask = (mask!=0)

        # add column mask
        if self.is_column_mask:
            for i in range(len(self.column_mask)):
                if not self.column_mask[i]:
                    mask[i] = 0
            mask = (mask!=0)
        return torch.from_numpy(mask)

    def select_action(self, state, steps_done, deterministic=False, is_softmax=False):
        """
            \epsilon-greedy select action
            inputs: 
                state: dict {"ct32": ct32, "ct22": ct22, "pp": pp, "stage_num": stage_num}
                steps_done
            outputs:
                selected actions
        """
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        ct32, ct22, pp, stage_num = self.merge(state.cpu(), 0)
        info["state_ct32"] = ct32
        info["state_ct22"] = ct22
        # ct32, ct22, pp, stage_num = \
        #     decomposed_state["ct32"], decomposed_state["ct22"], decomposed_state["pp"], decomposed_state["stage_num"]
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        
        state = torch.cat((ct32, ct22), dim=0).float()
        # state = torch.cat((ct32, ct22), dim=0)
        
        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(state.shape)
                mask = self.mask_with_legality(state).to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                q = self(state, state_mask=mask)
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

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
            mask = self.mask_with_legality(state)
            simple_mask = self.mask(state)
            index = torch.zeros((int(self.int_bit_width*2))*4)
            for i in range (0,(int(self.int_bit_width*2))*4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info

    # 状态转移函数
    def transition(self, state, action):
        state = np.reshape(state, (2,int(self.int_bit_width*2)))
        action = int(action)
        act_col = int(action // 4)
        action = int(action % 4)
        #total column number cannot exceed int(self.int_bit_width*2)
        pp = np.zeros(int(self.int_bit_width*2)+1)

        # partial products generated by the booth encoding
        for i in range(int(self.int_bit_width*2)+1):
            if self.is_mac:
                pp[i] = MacPartialProduct[self.bit_width][i]
            else:
                pp[i] = PartialProduct[self.bit_width][i]

        for i in range(1,int(self.int_bit_width*2)): # state 1 - (2self.int_bit_width-1) 有效列；第零列是补齐，方便运算；
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #change the CT structure, 执行动作，更新state记录的compressor 结构，以及partial product，partial product应该是用来legal的
        if action == 0:
            state[1][act_col] = state[1][act_col] + 1
            pp[act_col] = pp[act_col] - 1
            pp[act_col+1] = pp[act_col+1] + 1
        elif action == 1:
            state[1][act_col] = state[1][act_col] - 1
            pp[act_col] = pp[act_col] + 1
            pp[act_col+1] = pp[act_col+1] - 1
        elif action == 2:
            state[1][act_col] = state[1][act_col] + 1
            state[0][act_col] = state[0][act_col] - 1
            pp[act_col] = pp[act_col] + 1
        elif action == 3:
            state[1][act_col] = state[1][act_col] - 1
            state[0][act_col] = state[0][act_col] + 1
            pp[act_col] = pp[act_col] - 1
        #legalization
        # 第i列的动作只会影响第i列和第i+1列的partial product；
        for i in range(act_col+1,int(self.int_bit_width*2)):
            if (pp[i] == 1 or pp[i] == 2):
                break
            elif (pp[i] == 3):
                state[0][i] = state[0][i] + 1
                pp[i+1] = pp[i+1] + 1
                pp[i] = 1           
            elif (pp[i] == 0):
                if (state[1][i] >= 1):
                    state[1][i] = state[1][i] - 1
                    pp[i+1] = pp[i+1] -1
                    pp[i] = 1
                else:
                    state[0][i] = state[0][i] - 1
                    pp[i+1] = pp[i+1] -1
                    pp[i] = 2
        state = np.reshape(state, (1,2,int(self.int_bit_width*2))) # 这里的state 为什么reshape呢？
        return state 

    def merge(self, raw_state, thread_num=0):
        state = np.zeros_like(raw_state)
        state[0] = raw_state[0]
        state[1] = raw_state[1]
        #print(state)
        #merge
        stage_num = 0
        ct32 = np.zeros([1,int(self.int_bit_width*2)])
        ct22 = np.zeros([1,int(self.int_bit_width*2)])
        ct32[0] = state[0]
        ct22[0] = state[1]
        pp = np.zeros([1,int(self.int_bit_width*2)])
        if self.is_mac:
            pp[0] = MacPartialProduct[self.bit_width][:-1]
        else:
            pp[0] = PartialProduct[self.bit_width][:-1]

        for i in range(1,int(self.int_bit_width*2)):
            j = 0
            while(j <= stage_num):
                #print(stage_num)
                ct32[j][i] = state[0][i]
                ct22[j][i] = state[1][i]
                if (j==0):
                    pp[j][i] = pp[j][i]
                else:
                    pp[j][i] = pp[j-1][i] + ct32[j-1][i-1] + ct22[j-1][i-1]
                    
                if ((ct32[j][i]*3 + ct22[j][i]*2) <= pp[j][i]):
                    pp[j][i] = pp[j][i] - ct32[j][i]*2 - ct22[j][i-1]
                    state[0][i] = state[0][i] - ct32[j][i]
                    state[1][i] = state[1][i] - ct22[j][i]
                    break
                else :
                    if(j == stage_num):
                        stage_num += 1
                        ct32 = np.r_[ct32,np.zeros([1,int(self.int_bit_width*2)])]
                        ct22 = np.r_[ct22,np.zeros([1,int(self.int_bit_width*2)])]
                        pp = np.r_[pp,np.zeros([1,int(self.int_bit_width*2)])]
                        
                    if(pp[j][i]%3 == 0):
                        # 3:2 first
                        if (ct32[j][i] >= pp[j][i]//3): 
                            ct32[j][i] = pp[j][i]//3
                            ct22[j][i] = 0
                        else:
                            ct32[j][i] = ct32[j][i]
                            if(ct22[j][i] >= (pp[j][i]-ct32[j][i]*3)//2):
                                ct22[j][i] = (pp[j][i]-ct32[j][i]*3)//2
                            else:
                                ct22[j][i] = ct22[j][i]
                    if(pp[j][i]%3 == 1):
                        # 3:2 first
                        if (ct32[j][i] >= pp[j][i]//3): 
                            ct32[j][i] = pp[j][i]//3
                            ct22[j][i] = 0
                        else:
                            ct32[j][i] = ct32[j][i]
                            if(ct22[j][i] >= (pp[j][i]-ct32[j][i]*3)//2):
                                ct22[j][i] = (pp[j][i]-ct32[j][i]*3)//2
                            else:
                                ct22[j][i] = ct22[j][i]
                    if(pp[j][i]%3 == 2):
                        # 3:2 first
                        if (ct32[j][i] >= pp[j][i]//3): 
                            ct32[j][i] = pp[j][i]//3
                            if (ct22[j][i] >= 1):
                                ct22[j][i] = 1
                        else:
                            ct32[j][i] = ct32[j][i]
                            if(ct22[j][i] >= (pp[j][i]-ct32[j][i]*3)//2):
                                ct22[j][i] = (pp[j][i]-ct32[j][i]*3)//2
                            else:
                                ct22[j][i] = ct22[j][i]
                    pp[j][i] = pp[j][i] - ct32[j][i]*2 - ct22[j][i]
                    state[0][i] = state[0][i] - ct32[j][i]
                    state[1][i] = state[1][i] - ct22[j][i]
                j = j + 1
        sum = ct32.sum() + ct22.sum()
        sum = int(sum)

        #write to file
        file_name = f'./build/ct_test_{self.task_index}' + str(thread_num) + '.txt'
        f = open(file_name, mode = 'w')
        f.write(str(self.str_bit_width) + ' ' + str(self.str_bit_width))
        f.write('\n')
        f.write(str(sum))
        f.write('\n')
        for i in range(0,stage_num+1):
            for j in range(0,int(self.int_bit_width*2)):
                for k in range(0,int(ct32[i][int(self.int_bit_width*2)-1-j])): 
                    f.write(str(int(self.int_bit_width*2)-1-j))
                    f.write(' 1')
                    f.write('\n')
                for k in range(0,int(ct22[i][int(self.int_bit_width*2)-1-j])): 
                    f.write(str(int(self.int_bit_width*2)-1-j))
                    f.write(' 0')
                    f.write('\n')
        return ct32, ct22, pp, stage_num

class ThreeDDeepQPolicy(DeepQPolicy):
    def select_action(self, state, steps_done, deterministic=False, is_softmax=False):
        """
            \epsilon-greedy select action
            inputs: 
                state: dict {"ct32": ct32, "ct22": ct22, "pp": pp, "stage_num": stage_num}
                steps_done
            outputs:
                selected actions
        """
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        #merge()：根据state进行解码，得到3-2压缩器和2-2压缩器的排列
        # set_trace()
        ct32 = state[0:1]
        ct22 = state[1:2]
        stage_num = ct32.shape[0]
        info["state_ct32"] = ct32
        info["state_ct22"] = ct22
        if deterministic:
            eps_threshold = 0.
        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold
        # get pytorch state
        # ct32 = torch.tensor(np.array([ct32]))
        # ct22 = torch.tensor(np.array([ct22]))
        # set_trace()
        state = torch.cat((ct32, ct22), dim=0).float()
        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(state.shape)
                # set_trace()
                mask, cur_state_stage_num = self.mask_with_legality(state)
                mask = mask.to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                # # set_trace()
                q = self(state, state_mask=mask)
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["stage_num"] = cur_state_stage_num
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
            mask, cur_state_stage_num = self.mask_with_legality(state)
            simple_mask = self.mask(state)
            index = torch.zeros((int(self.int_bit_width*2))*self.action_num*self.MAX_STAGE_NUM)
            for i in range (0,(int(self.int_bit_width*2))*self.action_num*self.MAX_STAGE_NUM):
                index[i] = i
            index = torch.masked_select(index, mask)
            info["stage_num"] = cur_state_stage_num
            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0
            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info
                
    def mask_with_legality(self, state):
        #print(state.shape)
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32 = state[0] # stage_num*N
        ct22 = state[1]

        state_np = torch.reshape(state, (2,self.MAX_STAGE_NUM*int(self.int_bit_width*2)))
        state_np = state_np.numpy()
        np.savetxt('./build/state.txt', state_np, fmt="%d", delimiter=" ")
        
        legal_act = []
        # set_trace()
        mask = np.zeros((int(self.int_bit_width*2))*self.action_num*self.MAX_STAGE_NUM)
        # pp 代表当前stage分配完压缩器后剩余的bit数; 问题是最后没有压缩器分配了，这一行还是有可能有1-2bits
        # comments: pp 和 pp1 为什么多初始化一位
        # comments：pp1 多的这一位始终是0，导致后面的约束没用
        pp = np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)+1])
        # pp1 代表当前stage执行压缩树后传递给下一stage的总bit数
        pp1 =  np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)+1]) 
        PP= self.compute_PP(ct32,ct22)
        for i in range(int(self.int_bit_width*2)+1):
            # set_trace()
            pp[0][i] = PartialProduct[self.bit_width][i]
            pp1[0][i] = PartialProduct[self.bit_width][i] 
        #pp1表示当前stage结束后后该列的PP
        cur_state_stage_num = 0
        for k in range(self.MAX_STAGE_NUM): 
            for i in range(0,int(self.int_bit_width*2)):
                # set_trace()
                if k==0:
                    pp[k][i] = pp[k][i]  - ct32[k][i]*3 - ct22[k][i]*2 
                    if i==0:
                        pp1[k][i] = pp1[k][i] - ct32[k][i]*2 - ct22[k][i]
                    else:
                        pp1[k][i] = pp1[k][i] - ct32[k][i]*2 - ct22[k][i] + ct32[k][i-1] + ct22[k][i-1]
                else:
                    if i==0:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i] - ct32[k][i]*3 - ct22[k][i]*2
                        pp1[k][i] = pp1[k-1][i] - ct32[k][i]*2 - ct22[k][i]
                    else:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i]+ ct32[k-1][i-1] + ct22[k-1][i-1] - ct32[k][i]*3 - ct22[k][i]*2
                        pp1[k][i] = pp1[k-1][i] - ct32[k][i]*2 - ct22[k][i] + ct32[k][i-1] + ct22[k][i-1]
        for k in range(self.MAX_STAGE_NUM):
            for i in range(0,int(self.int_bit_width*2)):
                # set_trace()
                if PP[i]==2:
                    if (pp[k][i]>=2):
                        legal_act.append((k,i,0))
                    if (ct22[k][i] >= 1) and pp[k][i]>=1:
                        legal_act.append((k,i,3))
                elif PP[i]==1:
                    if (ct32[k][i]>=1):
                        legal_act.append((k,i,2))
                    if (ct22[k][i]>=1):
                        legal_act.append((k,i,1)) 
            
            tmp=0 
            for i in range(0,int(self.int_bit_width*2)):
                # set_trace()
                if pp1[k][i] not in [1,2]:
                    tmp=1
                    break
            # if tmp==0 or k > 6:
            if tmp==0:
                print(f"current stage: {k} break loop")
                cur_state_stage_num = k
                break
        print(len(legal_act)) 
        for stage,act_col, action in legal_act:
            df = pd.read_csv("./build/state.txt", header=None, sep=' ')
            df = df.to_numpy()
            state = torch.reshape(torch.tensor(df),(2,self.MAX_STAGE_NUM,int(2*self.int_bit_width)))
            ct32 = np.array(state[0]) # stage_num*N
            
            #legalization:一个合法性判断就是该动作实行后，能否通过其他动作最后使得pp最后满足条件，所以每次判断前pp矩阵记得初始化
            action=stage*(self.action_num*int(2*self.int_bit_width))+act_col * self.action_num + action
            if self.is_ilegal(state,action):
                # set_trace()
                mask[action]=1
        
        index = torch.zeros((int(self.int_bit_width*2))*self.action_num*self.MAX_STAGE_NUM)
        mask = (mask!=0)
        for i in range (0,(int(self.int_bit_width*2)*self.action_num*self.MAX_STAGE_NUM)):
            index[i] = i
        index = torch.masked_select(index, torch.from_numpy(mask))
        df = pd.read_csv("./build/state.txt", header=None, sep=' ')
        df = df.to_numpy()
        state = torch.reshape(torch.tensor(df),(2,self.MAX_STAGE_NUM,int(2*self.int_bit_width))) 
        
        return torch.from_numpy(mask), cur_state_stage_num

    def is_ilegal(self,state,action):
        # set_trace()
        state=self.fix_state(state)
        state = np.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32=np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)])
        ct22=np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)])
        ct32 = copy.deepcopy(state[0]) # stage_num*N
        ct22 = copy.deepcopy(state[1])
        action = int(action)

        # set_trace()
        stage= int(action) // (self.action_num*(int(self.int_bit_width*2)))
        act_col = (int(action) % (self.action_num*(int(self.int_bit_width*2)))) // self.action_num
        action= (int(action) % (self.action_num*(int(self.int_bit_width*2))))   %  self.action_num


        pp = np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)+1])
        #print(stage,act_col,action)

        for i in range(int(self.int_bit_width*2)+1):
            # set_trace()
            pp[0][i] = PartialProduct[self.bit_width][i]
        for k in range(self.MAX_STAGE_NUM): 
            # set_trace()
            for i in range(0,int(self.int_bit_width*2)):
                # set_trace()
                if k==0:
                    pp[k][i] = pp[k][i]  - ct32[k][i]*3 - ct22[k][i]*2 
                else:
                    if i==0:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i] - ct32[k][i]*3 - ct22[k][i]*2
                    else:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i]+ ct32[k-1][i-1] + ct22[k-1][i-1] - ct32[k][i]*3 - ct22[k][i]*2
        # set_trace() 
        # 1. executing action to update the ct32 and ct22 matrix and pp
        if action == 0:              
            ct22[stage][act_col]=ct22[stage][act_col]+1
            
            pp[stage][act_col] = pp[stage][act_col] - 2
            for k in range(stage+1,self.MAX_STAGE_NUM):
                pp[k][act_col] = pp[k][act_col] - 1
                pp[k][act_col+1] = pp[k][act_col+1] + 1
        elif action == 1:
            ct22[stage][act_col]=ct22[stage][act_col]-1
            
            pp[stage][act_col] = pp[stage][act_col] + 2
            for k in range(stage+1,self.MAX_STAGE_NUM):
                pp[k][act_col] = pp[k][act_col] + 1
                pp[k][act_col+1] = pp[k][act_col+1] - 1
        elif action == 2:
            ct22[stage][act_col]=ct22[stage][act_col]+1
            ct32[stage][act_col]=ct32[stage][act_col]-1
            
            for k in range(stage,self.MAX_STAGE_NUM):
                pp[k][act_col] = pp[k][act_col] + 1
        elif action == 3:
            ct22[stage][act_col]=ct22[stage][act_col]-1
            ct32[stage][act_col]=ct32[stage][act_col]+1
            
            for k in range(stage,self.MAX_STAGE_NUM):
                pp[k][act_col] = pp[k][act_col] - 1
        elif action == 4 :
            ct32[stage][act_col]=ct32[stage][act_col]+1
            
            pp[stage][act_col] = pp[stage][act_col] - 2
            for k in range(stage+1,self.MAX_STAGE_NUM):
                pp[k][act_col] = pp[k][act_col] - 2
                pp[k][act_col+1] = pp[k][act_col+1] +1

        #legalization:一个合法性判断就是该动作实行后，能否通过其他动作最后使得pp最后满足条件，所以每次判断前pp矩阵记得初始化
        # 基于stage合法化的化，不能从下一列开始，因为可以从该列的其他stage来使得该列合法化。
        # 2. compute the updated final partial product given the updated ct32 and ct22 matrix
        PP=self.compute_PP(ct32,ct22)
        mask=0
        col_legal = 1
        tot_32=np.sum(ct32,axis=0)
        tot_22=np.sum(ct22,axis=0)
        legal_num_column = 0
        # 3. try to legalize the sub compressor tree with stage 
        for i in range(act_col,int(self.int_bit_width*2)):
            # 3.1 first legalize the current column(act_col) via wallace assignment mechanism
            # just update PP and tot_32 and tot_22 
            # then assign these 32 and 22 to act col
            if (PP[i] == 1 or PP[i] == 2):
                mask = 1
                if i != act_col:
                    break
                # 从整体看，然后相应更新stage到MAX STAGE NUM act_col的ct32
                # only when i==act_col, we will execute this code
                # 先检查这一列是否有pp小于0的情况，如果有，则执行合法化，如果没，则合法；
                # for k in range(stage,self.MAX_STAGE_NUM):
                #     if pp[k][i]<0:
                #         col_legal = 0 # 说明这一列不合法，也需要合法化
                # if mask == 0:
                    # mask 为0说明这一列存在不合法的可能性，需要尝试合法化；
                    
                    # (0,stage-1)不动，

                # v1: legal by remove 32 or remove 22 (comments: this may lead to too large stage)
                for k in range(stage,self.MAX_STAGE_NUM):
                    if pp[k][i]<0:
                        mask = 0
                        if ct32[k][i] >= 1 :
                            ct32[k][i] = ct32 [k][i] -1 
                            ct22[k][i] = ct22 [k][i] +1 
                            tot_32[i] = tot_32[i]-1
                            tot_22[i] = tot_22[i]+1
                            PP[i]=2
                            mask=1
                            for j in range(k,self.MAX_STAGE_NUM):
                                pp[j][i] = pp[j][i] + 1  
                        # v7: no remove 22 legal
                        elif ct22[k][i] >=1 :   
                            ct22[k][i] = ct22 [k][i] - 1
                            tot_22[i] = tot_22[i]-1
                            PP[i]=2
                            PP[i+1] -= 1
                            mask=1
                            pp[k][i] = pp[k][i] + 2
                            for j in range(k+1,self.MAX_STAGE_NUM):
                                pp[j][i] = pp[j][i] + 1
                                pp[j][i+1] = pp[j][i+1] - 1
                        else:
                            break  
                            
                if mask==0:
                    return mask
            elif i==int(self.int_bit_width*2):
                mask=1
                break
            elif PP[i]==3:
                # add a 32 at column i
                PP[i]=1
                PP[i+1]=PP[i+1]+1
                tot_32[i]=tot_32[i]+1
            elif PP[i]==0:
                # remove a 22 at column i
                if (tot_22[i]>=1):
                    tot_22[i]=tot_22[i]-1
                    PP[i]=1
                    PP[i+1]=PP[i+1]-1
                # remove a 32 at column i
                elif (tot_32[i]>=1):
                    tot_32[i]=tot_32[i]-1
                    PP[i+1]=PP[i+1]-1
                    PP[i]=2
                else:
                    print("warning!!! PP=0 no valid action")
                    break
            legal_num_column += 1
            if legal_num_column >= self.legal_num_column:
                mask = 0
                break
            # # set_trace()             
        if mask==0:
            return mask
        
        if col_legal == 1:
            # col_legal 为1 说明act_col列是合法的，不再需要合法化；
            legal_col=act_col+1
        elif col_legal == 0:
            # 否则是需要合法化的
            legal_col=act_col
        
        # legal_stage=stage+1 # v4           
        # legal_stage=stage # v1
        legal_stage=0 # v3
        # # set_trace()
        ct32[legal_stage:,legal_col:]=0 # 应不应该从 stage 开始算合法化呢？
        ct22[legal_stage:,legal_col:]=0

        legal_PP=np.zeros((int(self.int_bit_width*2)))

        sum_32=np.sum(ct32[0:legal_stage,:],axis=0)
        sum_22=np.sum(ct22[0:legal_stage,:],axis=0)
        # set_trace()
        for j in range(legal_col,int(self.int_bit_width*2)):
            legal_PP[j]=PartialProduct[self.bit_width][j]-sum_32[j]*2-sum_22[j]+sum_22[j-1]+sum_32[j-1]
            
        # comments: 这个合法化感觉有问题
        # comments: 不应该是这样合法化呀，
        # 应该是把子的partial product 和 子state记录一下，然后对子state matrix state进行整体合法化调整，如果能调整到即可；然后对其分解得到新的ct32/22
        # 目前这段是按照Wallace tree 重新分配了，那很糟糕；初衷不是这个；
        left_32=tot_32-sum_32
        left_22=tot_22-sum_22
        for j in range(legal_col,int(self.int_bit_width*2)):
            for k in range(legal_stage,self.MAX_STAGE_NUM):
                if k>legal_stage:
                    legal_PP[j]=legal_PP[j]+ct32[k-1][j-1]+ct22[k-1][j-1]
                
                if left_32[j]>= legal_PP[j] // 3:
                    ct32[k][j] = legal_PP[j] // 3
                    if (legal_PP[j] % 3 == 2) and left_22[j]>=1:
                        ct22[k][j] = 1
                else:
                    ct32[k][j]=left_32[j]
                    if (left_22[j]>=(legal_PP[j]-ct32[k][j]*3)//2):
                        ct22[k][j] = (legal_PP[j]-ct32[k][j]*3) //2
                    else:
                        ct22[k][j] = left_22[j]
                
                left_32[j]=left_32[j]-ct32[k][j]
                left_22[j]=left_22[j]-ct22[k][j]        
                legal_PP[j]=legal_PP[j] - 2*ct32[k][j] -ct22[k][j]
            legal_PP[j]=legal_PP[j] + ct32[self.MAX_STAGE_NUM-1][j-1]+ct22[self.MAX_STAGE_NUM-1][j-1]
            if legal_PP[j] in [1,2]:
                mask=1
            else:
                mask=0  
                break
        return mask

    def fix_state(self,state):
        state=np.array(state)
        ct32 = np.array([state[0]])
        ct22 = np.array([state[1]])
        stage_num=ct22.shape[1]
        # set_trace()
        if stage_num < self.MAX_STAGE_NUM: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros),axis=1)
            ct22 = np.concatenate((ct22, zeros),axis=1)
        state =np.concatenate((ct32, ct22),axis=0)
        return state

    def compute_PP(self,ct32,ct22):
        # set_trace()
        ct32=np.array(ct32)
        ct22=np.array(ct22)
        #print(ct32)
        pp = np.zeros(int(self.int_bit_width*2)+1)
        for i in range(int(self.int_bit_width*2)+1):
            pp[i] = PartialProduct[self.bit_width][i]
        ct32=np.sum(ct32,axis=0)
        ct22=np.sum(ct22,axis=0)
        #print(pp,ct32,ct22)
        pp[0]=pp[0]-2*ct32[0]-ct22[0]
        for i in range(1,int(self.int_bit_width*2)):
            pp[i]=pp[i]-2*ct32[i]-ct22[i]+ct22[i-1]+ct32[i-1]
        return pp

    def compute_pp(self,ct32,ct22):
        ct32=np.array(ct32)
        ct22=np.array(ct22)
        pp = np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)+1])
        for i in range(int(self.int_bit_width*2)+1):
            pp[0][i] = PartialProduct[self.bit_width][i]
        for k in range(self.MAX_STAGE_NUM):
            for i in range(0,int(self.int_bit_width*2)):
                if k==0:
                    pp[k][i] = pp[k][i]  - ct32[k][i]*3 - ct22[k][i]*2 
                else:
                    if i==0:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i] - ct32[k][i]*3 - ct22[k][i]*2
                    else:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i]+ ct32[k-1][i-1] + ct22[k-1][i-1] - ct32[k][i]*3 - ct22[k][i]*2
        return pp

    def mask(self, state):
        #tensor T  K*2N*ST
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        ct32 = state[0]
        ct22 = state[1]

        #state = torch.cat((ct32, ct22), dim=0)
        state_np = torch.reshape(state, (2,self.MAX_STAGE_NUM*int(self.int_bit_width*2)))
        state_np = state_np.numpy()
        np.savetxt('./build/state.txt', state_np, fmt="%d", delimiter=" ")
        
        legal_act = []
        ## mask 2n*4*stage
        mask = np.zeros((int(self.int_bit_width*2))*self.action_num*self.MAX_STAGE_NUM)
        #initial_state = state
        pp = np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)+1])
        pp1 =  np.zeros([self.MAX_STAGE_NUM,int(self.int_bit_width*2)+1])
        PP= self.compute_PP(ct32,ct22)
        for i in range(int(self.int_bit_width*2)+1):
                pp[0][i] = PartialProduct[self.bit_width][i]
                pp1[0][i] = PartialProduct[self.bit_width][i] 
        # 错误思路：直接计算当前列的部分积，但这样直接计算压缩后的结果是不对的，因为进位和输出是下一阶段才能考虑的
        # 所以当前阶段的压缩是在上一阶段的部分积上进行的。
        
        for k in range(self.MAX_STAGE_NUM): 
            for i in range(0,int(self.int_bit_width*2)):
                if k==0:
                    pp[k][i] = pp[k][i]  - ct32[k][i]*3 - ct22[k][i]*2 
                    if i==0:
                        pp1[k][i] = pp1[k][i] - ct32[k][i]*2 - ct22[k][i]
                    else:
                        pp1[k][i] = pp1[k][i] - ct32[k][i]*2 - ct22[k][i] + ct32[k][i-1] + ct22[k][i-1]
                else:
                    if i==0:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i] - ct32[k][i]*3 - ct22[k][i]*2
                        pp1[k][i] = pp1[k-1][i] - ct32[k][i]*2 - ct22[k][i]
                    else:
                        pp[k][i] = pp[k-1][i] + ct32[k-1][i] + ct22[k-1][i]+ ct32[k-1][i-1] + ct22[k-1][i-1] - ct32[k][i]*3 - ct22[k][i]*2
                        pp1[k][i] = pp1[k-1][i] - ct32[k][i]*2 - ct22[k][i] + ct32[k][i-1] + ct22[k][i-1]
        for k in range(self.MAX_STAGE_NUM):
            for i in range(0,int(self.int_bit_width*2)):
                
                if PP[i]==2:
                    if (pp[k][i]>=2):
                        legal_act.append((k,i,0))
                    if (ct22[k][i] >= 1) and pp[k][i]>=1:
                        legal_act.append((k,i,3))
                elif PP[i]==1:
                    if (ct32[k][i]>=1):
                        legal_act.append((k,i,2))
                    if (ct22[k][i]>=1):
                        legal_act.append((k,i,1)) 
            
            tmp=0 
            for i in range(0,int(self.int_bit_width*2)):
                if pp1[k][i] not in [1,2]:
                    tmp=1
                    break
            if tmp==0:
                break

        for stage,act_col, action in legal_act:
            mask[stage*self.action_num*int(self.int_bit_width*2) + act_col * self.action_num + action] = 1
        mask = (mask!=0)
        return torch.from_numpy(mask)

class ThreeDFactorDeepQPolicy(ThreeDDeepQPolicy):
    def __init__(
        self, block, **policy_kwargs
    ):
        super(ThreeDFactorDeepQPolicy, self).__init__(
            block, is_factor=True, **policy_kwargs
        )
        assert self.is_rnd_predictor != True
        self.num_action_stage = self.MAX_STAGE_NUM
        self.num_action_column = int(self.num_classes / (self.action_num * self.num_action_stage))
        self.num_action_type = self.action_num
        self.fc_stage = nn.Linear(512 * block.expansion, self.num_action_stage)
        self.fc_column = nn.Linear(512 * block.expansion, self.num_action_column)
        self.fc_type = nn.Linear(512 * block.expansion, self.num_action_type)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def _combine(self, output_stage, output_column, output_type):
        # each stage (num_col * num_action)

        batch_size = output_column.shape[0]
        num_classes = output_stage.shape[1] * output_column.shape[1] * output_type.shape[1]

        output = torch.zeros(
            (batch_size, num_classes),
            dtype=torch.float,
            device=output_column.device
        )
        for k in range(output_stage.shape[1]):
            for i in range(output_column.shape[1]):
                for j in range(output_type.shape[1]):
                    # k*self.action_num*self.num_action_column + i*self.action_num + j
                    cur_index = k*self.action_num*self.num_action_column + i*self.action_num + j
                    output[:,cur_index] = output_stage[:,k] + output_column[:,i] + output_type[:,j]
        return output
    
    def forward(self, x, is_target=False, state_mask=None):
        x = x.to(self.device)
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
        output_stage = self.fc_stage(output)
        output_column = self.fc_column(output) # (batch_size, num_column)
        output_type = self.fc_type(output) # (batch_size, num_type)
        output = self._combine(output_stage, output_column, output_type)
        output = output.masked_fill(~mask.to(self.device),-1000)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(x, dim=-1)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return output

class MultiObjConditionedDeepQPolicy(DeepQPolicy):
    def __init__(
        self, block, model_hidden_dim=256, wallace_area=4, wallace_delay=1, 
        condition_input_num=2, **policy_kwargs
    ):
        super(MultiObjConditionedDeepQPolicy, self).__init__(
            block, **policy_kwargs
        )
        assert self.is_rnd_predictor != True
        self.model_hidden_dim = model_hidden_dim
        self.wallace_area = wallace_area
        self.wallace_delay = wallace_delay
        self.condition_input_num = condition_input_num
        # first encode the embedding
        self.fully_connected_layer = nn.Sequential(
                nn.Linear(512 * block.expansion, self.model_hidden_dim),
                nn.ReLU()
            )
        # area model 
        self.area = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_classes)
        )
        # delay model
        self.delay = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_classes)
        )
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x, weight_condition, is_target=False, state_mask=None):
        x = x.to(self.device)
        weight_condition = weight_condition.to(self.device)
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
        # base encoder
        output = self.fully_connected_layer(output)
        # concat with condition
        conditioned_input = torch.cat(
            [output, weight_condition], dim=1
        )
        # get area delay q values
        area_output = self.area(conditioned_input)
        delay_output = self.delay(conditioned_input)
        output = self.wallace_area * area_output + self.wallace_delay * delay_output

        area_output = area_output.masked_fill(~mask.to(self.device),-1000)
        delay_output = delay_output.masked_fill(~mask.to(self.device),-1000)
        output = output.masked_fill(~mask.to(self.device),-1000)

        return area_output, delay_output, output

    def select_action(self, state, steps_done, task_weight_vector, deterministic=False, is_softmax=False):
        """
            \epsilon-greedy select action
            inputs: 
                state: dict {"ct32": ct32, "ct22": ct22, "pp": pp, "stage_num": stage_num}
                steps_done
            outputs:
                selected actions
        """
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        ct32, ct22, pp, stage_num = self.merge(state.cpu(), 0)
        info["state_ct32"] = ct32
        info["state_ct22"] = ct22
        # ct32, ct22, pp, stage_num = \
        #     decomposed_state["ct32"], decomposed_state["ct22"], decomposed_state["pp"], decomposed_state["stage_num"]
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        
        state = torch.cat((ct32, ct22), dim=0).float()
        weight_condition = torch.tensor(task_weight_vector).float().unsqueeze(0)
        
        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                mask = self.mask_with_legality(state).to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                q_area, q_delay, q = self(state, weight_condition, state_mask=mask)
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q
                info["q_area"] = q_area
                info["q_delay"] = q_delay

                if is_softmax:
                    q_distribution = Categorical(logits=q)
                    action = q_distribution.sample()
                else:
                    action = q.max(1)[1]
                return action.view(1, 1), info
        else:
            mask = self.mask_with_legality(state)
            simple_mask = self.mask(state)
            index = torch.zeros((int(self.int_bit_width*2))*4)
            for i in range (0,(int(self.int_bit_width*2))*4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0
            info["q_area"] = 0
            info["q_delay"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info

class MultiObjFactorDeepQPolicy(DeepQPolicy):
    def __init__(
        self, block, model_hidden_dim=256, wallace_area=4, wallace_delay=1, **policy_kwargs
    ):
        super(MultiObjFactorDeepQPolicy, self).__init__(
            block, is_factor=True, **policy_kwargs
        )
        assert self.is_rnd_predictor != True
        num_action_column = int(self.num_classes / 4)
        num_action_type = 4
        self.wallace_area = wallace_area
        self.wallace_delay = wallace_delay
        # area model 
        self.area_column = nn.Sequential(
                nn.Linear(512 * block.expansion, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, num_action_column)
        )
        self.area_type = nn.Sequential(
                nn.Linear(512 * block.expansion, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, num_action_type)
        )
        
        # delay model
        self.delay_column = nn.Sequential(
                nn.Linear(512 * block.expansion, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, num_action_column)
        )
        self.delay_type = nn.Sequential(
                nn.Linear(512 * block.expansion, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, num_action_type)
        )

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def _combine(self, output_column, output_type):
        batch_size = output_column.shape[0]
        num_classes = output_column.shape[1] * output_type.shape[1]
        output = torch.zeros(
            (batch_size, num_classes),
            dtype=torch.float,
            device=output_column.device
        )
        for i in range(output_column.shape[1]):
            for j in range(output_type.shape[1]):
                output[:,i*4+j] = output_column[:,i] + output_type[:,j]
        return output
    
    def forward(self, x, is_target=False, state_mask=None):
        x = x.to(self.device)
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

        area_output_column = self.area_column(output)
        area_output_type = self.area_type(output)
        delay_output_column = self.delay_column(output)
        delay_output_type = self.delay_type(output)

        area_output = self._combine(area_output_column, area_output_type)
        delay_output = self._combine(delay_output_column, delay_output_type)
        output = self.wallace_area * area_output + self.wallace_delay * delay_output
        area_output = area_output.masked_fill(~mask.to(self.device),-1000)
        delay_output = delay_output.masked_fill(~mask.to(self.device),-1000)
        output = output.masked_fill(~mask.to(self.device),-1000)
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(x, dim=-1)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return area_output, delay_output, output

    def select_action(self, state, steps_done, deterministic=False, is_softmax=False):
        """
            \epsilon-greedy select action
            inputs: 
                state: dict {"ct32": ct32, "ct22": ct22, "pp": pp, "stage_num": stage_num}
                steps_done
            outputs:
                selected actions
        """
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        ct32, ct22, pp, stage_num = self.merge(state.cpu(), 0)
        info["state_ct32"] = ct32
        info["state_ct22"] = ct22
        # ct32, ct22, pp, stage_num = \
        #     decomposed_state["ct32"], decomposed_state["ct22"], decomposed_state["pp"], decomposed_state["stage_num"]
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        
        state = torch.cat((ct32, ct22), dim=0).float()
        # state = torch.cat((ct32, ct22), dim=0)
        
        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(state.shape)
                mask = self.mask_with_legality(state).to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                q_area, q_delay, q = self(state, state_mask=mask)
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q
                info["q_area"] = q_area
                info["q_delay"] = q_delay

                if is_softmax:
                    q_distribution = Categorical(logits=q)
                    action = q_distribution.sample()
                else:
                    action = q.max(1)[1]
                return action.view(1, 1), info
        else:
            mask = self.mask_with_legality(state)
            simple_mask = self.mask(state)
            index = torch.zeros((int(self.int_bit_width*2))*4)
            for i in range (0,(int(self.int_bit_width*2))*4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0
            info["q_area"] = 0
            info["q_delay"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info

class FactorDeepQPolicy(DeepQPolicy):
    def __init__(
        self, block, **policy_kwargs
    ):
        super(FactorDeepQPolicy, self).__init__(
            block, is_factor=True, **policy_kwargs
        )
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
            (batch_size, num_classes),
            dtype=torch.float,
            device=output_column.device
        )
        for i in range(output_column.shape[1]):
            for j in range(output_type.shape[1]):
                output[:,i*4+j] = output_column[:,i] + output_type[:,j]
        return output
    
    def forward(self, x, is_target=False, state_mask=None):
        x = x.to(self.device)
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
        output_column = self.fc_column(output) # (batch_size, num_column)
        output_type = self.fc_type(output) # (batch_size, num_type)
        output = self._combine(output_column, output_type)
        output = output.masked_fill(~mask.to(self.device),-1000)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(x, dim=-1)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return output

class MBRLFactorDeepQPolicy(FactorDeepQPolicy):
    def __init__(
        self, block, model_hidden_dim=256, **policy_kwargs
    ):
        super(MBRLFactorDeepQPolicy, self).__init__(
            block, **policy_kwargs
        )

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

    def area_forward(self, x):
        x = x.to(self.device)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.area_model(output)
        return output

    def delay_forward(self, x):
        x = x.to(self.device)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.delay_model(output)
        return output

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

class ColumnQPolicy(DeepQPolicy):
    def get_column_mask(self, mask):
        number_column = int(len(mask)/4)
        column_mask = np.ones(number_column)
        for i in range(number_column):
            cur_column_mask = mask[4*i:4*(i+1)]
            if torch.sum(cur_column_mask) == 0.:
                # mask i-th column
                column_mask[i] = 0

        return torch.from_numpy(column_mask)

    def forward(self, x, is_target=False, state_mask=None, is_mask_inf=False):
        x = x.to(self.device)
        if state_mask is not None:
            mask = state_mask
        else:
            if is_target:
                mask = self.mask_with_legality(x)
            else:
                mask = self.mask(x)
        # get column mask
        column_mask = self.get_column_mask(mask)
        column_mask = (column_mask!=0)

        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # mask output
        if is_mask_inf:
            neg_inf = torch.tensor(float('-inf'), device=self.device)
            output = output.masked_fill(~column_mask.to(self.device),neg_inf)
        else:
            output = output.masked_fill(~column_mask.to(self.device),-1000)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        #action_prob = F.softmax(x, dim=-1)


        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return output 

    def decode_action(self, action_column, mask):
        index = torch.zeros(4)
        for i in range(4):
            index[i] = i
        index = torch.masked_select(index, mask[4*action_column:4*(action_column+1)])
        # retuan an action list
        action = [int(4*action_column + ind) for ind in index]
        # action = int(random.choice(index)) + 4*action_column
        num_valid_action = torch.sum(mask[4*action_column:4*(action_column+1)])
        return action, num_valid_action

    def select_action(self, state, steps_done, deterministic=False, is_softmax=False):
        """
            \epsilon-greedy select action
            inputs: 
                state: dict {"ct32": ct32, "ct22": ct22, "pp": pp, "stage_num": stage_num}
                steps_done
            outputs:
                selected actions
        """
        info = {}
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        ct32, ct22, pp, stage_num = self.merge(state.cpu(), 0)
        info["state_ct32"] = ct32
        info["state_ct22"] = ct22
        # ct32, ct22, pp, stage_num = \
        #     decomposed_state["ct32"], decomposed_state["ct22"], decomposed_state["pp"], decomposed_state["stage_num"]
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        
        state = torch.cat((ct32, ct22), dim=0).float()
        # state = torch.cat((ct32, ct22), dim=0)
        
        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(state.shape)
                mask = self.mask_with_legality(state).to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                q = self(state, state_mask=mask, is_mask_inf=True)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q

                if is_softmax:
                    q_distribution = Categorical(logits=q)
                    action_column = q_distribution.sample()
                else:
                    action_column = q.max(1)[1]
                action, num_valid_action = self.decode_action(action_column, mask.cpu())
                info["num_valid_action"] = num_valid_action
                print(f"debug log action: {action}")
                return torch.tensor([[action]], device=self.device, dtype=torch.long), action_column.view(1, 1), info
        else:
            mask = self.mask_with_legality(state)
            simple_mask = self.mask(state)
            index = torch.zeros((int(self.int_bit_width*2))*4)
            for i in range (0,(int(self.int_bit_width*2))*4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0
            
            action = int(random.choice(index))
            action_column = int(action / 4)
            num_valid_action = torch.sum(mask[4*action_column:4*(action_column+1)])
            info["num_valid_action"] = num_valid_action
            return torch.tensor([[action]], device=self.device, dtype=torch.long), torch.tensor([[action_column]], device=self.device, dtype=torch.long), info

if __name__ == "__main__":
    from o1_environment import RefineEnv, ThreeDRefineEnv
    import random
    import numpy as np
    import torch
    import time 
    import os

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set CuDNN to be deterministic. Notice that this may slow down the training.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    
    device = "cuda:0"
    q_policy = ThreeDDeepQPolicy(
        BasicBlock, pool_is_stochastic=True, device=device, num_classes=16*4*4
    ).to(device)
    env = ThreeDRefineEnv(1, q_policy, initial_state_pool_max_len=20)
    state, _ = env.reset()
    print(state)
    # set_trace()
    st = time.time()
    for i in range(2):
        action, info = q_policy.select_action(torch.tensor(state), 1, deterministic=True)
        q_value = info["q_value"]
        print(f"action: {action}, stage num {info['stage_num']}, q value {q_value}")
        # q_policy.partially_reset()
    et = time.time() - st
    print(f"time: {et}")
