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
        self, block, num_block=[2, 2, 2, 2], 
        num_classes=(8*2)*4, EPS_START = 0.9, legal_num_column=200,
        EPS_END = 0.10, EPS_DECAY = 500, MAX_STAGE_NUM=4, task_index=0,
        bit_width='8_bits_booth', width=8, num=8, action_num=4, device='cpu',
        is_rnd_predictor=False, pool_is_stochastic=True, is_column_mask=False, is_factor=False
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
        self.width = width
        self.num = num
        self.action_num = action_num
        self.device = device
        self.is_rnd_predictor = is_rnd_predictor
        self.pool_is_stochastic = pool_is_stochastic
        self.num_classes = num_classes
        self.is_factor = is_factor
        self.legal_num_column = legal_num_column
        self.task_index = task_index

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
        elif not self.is_factor:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.is_column_mask = is_column_mask
        if is_column_mask:
            num_column_mask = int(self.width*4)
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
        
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.width)))
        #state = torch.reshape(state, (2,6,int(self.width)))
        ct32 = state[0]
        ct22 = state[1]
        ct32 = ct32.sum(axis=0).unsqueeze(0)
        ct22 = ct22.sum(axis=0).unsqueeze(0)
        state = torch.cat((ct32, ct22), dim=0)
        #state = torch.reshape(state, (2,int(self.width)))
        state_np = state.cpu().numpy()
        np.savetxt(f'./build/state_{self.task_index}.txt', state_np, fmt="%d", delimiter=" ")
        legal_act = []
        mask = np.zeros((int(self.width))*4)
        #initial_state = state
        pp = np.zeros(int(self.width))
        PartialProduct = np.full(self.width,self.num)
        for i in range(int(self.width)):
            pp[i] = PartialProduct[i]

        for i in range(2,int(self.width)):
            pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]

        #initial_pp = pp
        for i in range(2,int(self.width)):
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
        state = torch.reshape(state, (2,self.MAX_STAGE_NUM,int(self.width)))
        ct32 = state[0]
        ct22 = state[1]
        ct32 = ct32.sum(axis=0).unsqueeze(0)
        ct22 = ct22.sum(axis=0).unsqueeze(0)
        state = torch.cat((ct32, ct22), dim=0)

        #state = torch.reshape(state, (2,int(self.width)))
        try:
            state_np = state.numpy()
        except:
            state_np = state.cpu().numpy()
        np.savetxt(f'./build/state_{self.task_index}.txt', state_np, fmt="%d", delimiter=" ")
        legal_act = []
        mask = np.zeros((int(self.width))*4)
        #initial_state = state
        pp = np.zeros(int(self.width)+1)
        PartialProduct = np.full(self.width,self.num)
  
        for i in range(int(self.width)+1):
            if i != self.width:
                pp[i] = PartialProduct[i]
            else:
                pp[i]=0

        for i in range(int(self.width)):
            if i==0:
                pp[i] = pp[i] - state[0][i]*2 - state[1][i]
            else:
                pp[i] = pp[i] + state[0][i-1] + state[1][i-1] - state[0][i]*2 - state[1][i]
        #initial_pp = pp
        for i in range(int(self.width)):
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
            pp = np.zeros(int(self.width)+1)

            PartialProduct = np.full(self.width,self.num)
            for i in range(int(self.width)+1):
                if i!=self.width:
                    pp[i] = PartialProduct[i]
                else:
                    pp[i] = 0
            
            for i in range(int(self.width)):
                if i==0:
                    pp[i] = pp[i]  - state[0][i]*2 - state[1][i]
                else:
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
            for i in range(act_col+1,int(self.width)+1):
                if (pp[i] == 1 or pp[i] == 2):
                    mask[act_col * 4 + action] = 1
                    break
                #column number restriction
                elif (i == int(self.width)):
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
        index = torch.zeros((int(self.width))*4)
        mask = (mask!=0)
        for i in range (0,(int(self.width))*4):
            index[i] = i
        index = torch.masked_select(index, torch.from_numpy(mask))
        df = pd.read_csv(f"./build/state_{self.task_index}.txt", header=None, sep=' ')
        df = df.to_numpy()
        state = torch.tensor(df)
        for action in index:
            next_state = self.transition(np.array(state), action)
            next_state = np.reshape(next_state, (2,int(self.width)))
            ct32, ct22, pp, stage_num = self.merge(next_state, 0)
            if stage_num >= self.MAX_STAGE_NUM: # 大于 max stage num 的动作也会被mask掉啊，约束这么多
                mask[int(action)] = 0
        mask = (mask!=0)
        # add column mask
        if self.is_column_mask:
            for i in range(len(self.column_mask)):
                if not self.column_mask[i]:
                    mask[i] = 0
            mask = (mask!=0)
        #sprint(mask)
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
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.width))
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
            index = torch.zeros((int(self.width))*4)
            for i in range (0,(int(self.width))*4):
                index[i] = i
            index = torch.masked_select(index, mask)

            info["mask"] = mask
            info["simple_mask"] = simple_mask
            info["q_value"] = 0

            return torch.tensor([[int(random.choice(index))]], device=self.device, dtype=torch.long), info

    # 状态转移函数
    def transition(self, state, action):
        state = np.reshape(state, (2,int(self.width)))
        action = int(action)
        act_col = int(action // 4)
        action = int(action % 4)
        #total column number cannot exceed int(self.width)
        pp = np.zeros(int(self.width)+1)

        # partial products generated by the booth encoding

        PartialProduct = np.full(self.width,self.num)
        for i in range(int(self.width)+1):
            if i != self.width:
                pp[i] = PartialProduct[i]
            else:
                pp[i]=0

        for i in range(1,int(self.width)): # state 1 - (2self.width-1) 有效列；第零列是补齐，方便运算；
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
        for i in range(act_col+1,int(self.width)):
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
        state = np.reshape(state, (1,2,int(self.width))) # 这里的state 为什么reshape呢？
        return state 

    def merge(self, raw_state, thread_num=0):
        state = np.zeros_like(raw_state)
        state[0] = raw_state[0]
        state[1] = raw_state[1]
        #print(state)
        #merge
        stage_num = 0
        ct32 = np.zeros([1,int(self.width)])
        ct22 = np.zeros([1,int(self.width)])
        ct32[0] = state[0]
        ct22[0] = state[1]

        PartialProduct = np.full(self.width,self.num)
        pp = np.zeros([1,int(self.width)])
        for i in range(int(self.width)):
            pp[0][i] = PartialProduct[i]


        for i in range(int(self.width)):
            j = 0
            while(j <= stage_num):
                #print(stage_num)
                ct32[j][i] = state[0][i]
                ct22[j][i] = state[1][i]
                if (j==0):
                    pp[j][i] = pp[j][i]
                else:
                    if i== 0:
                        pp[j][i] = pp[j-1][i] 
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
                        ct32 = np.r_[ct32,np.zeros([1,int(self.width)])]
                        ct22 = np.r_[ct22,np.zeros([1,int(self.width)])]
                        pp = np.r_[pp,np.zeros([1,int(self.width)])]
                        
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
        f.write(str(self.width) + ' ' + str(self.width))
        f.write('\n')
        f.write(str(sum))
        f.write('\n')
        for i in range(0,stage_num+1):
            for j in range(0,int(self.width)):
                for k in range(0,int(ct32[i][int(self.width)-1-j])): 
                    f.write(str(int(self.width)-1-j))
                    f.write(' 1')
                    f.write('\n')
                for k in range(0,int(ct22[i][int(self.width)-1-j])): 
                    f.write(str(int(self.width)-1-j))
                    f.write(' 0')
                    f.write('\n')
        return ct32, ct22, pp, stage_num

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
            zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.width))
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
            index = torch.zeros((int(self.width))*4)
            for i in range (0,(int(self.width))*4):
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
