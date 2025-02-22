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

from o2_policy import DeepQPolicy, BasicBlock
from utils.mlp import MLP
from ipdb import set_trace

class SigmoidMLP(nn.Module):
    def __init__(
            self,
            input_dim=8,
            output_dim=8,
            hidden_sizes=[128,128]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

        self.mlp = MLP(
            input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits.squeeze()
    
class SoftmaxMLP(nn.Module):
    def __init__(
            self,
            input_dim=8,
            output_dim=12,
            hidden_sizes=[64,64]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

        self.mlp = MLP(
            input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.mlp(x)
        probs = self.softmax(logits)

        return probs, logits

class MultiTaskFactorDeepQPolicy(DeepQPolicy):
    def __init__(
        self, block, model_hidden_dim=256, is_non_linear=True,
        task_weight_vectors=[[4,1],[3,2],[2,3],[1,4]], 
        is_condition_vector=False, **policy_kwargs
    ):
        super(MultiTaskFactorDeepQPolicy, self).__init__(
            block, is_factor=True, **policy_kwargs
        )
        assert self.is_rnd_predictor != True
        self.num_action_column = int(self.num_classes / 4)
        self.num_action_type = 4
        self.model_hidden_dim = model_hidden_dim
        self.task_weight_vectors = task_weight_vectors
        self.num_tasks = len(task_weight_vectors)
        self.is_non_linear = is_non_linear
        
        if not is_condition_vector:
            self.fc_column = nn.ModuleList()
            self.fc_type = nn.ModuleList()
            for _ in range(self.num_tasks):
                if self.is_non_linear:
                    fc_column = nn.Sequential(
                        nn.Linear(512 * block.expansion, self.model_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.model_hidden_dim, self.num_action_column)
                    )
                else:
                    fc_column = nn.Linear(512 * block.expansion, self.num_action_column)
                self.fc_column.append(fc_column)
                if self.is_non_linear:
                    fc_type = nn.Sequential(
                        nn.Linear(512 * block.expansion, self.model_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.model_hidden_dim, self.num_action_type)
                    )
                else:
                    fc_type = nn.Linear(512 * block.expansion, self.num_action_type)
                self.fc_type.append(fc_type)

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
        # 输入state，输出 multi-task agent, 输出一个list的q value
        output_list = []
        x = x.to(self.device)
        if state_mask is not None:
            mask = state_mask
        else:
            if is_target:
                mask = self.mask_with_legality(x)
            else:
                mask = self.mask(x)
        # resnet encoder
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        # multi-task factor q value output
        for i in range(self.num_tasks):
            output_column = self.fc_column[i](output)
            output_type = self.fc_type[i](output)
            output_i = self._combine(output_column, output_type)
            output_i = output_i.masked_fill(~mask.to(self.device),-1000)
            output_list.append(output_i)
        return output_list

    def select_action(self, state, steps_done, task_index, deterministic=False, is_softmax=False):
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
                q_list = self(state, state_mask=mask)
                q = q_list[task_index]
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q
                info["task_index"] = task_index
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

"""
    class MultiTaskVectorFactorDeepQPolicy
"""
class MultiTaskVectorFactorConditionDeepQPolicy(MultiTaskFactorDeepQPolicy):
    def __init__(
        self, block, model_hidden_dim=256, is_non_linear=True,
        task_weight_vectors=[[4,1],[3,2],[2,3],[1,4]], 
        condition_input_num=3, **policy_kwargs
    ):
        super(MultiTaskVectorFactorConditionDeepQPolicy, self).__init__(
            block, model_hidden_dim=model_hidden_dim, is_non_linear=is_non_linear,
            task_weight_vectors=task_weight_vectors, is_condition_vector=True, **policy_kwargs
        )
        self.condition_input_num = condition_input_num
        # first encode the embedding
        # self.fully_connected_layer = nn.Sequential(
        #         nn.Linear(512 * block.expansion, self.model_hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(self.model_hidden_dim, self.model_hidden_dim)
        #     )
        self.fully_connected_layer = nn.Sequential(
                nn.Linear(512 * block.expansion, self.model_hidden_dim)
            )
        
        self.fc_column_area = nn.ModuleList()
        self.fc_column_delay = nn.ModuleList()
        self.fc_type_area = nn.ModuleList()
        self.fc_type_delay = nn.ModuleList()
        for _ in range(self.num_tasks):
            # column q area delay
            fc_column_area = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_column)
            )
            fc_column_delay = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_column)
            )
            self.fc_column_area.append(fc_column_area)
            self.fc_column_delay.append(fc_column_delay)
            
            # type q area delay
            fc_type_area = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_type)
            )
            fc_type_delay = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_type)
            )
            self.fc_type_area.append(fc_type_area)
            self.fc_type_delay.append(fc_type_delay)

    def forward(self, x, weight_condition, delay_condition, is_target=False, state_mask=None):
        area_output_list = []
        delay_output_list = []
        weighted_output_list = []

        x = x.to(self.device)
        weight_condition = weight_condition.to(self.device)
        delay_condition = delay_condition.to(self.device)
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
            [output, weight_condition, delay_condition], dim=1
        )
        # multi-task vector factor q value output
        for i in range(self.num_tasks):
            # area obj
            output_column_area = self.fc_column_area[i](conditioned_input)
            output_type_area = self.fc_type_area[i](conditioned_input)
            output_i_area = self._combine(
                output_column_area, output_type_area
            )
            

            # delay obj
            output_column_delay = self.fc_column_delay[i](conditioned_input)
            output_type_delay = self.fc_type_delay[i](conditioned_input)
            output_i_delay = self._combine(
                output_column_delay, output_type_delay
            )

            weighted_output = weight_condition[0,0] * output_i_area + weight_condition[0,1] * output_i_delay

            output_i_area = output_i_area.masked_fill(~mask.to(self.device),-1000)
            area_output_list.append(output_i_area)
            output_i_delay = output_i_delay.masked_fill(~mask.to(self.device),-1000)
            delay_output_list.append(output_i_delay)
            weighted_output = weighted_output.masked_fill(~mask.to(self.device),-1000)
            weighted_output_list.append(weighted_output)

        return area_output_list, delay_output_list, weighted_output_list
    
    def select_action(self, state, steps_done, task_index, task_weight_vector, target_delay, deterministic=False, is_softmax=False):
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
        delay_condition = torch.tensor([target_delay]).float().unsqueeze(0)
        
        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                mask = self.mask_with_legality(state).to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                q_area_list, q_delay_list, q_list = self(state, weight_condition, delay_condition, state_mask=mask)
                q = q_list[task_index]
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q
                info["task_index"] = task_index
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

class MultiTaskVectorFactorConditionDeepQPolicyV2(MultiTaskFactorDeepQPolicy):
    def __init__(
        self, block, model_hidden_dim=256, is_non_linear=True,
        task_weight_vectors=[[4,1],[3,2],[2,3],[1,4]], 
        condition_input_num=3, **policy_kwargs
    ):
        super(MultiTaskVectorFactorConditionDeepQPolicyV2, self).__init__(
            block, model_hidden_dim=model_hidden_dim, is_non_linear=is_non_linear,
            task_weight_vectors=task_weight_vectors, is_condition_vector=True, **policy_kwargs
        )
        self.condition_input_num = condition_input_num
        # first encode the embedding
        self.fully_connected_layer = nn.Sequential(
                nn.Linear(512 * block.expansion, self.model_hidden_dim),
                nn.ReLU()
            )
        
        self.fc_column_area = nn.ModuleList()
        self.fc_column_delay = nn.ModuleList()
        self.fc_type_area = nn.ModuleList()
        self.fc_type_delay = nn.ModuleList()
        for _ in range(self.num_tasks):
            # column q area delay
            fc_column_area = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_column)
            )
            fc_column_delay = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_column)
            )
            self.fc_column_area.append(fc_column_area)
            self.fc_column_delay.append(fc_column_delay)
            
            # type q area delay
            fc_type_area = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_type)
            )
            fc_type_delay = nn.Sequential(
                nn.Linear(self.model_hidden_dim+self.condition_input_num, self.model_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.model_hidden_dim, self.num_action_type)
            )
            self.fc_type_area.append(fc_type_area)
            self.fc_type_delay.append(fc_type_delay)

    def forward(self, x, weight_condition, delay_condition, is_target=False, state_mask=None):
        area_output_list = []
        delay_output_list = []
        weighted_output_list = []

        x = x.to(self.device)
        weight_condition = weight_condition.to(self.device)
        delay_condition = delay_condition.to(self.device)
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
            [output, weight_condition, delay_condition], dim=1
        )
        # multi-task vector factor q value output
        for i in range(self.num_tasks):
            # area obj
            output_column_area = self.fc_column_area[i](conditioned_input)
            output_type_area = self.fc_type_area[i](conditioned_input)
            output_i_area = self._combine(
                output_column_area, output_type_area
            )
            

            # delay obj
            output_column_delay = self.fc_column_delay[i](conditioned_input)
            output_type_delay = self.fc_type_delay[i](conditioned_input)
            output_i_delay = self._combine(
                output_column_delay, output_type_delay
            )

            weighted_output = weight_condition[0,0] * output_i_area + weight_condition[0,1] * output_i_delay

            output_i_area = output_i_area.masked_fill(~mask.to(self.device),-1000)
            area_output_list.append(output_i_area)
            output_i_delay = output_i_delay.masked_fill(~mask.to(self.device),-1000)
            delay_output_list.append(output_i_delay)
            weighted_output = weighted_output.masked_fill(~mask.to(self.device),-1000)
            weighted_output_list.append(weighted_output)

        return area_output_list, delay_output_list, weighted_output_list
    
    def select_action(self, state, steps_done, task_index, task_weight_vector, target_delay, deterministic=False, is_softmax=False):
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
        delay_condition = torch.tensor([target_delay]).float().unsqueeze(0)
        
        if deterministic:
            eps_threshold = 0.

        info["stage_num"] = stage_num
        info["eps_threshold"] = eps_threshold

        if sample >= eps_threshold:
            with torch.no_grad():
                mask = self.mask_with_legality(state).to(self.device)
                simple_mask = self.mask(state)
                state = state.unsqueeze(0)
                q_area_list, q_delay_list, q_list = self(state, weight_condition, delay_condition, state_mask=mask)
                q = q_list[task_index]
                neg_inf = torch.tensor(float('-inf'), device=self.device)
                q = q.masked_fill(~mask,neg_inf)

                info["mask"] = mask.cpu()
                info["simple_mask"] = simple_mask
                info["q_value"] = q
                info["task_index"] = task_index
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

    # MultiTaskFactorDeepQPolicy
    # q_policy = MultiTaskFactorDeepQPolicy(
    #     BasicBlock, pool_is_stochastic=True, device=device, num_classes=16*4
    # ).to(device)
    # env = RefineEnv(1, q_policy, is_multi_obj=True, initial_state_pool_max_len=20)
    # state, _ = env.reset()
    # print(state)
    # # set_trace()
    # st = time.time()
    # for i in range(2):
    #     action, info = q_policy.select_action(torch.tensor(state), 1, 0, deterministic=True)
    #     q_value = info["q_value"]
    #     print(f"action: {action}, stage num {info['stage_num']}, q value {q_value}")
    #     # q_policy.partially_reset()
    # et = time.time() - st
    # print(f"time: {et}")

    # MultiTaskVectorFactorConditionDeepQPolicy
    q_policy = MultiTaskVectorFactorConditionDeepQPolicy(
        BasicBlock, pool_is_stochastic=True, device=device, num_classes=16*4
    ).to(device)
    env = RefineEnv(1, q_policy, is_multi_obj=False, is_multi_obj_condiiton=True, initial_state_pool_max_len=20)
    state, _ = env.reset()
    print(state)
    # set_trace()
    st = time.time()
    for i in range(2):
        action, info = q_policy.select_action(torch.tensor(state), 1, 0, [4,1], 1, deterministic=True)
        q_value = info["q_value"]
        print(f"action: {action}, stage num {info['stage_num']}, q value {q_value}")
        # q_policy.partially_reset()
    et = time.time() - st
    print(f"time: {et}")


    # mlp = SoftmaxMLP(8, 12).to("cuda:1")
    # input_x = torch.randn((1,8), dtype=torch.float, device="cuda:1")
    # probs, logits = mlp(input_x)

    # print(f"probs: {probs}, logits: {logits}")
