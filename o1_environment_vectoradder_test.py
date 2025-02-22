import os
import copy
import math
import numpy as np
import torch
from multiprocessing import Pool
import random 
from collections import deque
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp

from utils.vectoraddertoverilog import *
from o5_utils_vector_adder import abc_constr_gen, sta_scripts_gen, ys_scripts_gen, ys_scripts_v2_gen, ys_scripts_v3_gen, ys_scripts_v5_gen, get_ppa, EasyMacPath, EasyMacTarPath, BenchmarkPath
from ipdb import set_trace

class BaseEnv():
    def __init__(
        self, seed, **env_kwargs
    ):
        self.seed = seed 
        self.env_kwargs = env_kwargs

        self.set_seed()

    def set_seed(self, seed=None):
        if seed:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(self.seed)
    
    def reset(self):
        raise NotImplementedError 
    
    def step(self):
        raise NotImplementedError

class RefineEnvVectorAdder(BaseEnv):
    def __init__(
        self, seed, q_policy,
        build_path="build", synthesis_path="dqn", mul_booth_file="mul_booth_8.test3",
        num = 17,width="8_bits_booth", target_delay=[50,250,400,650], 
        wallace_area=((517+551+703+595)/4), wallace_delay=((1.0827+1.019+0.9652+0.9668)/4),
        weight_area=4, weight_delay=1, ppa_scale=100, initial_state_pool_max_len=0, 
        load_initial_state_pool_npy_path='None', load_pool_index=1, reward_scale=100, long_term_reward_scale=1.0, 
        reset_state_policy="random", random_reset_steps=201, store_state_type='less', is_policy_column=False,
        is_policy_seq=False, alpha=1, is_debug=False, reward_type="simulate", ppa_model_path=None, MAX_STAGE_NUM=4,
        action_num=4, synthesis_type="v1", normalize_reward_type="wallace", is_multi_obj=False, 
        is_multi_obj_condiiton=False, task_index=0, max_target_delay=650, **env_kwargs
    ):
        super().__init__(
            seed, **env_kwargs
        )
        self.num = num
        self.width = width
        self.cur_state = None
        self.compressed_state = None
        self.task_index = task_index
        self.max_target_delay = max_target_delay
        # makedir synthesis path
        self.synthesis_path = f"{synthesis_path}_{self.num}*{self.width}_{self.task_index}"
        
        self.initial_cwd_path = os.getcwd()
        self.synthesis_path = os.path.join(self.initial_cwd_path, self.synthesis_path)
        if not os.path.exists(self.synthesis_path):
            os.mkdir(self.synthesis_path)
        # makedir build path for compressor tree file text
        self.build_path = os.path.join(self.initial_cwd_path, build_path)
        if not os.path.exists(self.build_path):
            os.mkdir(self.build_path)
        # mul_booth file
        self.mul_booth_file = mul_booth_file
        # target delay 
        self.target_delay = target_delay
        self.n_processing = len(self.target_delay)
        # wallace area delay
        self.wallace_area = wallace_area
        self.wallace_delay = wallace_delay
        self.weight_area = weight_area
        self.weight_delay = weight_delay
        self.ppa_scale = ppa_scale
        self.last_area = 0
        self.last_delay = 0
        self.last_ppa = 0
        self.last_normalize_area = 0
        self.last_normalize_delay = 0
        self.long_term_reward_scale = long_term_reward_scale
        self.reward_scale = reward_scale
        # reset policy
        self.reset_state_policy = reset_state_policy
        self.random_reset_steps = random_reset_steps
        self.store_state_type = store_state_type
        self.is_policy_column = is_policy_column
        self.is_policy_seq = is_policy_seq
        self._alpha = alpha
        # pp_encode type
        # debug mode
        self.is_debug = is_debug
        # reward type
        self.reward_type = reward_type
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        # ppa model path
        self.ppa_model_path = ppa_model_path
        # synthesis type
        self.synthesis_type = synthesis_type
        # normalize reward type
        self.normalize_reward_type = normalize_reward_type
        # multi obj
        self.is_multi_obj = is_multi_obj
        self.is_multi_obj_condiiton = is_multi_obj_condiiton
        
        # action num
        self.action_num = action_num
        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5
        }
        self.q_policy = q_policy
        if q_policy is not None:
            self.device = q_policy.device
        if self.ppa_model_path is not None:
            self.ppa_model = DeepQPolicy(
                BasicBlock,
                num_classes=1
            )
            self.ppa_model.load_state_dict(torch.load(self.ppa_model_path))
            self.ppa_model.to(self.device)
        # initial state pool
        self.initial_state_pool_max_len = initial_state_pool_max_len
        if initial_state_pool_max_len > 0:
            PartialProduct,InitialState = self.wallace_for_adder()
            self.initial_wallace_state = copy.deepcopy(InitialState)

    
            initial_partial_product = PartialProduct
            ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product, self.initial_wallace_state)
            threed_state = self._get_image_state(ct32, ct22, stage_num)
            self.initial_wallace_3d_state = threed_state
            self.initial_state_pool = deque([],maxlen=initial_state_pool_max_len)
            self.imagined_initial_state_pool = deque([],maxlen=initial_state_pool_max_len)
            if q_policy is not None:
                initial_mask = self.get_state_mask(q_policy)
            else:
                initial_mask = None
            if self.reward_type == "simulate":
                ppa, normalize_area, normalize_delay = self._compute_ppa(self.wallace_area, self.wallace_delay)
                self.initial_state_pool.append(
                    {
                        "state": self.initial_wallace_state,
                        "threed_state": threed_state,
                        "area": self.wallace_area,
                        "delay": self.wallace_delay,
                        "state_mask": initial_mask,
                        "ppa": ppa,
                        "count": 1,
                        "state_type": "best_ppa",
                        "normalize_area": normalize_area,
                        "normalize_delay": normalize_delay
                    }
                )
                self.imagined_initial_state_pool.append(
                    {
                        "state": self.initial_wallace_state,
                        "threed_state": threed_state,
                        "area": self.wallace_area,
                        "delay": self.wallace_delay,
                        "state_mask": initial_mask,
                        "ppa": ppa,
                        "count": 1,
                        "state_type": "best_ppa"
                    }
                )
            elif self.reward_type == "node_num":
                self.initial_state_pool.append(
                {
                    "state": self.initial_wallace_state,
                    "threed_state": threed_state,
                    "area": 0,
                    "delay": 0,
                    "state_mask": initial_mask,
                    "ppa": self.initial_wallace_state.sum(),
                    "count": 1,
                    "state_type": "best_ppa"
                }
            )
            elif self.reward_type == "node_num_v2":
                ppa = 3 * ct32.sum() + 2 * ct22.sum()
                self.initial_state_pool.append(
                {
                    "state": self.initial_wallace_state,
                    "threed_state": threed_state,
                    "area": 0,
                    "delay": 0,
                    "state_mask": initial_mask,
                    "ppa": ppa,
                    "count": 1,
                    "state_type": "best_ppa",
                    "normalize_area": 0,
                    "normalize_delay": 0
                }
            )
            elif self.reward_type == "ppa_model":
                predict_ppa = self._predict_state_ppa(ct32, ct22, stage_num)
                self.initial_state_pool.append(
                {
                    "state": self.initial_wallace_state,
                    "threed_state": threed_state,
                    "area": 0,
                    "delay": 0,
                    "state_mask": initial_mask,
                    "ppa": predict_ppa,
                    "count": 1,
                    "state_type": "best_ppa"
                }
            )
            
        # config abc and openroad sta
        self.config_abc_sta()
        # config easymac
        # self.config_easymac()
        # load initial state pool
        self.load_initial_state_pool_npy_path = load_initial_state_pool_npy_path
        self.load_pool_index = load_pool_index
        if self.load_initial_state_pool_npy_path != 'None':
            self.npy_pool = np.load(
                self.load_initial_state_pool_npy_path, allow_pickle=True
            ).item()

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.initial_state_pool_max_len > 0:
            if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                # push the best ppa state into the initial pool
                avg_area = np.mean(rewards_dict['area'])
                avg_delay = np.mean(rewards_dict['delay'])
                self.initial_state_pool.append(
                    {
                        "state": copy.deepcopy(state),
                        "area": avg_area,
                        "delay": avg_delay,
                        "ppa": rewards_dict['avg_ppa'],
                        "count": 1,
                        "state_mask": state_mask,
                        "state_type": "best_ppa",
                        "normalize_area": rewards_dict["normalize_area"],
                        "normalize_delay": rewards_dict["normalize_delay"]
                    }
                )
        if self.found_best_info["found_best_ppa"] > rewards_dict['avg_ppa']:
            self.found_best_info["found_best_ppa"] = rewards_dict['avg_ppa']
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict['area']) 
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict['delay'])
    def _compute_ppa(self, area, delay):
        if self.normalize_reward_type == "wallace":
            normalize_area = self.ppa_scale * (area / self.wallace_area)
            normalize_delay = self.ppa_scale * (delay / self.wallace_delay)
            ppa = self.weight_area * (area / self.wallace_area) + self.weight_delay * (delay / self.wallace_delay)
            ppa = self.ppa_scale * ppa
        elif self.normalize_reward_type == "constant":
            # balance the scale of area and delay to balance their influence
            normalize_area = self.ppa_scale * (area / 100)
            normalize_delay = self.ppa_scale * (delay * 10)
            ppa = self.weight_area * (area / 100) + self.weight_delay * (delay * 10)
            ppa = self.ppa_scale * ppa
        return ppa, normalize_area, normalize_delay

    def _normalize_area_delay(self, area, delay):
        if self.normalize_reward_type == "wallace":
            normalize_area = area / self.wallace_area
            normalize_delay = delay / self.wallace_delay
        elif self.normalize_reward_type == "constant":
            normalize_area = area / 100
            normalize_delay = delay * 10
        return normalize_area, normalize_delay
    
    def _get_image_state(self, ct32, ct22, stage_num):
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.width)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        return image_state
    
    def _predict_state_ppa(self, ct32, ct22, stage_num):
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.width)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        image_state = torch.tensor(
            image_state,
            dtype=torch.float,
            device=self.device
        )
        with torch.no_grad():
            predict_ppa = self.ppa_model(image_state.unsqueeze(0))  

        return predict_ppa.item()

    def _model_evaluation(self, ppa_model, ct32, ct22, stage_num):
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.width)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        image_state = torch.tensor(
            image_state,
            dtype=torch.float,
            device=self.device
        )
        with torch.no_grad():
            normalize_area, normalize_delay = ppa_model(
                image_state.unsqueeze(0)
            )
        normalize_area = normalize_area.item()
        normalize_delay = normalize_delay.item()

        avg_ppa = self.weight_area * normalize_area + self.weight_delay * normalize_delay
        
        # avg_ppa = avg_ppa * self.ppa_scale
        
        reward = self.last_ppa - avg_ppa
        last_state_ppa = self.last_ppa
        # update last area delay
        self.last_ppa = avg_ppa
        return reward, avg_ppa, last_state_ppa, normalize_area, normalize_delay
    
    def get_mutual_distance(self):
        number_states = len(self.initial_state_pool)
        mutual_distances = np.zeros((number_states, number_states))
        for i in range(number_states):
            cur_state = self.initial_state_pool[i]["state"]
            for j in range(number_states):
                mutual_dis = np.linalg.norm(
                    cur_state - self.initial_state_pool[j]["state"],
                    ord=2
                )
                mutual_distances[i,j] = mutual_dis
        mutual_distances = np.around(
            mutual_distances,
            decimals=2
        )
        return mutual_distances

    def get_state_mask(self, policy):
        if self.is_policy_column:
            _, _, next_state_policy_info = policy.select_action(
                    torch.tensor(self.initial_wallace_state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
        elif self.is_policy_seq:
            _, _, next_state_policy_info = policy.action(
                self.initial_wallace_state
            )
            self.wallace_seq_state = next_state_policy_info['seq_state_pth']
            return next_state_policy_info['mask_pth']
        elif self.is_multi_obj:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(self.initial_wallace_state), 0, 0,
                    deterministic=False,
                    is_softmax=False
                )
        elif self.is_multi_obj_condiiton:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(self.initial_wallace_state), 0, 0,
                    [self.wallace_area, self.wallace_delay], self.target_delay[0] / 1500,
                    deterministic=False,
                    is_softmax=False
                )
        else:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(self.initial_wallace_state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
        return next_state_policy_info['mask']


    def config_abc_sta(self, target_delay=None):
        # generate a config dir for each target delay
        if target_delay is None:
            target_delay = self.target_delay
        for i in range(len(target_delay)):
            ys_path = os.path.join(self.synthesis_path, f"ys{i}")
            if not os.path.exists(ys_path):
                os.mkdir(ys_path)
            abc_constr_gen(ys_path)
            sta_scripts_gen(ys_path)
    
    def select_state_from_pool(self, state_novelty, state_value):
        if state_novelty is None and state_value is None:
            sel_indexes = range(0, len(self.initial_state_pool))
            sel_index = random.sample(sel_indexes, 1)[0]
            initial_state = self.initial_state_pool[sel_index]["state"]
        else:            
            if self.reset_state_policy == "random":
                sel_indexes = range(0, len(self.initial_state_pool))
                sel_index = random.sample(sel_indexes, 1)[0]
                initial_state = self.initial_state_pool[sel_index]["state"]
            elif self.reset_state_policy == "novelty_driven":
                sel_index = np.argmax(state_novelty)
                initial_state = self.initial_state_pool[sel_index]["state"]
            elif self.reset_state_policy in ["value_driven", "average_value_driven"]:
                sel_index = np.argmax(state_value)
                initial_state = self.initial_state_pool[sel_index]["state"]
            elif self.reset_state_policy in ["softmax_value_driven", "average_softmax_value_driven"]:
                q_distribution = Categorical(logits=torch.tensor(state_value))
                sel_index = q_distribution.sample()
                initial_state = self.initial_state_pool[sel_index]["state"]
            elif self.reset_state_policy == "ppa_driven":
                sampling_probs = state_value**self._alpha / np.sum(state_value**self._alpha)
                sel_index = np.random.choice(
                    np.arange(state_value.shape[0]),
                    p=sampling_probs
                )
                initial_state = self.initial_state_pool[sel_index]["state"]
                # update count 
                self.initial_state_pool[sel_index]["count"] += 1
        return initial_state, sel_index

    def reset_from_pool(self, state_novelty, state_value):
        initial_state, sel_index = self.select_state_from_pool(state_novelty, state_value)
        self.cur_state = copy.deepcopy(initial_state)

        self.last_area = self.initial_state_pool[sel_index]["area"]
        self.last_delay = self.initial_state_pool[sel_index]["delay"]
        self.last_ppa = self.initial_state_pool[sel_index]["ppa"]
        self.last_normalize_area = self.initial_state_pool[sel_index]["normalize_area"]
        self.last_normalize_delay = self.initial_state_pool[sel_index]["normalize_delay"]
        # self.last_ppa = self.ppa_scale * (
        #     self.weight_area * (self.last_area / self.wallace_area) + self.weight_delay * (self.last_delay / self.wallace_delay)
        # )
        return initial_state, sel_index

    def reset_from_wallace(self):
        # baseline 算法使用
        _,initial_state = self.wallace_for_adder() 
        self.cur_state = copy.deepcopy(initial_state)
        self.last_area = self.wallace_area
        self.last_delay = self.wallace_delay
        self.last_ppa = self.ppa_scale * (
            self.weight_area * (self.last_area / self.wallace_area) + self.weight_delay * (self.last_delay / self.wallace_delay)
        )
        return initial_state

    def reset_from_loaded_pool(self):
        #TODO：接近废弃，不再使用
        state_pool = self.npy_pool["env_initial_state_pool"]
        if self.load_pool_index < len(state_pool):
            load_index = self.load_pool_index
        else:
            load_index = len(state_pool) - 1
        initial_state = state_pool[load_index]["state"]
        area = state_pool[load_index]["area"]
        delay = state_pool[load_index]["delay"]
        print(f"initial state: {initial_state}")
        self.cur_state = copy.deepcopy(initial_state)
        self.last_area = area 
        self.last_delay = delay
        self.last_ppa = self.ppa_scale * (
            self.weight_area * (self.last_area / self.wallace_area) + self.weight_delay * (self.last_delay / self.wallace_delay)
        )
        return initial_state

    def reset(self, state_novelty=None, state_value=None):
        if self.initial_state_pool_max_len > 0:
            initial_state, sel_index = self.reset_from_pool(state_novelty, state_value)
        else:
            sel_index = 0
            if self.load_initial_state_pool_npy_path != 'None':
                initial_state = self.reset_from_loaded_pool()
            else:
                initial_state = self.reset_from_wallace()
        return initial_state, sel_index

    def get_final_partial_product(self, initial_partial_product):
        final_partial_product = np.zeros(self.width + 1)
        for i in range(1, int(self.width)):
            final_partial_product[i] = initial_partial_product[i] + self.cur_state[0][i-1] + \
                self.cur_state[1][i-1] - 2 * self.cur_state[0][i] - self.cur_state[1][i]
        final_partial_product[self.width] = 0 # the last column 2*n+1 must contain 0 bits
        return final_partial_product

    def update_state(self, action_column, action_type, final_partial_product):
        #change the CT structure, 执行动作，更新state记录的compressor 结构，以及partial product，partial product应该是用来legal的
        if action_type == 0:
            # add a 2:2 compressor
            self.cur_state[1][action_column] += 1
            final_partial_product[action_column] -= 1
            final_partial_product[action_column+1] += 1
        elif action_type == 1:
            # remove a 2:2 compressor
            self.cur_state[1][action_column] -= 1
            final_partial_product[action_column] += 1
            final_partial_product[action_column+1] -= 1
        elif action_type == 2:
            # replace a 3:2 compressor with a 2:2 compressor
            self.cur_state[1][action_column] += 1
            self.cur_state[0][action_column] -= 1
            final_partial_product[action_column] += 1
        elif action_type == 3:
            # replace a 2:2 compressor with a 3:2 compressor
            self.cur_state[1][action_column] -= 1
            self.cur_state[0][action_column] += 1
            final_partial_product[action_column] -= 1
        else:
            raise NotImplementedError
        
        return final_partial_product

    def legalization(self, action_column, updated_partial_product):
        # start from the next column
        legal_num_column = 0
        for i in range(action_column+1, self.width):
            if updated_partial_product[i] in [1, 2]:
                # it is legal, so break
                break 
            elif updated_partial_product[i] == 3:
                # add a 3:2 compressor
                self.cur_state[0][i] += 1 
                updated_partial_product[i] = 1
                updated_partial_product[i+1] += 1
            elif updated_partial_product[i] == 0:
                # if 2:2 compressor exists, remove a 2:2
                if self.cur_state[1][i] >= 1:
                    self.cur_state[1][i] -= 1
                    updated_partial_product[i] += 1
                    updated_partial_product[i+1] -= 1
                # else: remove a 3:2
                else:
                    self.cur_state[0][i] -= 1
                    updated_partial_product[i] += 2
                    updated_partial_product[i+1] -= 1
            legal_num_column += 1
        return updated_partial_product, legal_num_column

    def decompose_compressor_tree(self, initial_partial_product, state):
        # 1. convert the current state to the EasyMac text file format, matrix to tensor
        next_state = np.zeros_like(state)
        next_state[0] = state[0]
        next_state[1] = state[1]
        stage_num = 0
        ct32 = np.zeros([1,int(self.width)])
        ct22 = np.zeros([1,int(self.width)])
        ct32[0] = next_state[0]
        ct22[0] = next_state[1]
        partial_products = np.zeros([1,int(self.width)])
        partial_products[0] = initial_partial_product
        # decompose each column sequentially
        for i in range(int(self.width)):
            j = 0 # j denotes the stage index, i denotes the column index
            while (j <= stage_num): # the condition is impossible to satisfy
                
                # j-th stage i-th column
                ct32[j][i] = next_state[0][i]
                ct22[j][i] = next_state[1][i]
                # initial j-th stage partial products
                if j == 0: # 0th stage
                    partial_products[j][i] = partial_products[j][i]
                else:
                    if i==0:
                        partial_products[j][i] = partial_products[j-1][i] 
                    else:

                        partial_products[j][i] = partial_products[j-1][i] + \
                        ct32[j-1][i-1] + ct22[j-1][i-1]

                # when to break 
                if (3*ct32[j][i] + 2*ct22[j][i]) <= partial_products[j][i]:
                    # print(f"i: {ct22[j][i]}, i-1: {ct22[j][i-1]}")
                    # update j-th stage partial products for the next stage
                    partial_products[j][i] = partial_products[j][i] - \
                        ct32[j][i]*2 - ct22[j][i]
                    # update the next state compressors
                    next_state[0][i] -= ct32[j][i]
                    next_state[1][i] -= ct22[j][i]
                    break # the only exit
                else:
                    if j == stage_num:
                        # print(f"j {j} stage num: {stage_num}")
                        # add initial next stage partial products and cts
                        stage_num += 1
                        ct32 = np.r_[ct32,np.zeros([1,int(self.width)])]
                        ct22 = np.r_[ct22,np.zeros([1,int(self.width)])]
                        partial_products = np.r_[partial_products,np.zeros([1,int(self.width)])]
                    # assign 3:2 first, then assign 2:2
                    # only assign the j-th stage i-th column compressors
                    if (ct32[j][i] >= partial_products[j][i]//3):
                        ct32[j][i] = partial_products[j][i]//3
                        if (partial_products[j][i]%3 == 2):
                            if (ct22[j][i] >= 1):
                                ct22[j][i] = 1
                        else:
                            ct22[j][i] = 0
                    else:
                        ct32[j][i] = ct32[j][i]
                        if(ct22[j][i] >= (partial_products[j][i]-ct32[j][i]*3)//2):
                            ct22[j][i] = (partial_products[j][i]-ct32[j][i]*3)//2
                        else:
                            ct22[j][i] = ct22[j][i]
                    
                    # update partial products
                    partial_products[j][i] = partial_products[j][i] - ct32[j][i]*2 - ct22[j][i]
                    next_state[0][i] = next_state[0][i] - ct32[j][i]
                    next_state[1][i] = next_state[1][i] - ct22[j][i]
                j += 1
        # 2. write the compressors information into the text file
        sum = int(ct32.sum() + ct22.sum())
        file_name = os.path.join(self.build_path, f"compressor_tree_test_{self.task_index}.txt")
        with open(file_name, mode="w") as f:
            f.write(str(self.num) + ' ' + str(self.width))
            f.write('\n')
            f.write(str(sum))
            f.write('\n')
            for i in range(0, stage_num+1):
                for j in range(0, int(self.width)):
                    # write 3:2 compressors
                    for k in range(0, int(ct32[i][self.width-1-j])):
                        f.write(str( int(self.width)-1-j ))
                        f.write(' 1')
                        f.write('\n')
                    for k in range(0, int( ct22[i][int(self.width)-1-j] )):
                        f.write(str( int(self.width)-1-j ))
                        f.write(' 0')
                        f.write('\n')
        print(f"stage num: {stage_num}")
        return ct32, ct22, partial_products, stage_num

    def get_reward(self, n_processing=None, target_delays=None):
        # 1. Use the EasyMac to generate RTL files
        compressor_file = os.path.join(self.build_path, f"compressor_tree_test_{self.task_index}.txt")
        rtl_file = os.path.join(self.synthesis_path, 'rtl')
        if not os.path.exists(rtl_file):
            os.mkdir(rtl_file)
        ct_file =os.path.join(self.build_path, f"compressor_tree_test_{self.task_index}.txt")
        num,width,ct  =read_ct(ct_file)
        rtl_generate_cmd = write_adder(rtl_file+'/Adder.v',width,ct,num )

        # 2. Use the RTL file to run openroad yosys
        if target_delays is None:
            n_processing = self.n_processing
            target_delays = self.target_delay

        ppas_dict = {
            "area": [],
            "delay": [],
            "power": []
        }
        """ 
        def collect_ppa(ppa_dict):
            for k in ppa_dict.keys():
                ppas_dict[k].append(ppa_dict[k])
        for i in range(n_processing):
            ys_path = os.path.join(self.synthesis_path, f"ys{i}")
            collect_ppa(self.simulate_for_ppa(self.target_delay[i],ys_path,self.synthesis_path,self.synthesis_type))
        return ppas_dict
        """
        print(n_processing,target_delays)
        with Pool(processes=n_processing) as pool:
            def collect_ppa(ppa_dict):
                for k in ppa_dict.keys():
                    ppas_dict[k].append(ppa_dict[k])

            for i, target_delay in enumerate(target_delays):
                ys_path = os.path.join(self.synthesis_path, f"ys{i}")
                pool.apply_async(
                    func=RefineEnvVectorAdder.simulate_for_ppa,
                    args=(target_delay, ys_path, self.synthesis_path, self.synthesis_type),
                    callback=collect_ppa
                )

            pool.close()
            pool.join()
        
        return ppas_dict

    def process_reward(self, rewards_dict):
        avg_area = np.mean(rewards_dict['area'])
        avg_delay = np.mean(rewards_dict['delay'])
        # compute ppa
        avg_ppa, normalize_area, normalize_delay = self._compute_ppa(
            avg_area, avg_delay
        )
        # immediate reward
        reward = self.last_ppa - avg_ppa
        area_reward = self.last_normalize_area - normalize_area
        delay_reward = self.last_normalize_delay - normalize_delay
        # long-term reward
        long_term_reward = (self.weight_area + self.weight_delay) * self.ppa_scale - avg_ppa
        reward = reward + self.long_term_reward_scale * long_term_reward
        last_state_ppa = self.last_ppa
        # update last area delay
        self.last_area = avg_area
        self.last_delay = avg_delay
        self.last_ppa = avg_ppa
        self.last_normalize_area = normalize_area
        self.last_normalize_delay = normalize_delay
        # normalize_area delay
        normalize_area_no_scale, normalize_delay_no_scale = self._normalize_area_delay(
            avg_area, avg_delay
        )        
        return reward, avg_ppa, last_state_ppa, normalize_area_no_scale, normalize_delay_no_scale, area_reward, delay_reward, normalize_area, normalize_delay

    @staticmethod
    def simulate_for_ppa(target_delay, ys_path, synthesis_path, synthesis_type):
        if synthesis_type == "v1":
            ys_scripts_gen(target_delay, ys_path, synthesis_path)
        elif synthesis_type == "v2":
            ys_scripts_v2_gen(target_delay, ys_path, synthesis_path)
        elif synthesis_type == "v3":
            ys_scripts_v3_gen(target_delay, ys_path, synthesis_path)
        elif synthesis_type == "v5":
            ys_scripts_v5_gen(target_delay, ys_path, synthesis_path)

        ppa_dict = get_ppa(ys_path)

        return ppa_dict

    def step(self, action, is_model_evaluation=False, ppa_model=None):
        """
            action is a number, action coding:
                action=0: add a 2:2 compressor
                action=1: remove a 2:2 compressor
                action=2: replace a 3:2 compressor
                action=3: replace a 2:2 compressor
            Input: cur_state, action
            Output: next_state
        """

        # 1. given initial partial product and compressor tree state, can get the final partial product
            # 其实这个压缩的过程可以建模为两种情况：一种是并行压缩，就要分阶段；一种是从低位到高位的顺序压缩，就没有阶段而言，就是让每一列消消乐；能不能把这两种建模结合呢？为什么要结合这两种呢？优缺点在哪里？
        # 2. perform action，update the compressor tree state and update the final partial product
        # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        # 4. Evaluate the updated compressor tree state to get the reward
            # 上一个state的average ppa 和 当前state 的 average ppa 的差值

        action_column = int(action) // 4
        action_type = int(action) % 4
        #initial_partial_product = PartialProduct[self.bit_width]
        initial_partial_product = np.full(self.width,self.num)
        state = self.cur_state
        print(state)
        # 1. compute final partial product from the lowest column to highest column
        # final_partial_product = self.get_final_partial_product(initial_partial_product)

        # # 2. perform action，update the compressor tree state and update the final partial product
        # updated_partial_product = self.update_state(action_column, action_type, final_partial_product)
        # # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        # legalized_partial_product, legal_num_column = self.legalization(action_column, updated_partial_product)
        
        legal_num_column = 0
        ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product, state)
        # 4. Decompose the compressor tree to multiple stages and write it to text
        next_state = copy.deepcopy(self.cur_state)
        #ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product, next_state)
        next_state = copy.deepcopy(self.cur_state)
        # 5. Evaluate the updated compressor tree state to get the reward
        if self.is_debug:
            # do not go through openroad simulation
            reward = 0
            rewards_dict = {
                "area": 0,
                "delay": 0,
                "avg_ppa": 0,
                "last_state_ppa": 0,
                "legal_num_column": 0,
                "normalize_area": 0,
                "normalize_delay":0
            }
        elif self.reward_type == "simulate":
            rewards_dict = {}
            if is_model_evaluation:
                assert ppa_model is not None
                reward, avg_ppa, last_state_ppa, normalize_area, normalize_delay = self._model_evaluation(
                    ppa_model, ct32, ct22, stage_num
                )
                rewards_dict['area'] = 0
                rewards_dict['delay'] = 0
            else:
                rewards_dict = self.get_reward()
                reward, avg_ppa, last_state_ppa, normalize_area_no_scale, normalize_delay_no_scale, area_reward, delay_reward, normalize_area, normalize_delay = self.process_reward(rewards_dict)
            print("2")
            rewards_dict['avg_ppa'] = avg_ppa
            rewards_dict['last_state_ppa'] = last_state_ppa
            rewards_dict['legal_num_column'] = legal_num_column
            rewards_dict['normalize_area_no_scale'] = normalize_area_no_scale
            rewards_dict['normalize_delay_no_scale'] = normalize_delay_no_scale
            rewards_dict['normalize_area'] = normalize_area
            rewards_dict['normalize_delay'] = normalize_delay
            rewards_dict['area_reward'] = area_reward
            rewards_dict['delay_reward'] = delay_reward
        elif self.reward_type == "node_num":
            ppa_estimation = next_state.sum()
            reward = self.last_ppa - ppa_estimation
            avg_ppa = ppa_estimation
            last_state_ppa = self.last_ppa
            rewards_dict = {
                "area": [0,0],
                "delay": [0,0],
                "avg_ppa": avg_ppa,
                "last_state_ppa": last_state_ppa,
                "legal_num_column": legal_num_column
            }
            self.last_ppa = ppa_estimation
        elif self.reward_type == "node_num_v2":
            ppa_estimation = 3 * ct32.sum() + 2 * ct22.sum()
            reward = self.last_ppa - ppa_estimation
            avg_ppa = ppa_estimation
            last_state_ppa = self.last_ppa
            rewards_dict = {
                "area": [0,0],
                "delay": [0,0],
                "avg_ppa": avg_ppa,
                "last_state_ppa": last_state_ppa,
                "legal_num_column": legal_num_column
            }
            self.last_ppa = ppa_estimation
        elif self.reward_type == "ppa_model":
            ppa_estimation = self._predict_state_ppa(
                ct32, ct22, stage_num
            )
            reward = self.reward_scale * (self.last_ppa - ppa_estimation)
            avg_ppa = ppa_estimation
            last_state_ppa = self.last_ppa
            rewards_dict = {
                "area": [0,0],
                "delay": [0,0],
                "avg_ppa": avg_ppa,
                "last_state_ppa": last_state_ppa,
                "legal_num_column": legal_num_column
            }
            self.last_ppa = ppa_estimation
        # print(f"ct32: {ct32} shape: {ct32.shape}")
        # print(f"ct22: {ct22} shape: {ct22.shape}")
        return next_state, reward, rewards_dict

    def get_ppa_full_delay_cons(self, test_state):
        initial_partial_product = np.full(self.width,self.num)
        ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product, test_state)
        # generate target delay
        target_delay=[]
        input_width = math.ceil(self.int_bit_width)
        if input_width == 8:
            for i in range(50,1000,10):
                target_delay.append(i)
        elif input_width == 16:
            for i in range(50,2000,10):
                target_delay.append(i)
        elif input_width == 32: 
            for i in range(50,3000,10):
                target_delay.append(i)
        elif input_width == 64: 
            for i in range(50,4000,10):
                target_delay.append(i)
        #for file in os.listdir(synthesis_path): 
        n_processing = 12
        # config_abc_sta
        self.config_abc_sta(target_delay=target_delay)
        # get reward 并行 openroad
        ppas_dict = self.get_reward(n_processing=n_processing, target_delays=target_delay)

        return ppas_dict

    
    def wallace_for_adder(self):
        pp = np.full(self.width,self.num).reshape(1,self.width)
        ct32 = np.zeros([1,self.width])
        ct22 = np.zeros([1,self.width])
        target = np.zeros(self.width)
        for i in range(0,self.width):
            target[i] = 2
        stage_num = 0
        while(True):
            for i in range(0,self.width):
                if(pp[stage_num][i]%3 == 0):
                    ct32[stage_num][i] = pp[stage_num][i]//3
                    ct22[stage_num][i] = 0
                elif(pp[stage_num][i]%3 == 1):
                    ct32[stage_num][i] = pp[stage_num][i]//3
                    ct22[stage_num][i] = 0
                elif(pp[stage_num][i]%3 == 2):
                    ct32[stage_num][i] = pp[stage_num][i]//3
                    if stage_num == 0:
                        ct22[stage_num][i] = 0
                    else:
                        ct22[stage_num][i] = 1
            stage_num = stage_num + 1
            pp = np.r_[pp,np.zeros([1,self.width])]
            pp[stage_num][0] = pp[stage_num-1][0] - ct32[stage_num-1][0]*2 - ct22[stage_num-1][0]
            for i in range(1,self.width): 
                pp[stage_num][i] = pp[stage_num-1][i] + ct32[stage_num-1][i-1] + ct22[stage_num-1][i-1]  - ct32[stage_num-1][i]*2 - ct22[stage_num-1][i]
            if (pp[stage_num] <= target).all():
                break
            else:
                ct32 = np.r_[ct32,np.zeros([1,self.width])]
                ct22 = np.r_[ct22,np.zeros([1,self.width])]
        ct32=np.sum(ct32,axis=0)
        ct22=np.sum(ct22,axis=0)
        ct=np.vstack((ct32,ct22))
        return pp[0],ct

if __name__ == '__main__':
    env = RefineEnvVectorAdder(
        1, None, mul_booth_file="mul.test2", num=32,width=64,
        target_delay=[50,2000,3000,4000], initial_state_pool_max_len=20,load_pool_index=3, reward_type="simulate",
        # load_initial_state_pool_npy_path='./outputs/2023-09-18/14-40-49/logger_log/test/dqn8bits_reset_v2_initialstate/dqn8bits_reset_v2_initialstate_2023_09_18_14_40_55_0000--s-0/itr_25.npy'
        wallace_area = ((517+551+703+595)/4), wallace_delay=((1.0827+1.019+0.9652+0.9668)/4),
        load_initial_state_pool_npy_path='None', synthesis_type="v1", is_debug=False
    )
    #state, _ = env.reset()
    #print(env.wallace_for_adder())
    #state = []
    _,state = env.wallace_for_adder()
    print(f"before state: {state} shape: {state.shape}")
    env.cur_state = state
    next_state, reward, rewards_dict = env.step(torch.tensor([5]))
    print(f"next state: {next_state} shape: {next_state.shape}")
    # state = env.reset()
    print(f"rewards: {rewards_dict}")

