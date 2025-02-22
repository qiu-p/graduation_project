"""
    DQN training algorithm drawed by the paper 
    "RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning"
"""
import copy
import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from collections import deque
import pandas as pd
from paretoset import paretoset
from pygmo import hypervolume
import joblib
import time
from scipy.spatial import ConvexHull, Delaunay

from o0_netlist import NetList
from o0_logger import logger
from o5_utils import Transition, MBRLTransition, MultiObjTransition, PDMultiObjTransition, MBRLMultiObjTransition
from o0_global_const import PartialProduct, DSRFeatureDim, MacPartialProduct
from o2_policy import MBRLPPAModel, BasicBlock
from utils.operators import Operators
from utils.expression_utils import Expression

from ipdb import set_trace

from ysh_logger import get_logger
ysh_logger = get_logger(logger_name='ysh', log_file='ysh_logger_output.txt')

class DQNAlgorithm():
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        is_target=False,
        deterministic=False,
        is_softmax=False,
        optimizer_class='RMSprop',
        q_net_lr=1e-2,
        batch_size=64,
        gamma=0.8,
        target_update_freq=10,
        len_per_episode=25,
        total_episodes=400,
        MAX_STAGE_NUM=4,
        # is double q
        is_double_q=False,
        # agent reset
        agent_reset_freq=0,
        agent_reset_type="xavier",
        device='cpu',
        action_num=4,
        # pareto
        reference_point=[2600, 1.8],
        # multiobj
        multiobj_type="pure_max", # [pure_max, weight_max]
        # store type
        store_type="simple", # simple or detail
        # end_exp_log
        end_exp_freq=25,
        is_mac=False
    ):
        self.env = env
        self.q_policy = q_policy
        self.target_q_policy = target_q_policy
        self.replay_memory = replay_memory

        # hyperparameter 
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.len_per_episode = len_per_episode
        self.total_episodes = total_episodes
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.device = device
        self.is_target = is_target
        self.is_double_q = is_double_q
        self.store_type = store_type
        self.action_num = action_num
        self.multiobj_type = multiobj_type
        self.end_exp_freq = end_exp_freq
        self.is_mac = is_mac
        # optimizer
        # TODO: lr, lrdecay, gamma
        # TODO: double q
        self.q_net_lr = q_net_lr
        if isinstance(optimizer_class, str):
            optimizer_class = eval('optim.'+optimizer_class)
            self.optimizer_class = optimizer_class
        self.policy_optimizer = optimizer_class(
            self.q_policy.parameters(),
            lr=self.q_net_lr
        )

        # loss function
        self.loss_fn = nn.SmoothL1Loss()

        # total steps
        self.total_steps = 0
        self.bit_width = env.bit_width
        self.encode_type = ''
        if 'booth' in self.bit_width:
            self.encode_type = 'booth'
        else:
            self.encode_type = 'and'
        self.int_bit_width = env.int_bit_width
        if self.is_mac:
            self.initial_partial_product = MacPartialProduct[self.bit_width][:-1]
        else:
            self.initial_partial_product = PartialProduct[self.bit_width][:-1]

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5
        }
        # agent reset
        self.agent_reset_freq = agent_reset_freq
        self.agent_reset_type = agent_reset_type

        self.deterministic = deterministic
        self.is_softmax = is_softmax

        # # pareto pointset
        self.pareto_pointset = {
            "area": [],
            "delay": [],
            "state": []
        }
        self.reference_point = reference_point
        # configure figure
        plt.switch_backend('agg')

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        self.replay_memory.push(
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1,-1),
            next_state_mask.reshape(1,-1)
            # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            # rewards_dict
        )

    def store_detail(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        self.replay_memory.push(
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1,-1),
            next_state_mask.reshape(1,-1),
            state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            rewards_dict
        )

    def compute_values(
        self, state_batch, action_batch, state_mask, is_average=False
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        net = NetList()
        for i in range(batch_size):
            cur_state = state_batch[i].cpu().numpy()
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            if i == 0:
                ysh_logger.info('ct32: {}'.format(str(ct32)))
                ysh_logger.info('ct22: {}'.format(str(ct22)))
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            # compute image state
            if action_batch is not None:
                # reshape 有问题************
                q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
                # q_values = self.target_q_policy(state.unsqueeze(0))                
                # if is_average:
                #     q_values = (q_values + 1000).detach()
                #     num = torch.count_nonzero(q_values)
                #     state_action_values[i] = q_values.sum() / (num+1e-4)
                is_with_power = True
                if self.is_double_q:
                    current_q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
                    index = torch.argmax(current_q_values)
                    state_action_values[i] = q_values.squeeze()[index].detach()
                elif is_with_power:
                    # state -> power
                    # action_list -> power_list
                    # 一个问题：q_value in R, power_coefficient 范围?
                    #
                    # report_power(bit_width, encode_type, ct, pp_wiring=None)
                    #       ct: [[1, 2, ...], [3, 4, ...]]
                    #
                    # state -> power_coefficient
                    cur_power = net.report_power(math.ceil(self.int_bit_width), self.encode_type, cur_state, None)
                    mask_with_legality = self.target_q_policy.mask_with_legality(state)
                    next_states = self.env.get_nextstates(cur_state, mask_with_legality)
                    next_powers = []
                    for next_state in next_states:
                        if next_state == None:
                            next_powers.append(0)
                        next_power = net.report_power(math.ceil(self.int_bit_width), self.encode_type, next_state, None)
                        next_powers.append(next_power)
                    power_coefficient = torch.zeros(int(self.int_bit_width*2)*4)
                    for i in range(int(self.int_bit_width*2)*4):
                        power_coefficient[i] = cur_power - next_powers[i]
                    q_values_woth_power = q_values.squeeze().detach() * power_coefficient
                    index = torch.argmax(q_values_woth_power)
                    state_action_values[i] = q_values.squeeze()[index].detach()
                    state_action_values[i] = q_values.max(1)[0].detach()
                    if i == 0:
                        ysh_logger.info('cur_state: {}'.format(str(cur_state)))
                        ysh_logger.info('mask_with_legality: {}'.format(str(mask_with_legality)))
                        ysh_logger.info('cur_power: {}'.format(str(cur_power)))
                        ysh_logger.info('next_powers: {}'.format(str(next_powers)))
                        ysh_logger.info('q_values_woth_power: {}'.format(str(q_values_woth_power)))
                        ysh_logger.info('power_coefficient: {}'.format(str(power_coefficient)))
                else:      
                    state_action_values[i] = q_values.max(1)[0].detach()
        return state_action_values

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_mask = torch.cat(batch.mask)
            next_state_mask = torch.cat(batch.next_state_mask)
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(
                state_batch, action_batch, state_mask
            )
            next_state_values = self.compute_values(
                next_state_batch, None, next_state_mask
            )
            target_state_action_values = (next_state_values * self.gamma) + reward_batch.to(self.device)

            loss = self.loss_fn(
                state_action_values.unsqueeze(1), 
                target_state_action_values.unsqueeze(1)
            )

            self.policy_optimizer.zero_grad()
            loss.backward()
            for param in self.q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()

            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy()
            }
        
        return loss, info

    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        if self.store_type == "detail":
            save_data_dict["replay_buffer"] = self.replay_memory.memory
        # env initial state pool 
        if self.env.initial_state_pool_max_len > 0:
            save_data_dict["env_initial_state_pool"] = self.env.initial_state_pool
        # best state best design
        save_data_dict["found_best_info"] = self.found_best_info
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]
        
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        best_state = copy.deepcopy(self.found_best_info["found_best_state"])
        ppas_dict = self.env.get_ppa_full_delay_cons(best_state)
        save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(self.total_steps, save_data_dict)

        # save q policy model
        q_policy_state_dict = self.target_q_policy.state_dict()
        logger.save_itr_params(self.total_steps, q_policy_state_dict)

    def run_experiments(self):
        for episode_num in range(self.total_episodes):
            self.run_episode(episode_num)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num)
        self.end_experiments(episode_num)

    def build_state_dict(self, next_state):
        ct32 = {}
        ct22 = {}
        for i in range(next_state.shape[1]):
            ct32[f"{i}-th bit column"] = next_state[0][i]
            ct22[f"{i}-th bit column"] = next_state[1][i]
        return ct32, ct22

    def get_mask_stats(self, mask):
        number_column = int(len(mask)/4)
        valid_number_each_column = np.zeros(number_column)
        for i in range(number_column):
            cur_column_mask = mask[4*i:4*(i+1)]
            valid_number_each_column[i] = torch.sum(cur_column_mask)
        counter = Counter(valid_number_each_column)
        return counter

    def log_mask_stats(self, policy_info):
        if policy_info is not None:
            # log mask average
            logger.tb_logger.add_scalar('mask avg valid number', torch.sum(policy_info["mask"]), global_step=self.total_steps)
            logger.tb_logger.add_scalar('simple mask avg valid number', torch.sum(policy_info["simple_mask"]), global_step=self.total_steps)
            
            mask_counter = self.get_mask_stats(policy_info["mask"])
            simple_mask_counter = self.get_mask_stats(policy_info["simple_mask"])
            for k in mask_counter.keys():
                logger.tb_logger.add_scalar(f'mask number {k}', mask_counter[k], global_step=self.total_steps)
            for k in simple_mask_counter.keys():
                logger.tb_logger.add_scalar(f'simple mask number {k}', simple_mask_counter[k], global_step=self.total_steps)

            if "num_valid_action" in policy_info.keys():
                logger.tb_logger.add_scalar('number valid action', policy_info["num_valid_action"], global_step=self.total_steps)

            if "state_ct32" in policy_info.keys():
                if len(policy_info["state_ct32"].shape) == 2:
                    logger.tb_logger.add_image(
                        'state ct32', np.array(policy_info["state_ct32"]), global_step=self.total_steps, dataformats='HW'
                    )
                    logger.tb_logger.add_image(
                        'state ct22', np.array(policy_info["state_ct22"]), global_step=self.total_steps, dataformats='HW'
                    )
                    logger.tb_logger.add_image(
                        'state ct32 sum ct22', np.array(policy_info["state_ct32"])+np.array(policy_info["state_ct22"]), global_step=self.total_steps, dataformats='HW'
                    )
                elif len(policy_info["state_ct32"].shape) == 3:
                    logger.tb_logger.add_image(
                        'state ct32', np.array(policy_info["state_ct32"]), global_step=self.total_steps, dataformats='CHW'
                    )
                    logger.tb_logger.add_image(
                        'state ct22', np.array(policy_info["state_ct22"]), global_step=self.total_steps, dataformats='CHW'
                    )
                    logger.tb_logger.add_image(
                        'state ct32 sum ct22', np.concatenate((np.array(policy_info["state_ct32"]), np.array(policy_info["state_ct22"])), axis=0), global_step=self.total_steps, dataformats='CHW'
                    )

        # log mask figure
        # fig1 = plt.figure()
        # x = np.linspace(1, len(policy_info["mask"]), num=len(policy_info["mask"]))
        # f1 = plt.plot(x, policy_info["mask"], c='r')

        # logger.tb_logger.add_figure('mask', fig1, global_step=self.total_steps)
        
        # fig2 = plt.figure()
        # f2 = plt.plot(x, policy_info["simple_mask"], c='b')
        # logger.tb_logger.add_figure('simple_mask', fig2, global_step=self.total_steps)
    def log_action_stats(self, action):
        pass

    def log_stats(
        self, loss, reward, rewards_dict,
        next_state, action, info, policy_info, action_column=0
    ):
        try:
            loss = loss.item()
            q_values = np.mean(info['q_values'])
            target_q_values = np.mean(info['target_q_values'])
            positive_rewards_number = info['positive_rewards_number']
        except:
            loss = loss
            q_values = 0.
            target_q_values = 0.
            positive_rewards_number = 0.

        logger.tb_logger.add_scalar('train loss', loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar('reward', reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('legal num column', rewards_dict['legal_num_column'], global_step=self.total_steps)
        if "legal_num_stage" in rewards_dict.keys():
            logger.tb_logger.add_scalar('legal num stage', rewards_dict['legal_num_stage'], global_step=self.total_steps)  
        if "legal_num_column_pp3" in rewards_dict.keys():
            logger.tb_logger.add_scalar('legal_num_column_pp3', rewards_dict['legal_num_column_pp3'], global_step=self.total_steps)  
            logger.tb_logger.add_scalar('legal_num_column_pp0', rewards_dict['legal_num_column_pp0'], global_step=self.total_steps)  
            
        if len(action.shape) <= 2:
            logger.tb_logger.add_scalar('action_index', action, global_step=self.total_steps)
        if policy_info is not None:
            logger.tb_logger.add_scalar('stage_num', policy_info["stage_num"], global_step=self.total_steps)
            logger.tb_logger.add_scalar('eps_threshold', policy_info["eps_threshold"], global_step=self.total_steps)
        
        logger.tb_logger.add_scalar('action_column', action_column, global_step=self.total_steps)
        logger.tb_logger.add_scalar('positive_rewards_number', positive_rewards_number, global_step=self.total_steps)
        
        try:
            for i in range(len(self.found_best_info)):
                logger.tb_logger.add_scalar(f'best ppa {i}-th weight', self.found_best_info[i]["found_best_ppa"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'best area {i}-th weight', self.found_best_info[i]["found_best_area"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'best delay {i}-th weight', self.found_best_info[i]["found_best_delay"], global_step=self.total_steps)
        except:
            logger.tb_logger.add_scalar(f'best ppa', self.found_best_info["found_best_ppa"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best area', self.found_best_info["found_best_area"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best delay', self.found_best_info["found_best_delay"], global_step=self.total_steps)
            
        # log q values info 
        logger.tb_logger.add_scalar('q_values', q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('target_q_values', target_q_values, global_step=self.total_steps)

        # log state 
        # ct32, ct22 = self.build_state_dict(next_state)
        # logger.tb_logger.add_scalars(
        #     'compressor 32', ct32, global_step=self.total_steps)
        # logger.tb_logger.add_scalars(
        #     'compressor 22', ct22, global_step=self.total_steps)
        # logger.tb_logger.add_image(
        #     'state image', next_state, global_step=self.total_steps, dataformats='HW'
        # )

        # log wallace area wallace delay
        logger.tb_logger.add_scalar('wallace area', self.env.wallace_area, global_step=self.total_steps)
        logger.tb_logger.add_scalar('wallace delay', self.env.wallace_delay, global_step=self.total_steps)
        
        # log env initial state pool
        # if self.env.initial_state_pool_max_len > 0:
        #     avg_area, avg_delay = self.get_env_pool_log()
        #     logger.tb_logger.add_scalars(
        #         'env state pool area', avg_area, global_step=self.total_steps)
        #     logger.tb_logger.add_scalars(
        #         'env state pool delay', avg_delay, global_step=self.total_steps)

        # log mask stats
        self.log_mask_stats(policy_info)
        self.log_action_stats(action)

    def get_env_pool_log(self):
        avg_area = {}
        avg_delay = {}
        for i in range(len(self.env.initial_state_pool)):
            avg_area[f"{i}-th state area"] = self.env.initial_state_pool[i]["area"]
            avg_delay[f"{i}-th state delay"] = self.env.initial_state_pool[i]["delay"]
        return avg_area, avg_delay

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the best ppa state into the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    self.env.initial_state_pool.append(
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
            elif self.env.store_state_type == "leq":
                if self.found_best_info['found_best_ppa'] >= rewards_dict['avg_ppa']:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa"
                        }
                    )
            elif self.env.store_state_type == "ppa_with_diversity":
                # store best ppa state
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa"
                        }
                    )
                # store better diversity state
                elif rewards_dict['avg_ppa'] < self.env.initial_state_pool[0]["ppa"]:
                    number_states = len(self.env.initial_state_pool)
                    worse_state_distances = np.zeros(number_states-1)
                    cur_state_distances = np.zeros(number_states)
                    for i in range(number_states):
                        cur_state_dis = np.linalg.norm(
                            state - self.env.initial_state_pool[i]["state"],
                            ord=2
                        )
                        cur_state_distances[i] = cur_state_dis
                        if i == 0:
                            continue
                        worse_state_dis = np.linalg.norm(
                            self.env.initial_state_pool[0]["state"] - self.env.initial_state_pool[i]["state"],
                            ord=2
                        )
                        worse_state_distances[i-1] = worse_state_dis
 
                    worse_state_min_distance = np.min(worse_state_distances)
                    cur_state_min_distance = np.min(cur_state_distances)
                    if cur_state_min_distance > worse_state_min_distance:
                        # push the diverse state into the pool
                        avg_area = np.mean(rewards_dict['area'])
                        avg_delay = np.mean(rewards_dict['delay'])
                        self.env.initial_state_pool.append(
                            {
                                "state": copy.deepcopy(state),
                                "area": avg_area,
                                "delay": avg_delay,
                                "ppa": rewards_dict['avg_ppa'],
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "diverse"
                            }
                        )
        if self.found_best_info["found_best_ppa"] > rewards_dict['avg_ppa']:
            self.found_best_info["found_best_ppa"] = rewards_dict['avg_ppa']
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict['area']) 
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict['delay'])
    def reset_agent(self):
        if self.total_steps % self.agent_reset_freq == 0:
            self.q_policy.partially_reset(
                reset_type=self.agent_reset_type
            )
            self.target_q_policy.partially_reset(
                reset_type=self.agent_reset_type
            )

    def _combine(self):
        combine_array = []
        for i in range(len(self.pareto_pointset["area"])):
            point = [self.pareto_pointset["area"][i], self.pareto_pointset["delay"][i]]
            combine_array.append(point)
        return np.array(combine_array)

    def process_and_log_pareto(self, episode_num, episode_area, episode_delay):
        # 1. compute pareto pointset
        area_list, delay_list = episode_area, episode_delay
        area_list.extend(self.pareto_pointset["area"])
        delay_list.extend(self.pareto_pointset["delay"])
        data_points = pd.DataFrame(
            {
                "area": area_list,
                "delay": delay_list
            }
        )
        pareto_mask = paretoset(data_points, sense=["min", "min"])
        pareto_points = data_points[pareto_mask]
        new_pareto_area_list = pareto_points["area"].values.tolist()
        new_pareto_delay_list = pareto_points["delay"].values.tolist()
        self.pareto_pointset["area"] = new_pareto_area_list
        self.pareto_pointset["delay"] = new_pareto_delay_list
        
        # 2. compute hypervolume given pareto set and reference point
        pareto_point_array = self._combine()
        hv = hypervolume(pareto_point_array)
        hv_value = hv.compute(self.reference_point)
        logger.tb_logger.add_scalar('hypervolume', hv_value, global_step=episode_num)
        logger.log(f"episode {episode_num}, hypervolume: {hv_value}")

        # 3. log pareto points
        fig1 = plt.figure()
        x = new_pareto_area_list
        y = new_pareto_delay_list
        f1 = plt.scatter(x, y, c='r')
        logger.tb_logger.add_figure('pareto points', fig1, global_step=episode_num)

    def log_and_save_pareto_points(self, ppas_dict, episode_num):
        save_data_dict = {}
        # save ppa_csv
        save_data_dict["testing_full_ppa"] = ppas_dict
        # compute pareto points
        area_list = ppas_dict["area"]
        delay_list = ppas_dict["delay"]
        data_points = pd.DataFrame(
            {
                "area": area_list,
                "delay": delay_list
            }
        )
        pareto_mask = paretoset(data_points, sense=["min", "min"])
        pareto_points = data_points[pareto_mask]
        true_pareto_area_list = pareto_points["area"].values.tolist()
        true_pareto_delay_list = pareto_points["delay"].values.tolist()

        combine_array = []
        for i in range(len(true_pareto_area_list)):
            point = [true_pareto_area_list[i], true_pareto_delay_list[i]]
            combine_array.append(point)
        hv = hypervolume(combine_array)
        hv_value = hv.compute(self.reference_point)
        # save hypervolume and log hypervolume
        save_data_dict["testing_hypervolume"] = hv_value
        logger.tb_logger.add_scalar('testing hypervolume', hv_value, global_step=episode_num)
        
        # save pareto points and log pareto points
        fig1 = plt.figure()
        x = true_pareto_area_list
        y = true_pareto_delay_list
        f1 = plt.scatter(x, y, c='r')
        logger.tb_logger.add_figure('testing pareto points', fig1, global_step=episode_num)

        save_data_dict["testing_pareto_points_area"] = true_pareto_area_list
        save_data_dict["testing_pareto_points_delay"] = true_pareto_delay_list
        
        return save_data_dict

    def run_episode(self, episode_num):
        episode_area = []
        episode_delay = []
        # init state 
        env_state, sel_index = self.env.reset()
        state = copy.deepcopy(env_state)
        for step in range(self.len_per_episode):
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'])
            elif self.store_type == "detail":        
                self.store_detail(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], policy_info['state_ct32'], policy_info['state_ct22'], next_state_policy_info['state_ct32'], next_state_policy_info['state_ct22'], rewards_dict)
            
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict, 
                next_state, action, info, policy_info
            )
            avg_ppa = rewards_dict['avg_ppa']
            episode_area.extend(rewards_dict["area"])
            episode_delay.extend(rewards_dict["delay"])
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_area, episode_delay)

class RNDDQNAlgorithm(DQNAlgorithm):
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        rnd_lr=3e-4,
        update_rnd_freq=10,
        int_reward_scale=1,
        evaluate_freq=5,
        evaluate_num=5,
        bonus_type="rnd", # rnd/noveld
        noveld_alpha=0.1,
        # n step q-learning
        n_step_num=5,
        **dqn_alg_kwargs
    ):
        super().__init__(
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            **dqn_alg_kwargs
        )
        # rnd model
        self.rnd_predictor = rnd_predictor
        self.rnd_target = rnd_target
        self.int_reward_run_mean_std = int_reward_run_mean_std
        self.rnd_lr = rnd_lr
        self.update_rnd_freq = update_rnd_freq
        self.int_reward_scale = int_reward_scale
        self.evaluate_freq = evaluate_freq
        self.evaluate_num = evaluate_num
        self.bonus_type = bonus_type
        self.noveld_alpha = noveld_alpha
        self.n_step_num = n_step_num
        # optimizer
        self.rnd_model_optimizer = self.optimizer_class(
            self.rnd_predictor.parameters(),
            lr=self.rnd_lr
        )
        # loss func
        self.rnd_loss = nn.MSELoss()    
        # log
        self.rnd_loss_item = 0.
        self.rnd_int_rewards = 0.
        self.rnd_ext_rewards = 0.
        # cosine similarity
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def update_reward_int_run_mean_std(self, rewards):
        mean, std, count = np.mean(rewards), np.std(rewards), len(rewards)
        self.int_reward_run_mean_std.update_from_moments(
            mean, std**2, count
        )

    def update_rnd_model(
        self, state_batch, state_mask
    ):
        loss = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            
            predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
            with torch.no_grad():
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
            # set_trace()
            loss[i] = self.rnd_loss(
                predict_value, target_value
            )
        loss = torch.mean(loss)
        self.rnd_model_optimizer.zero_grad()
        loss.backward()
        for param in self.rnd_predictor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.rnd_model_optimizer.step()
        # update log
        self.rnd_loss_item = loss.item()
        return loss

    def compute_int_rewards(
        self, state_batch, state_mask
    ):
        batch_size = len(state_batch)
        int_rewards = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            
            with torch.no_grad():
                predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
            # set_trace()
            int_rewards[i] = torch.sum(
                (predict_value - target_value)**2
            )
        return int_rewards

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_mask = torch.cat(batch.mask)
            next_state_mask = torch.cat(batch.next_state_mask)
            # update reward int run mean std
            self.update_reward_int_run_mean_std(
                reward_batch.cpu().numpy()
            )
            # compute reward int 
            int_rewards_batch = self.compute_int_rewards(
                next_state_batch, next_state_mask
            )
            if self.bonus_type == "noveld":
                int_rewards_last_state_batch = self.compute_int_rewards(
                    state_batch, state_mask
                )
                int_rewards_batch = int_rewards_batch - self.noveld_alpha * int_rewards_last_state_batch

            int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(
                state_batch, action_batch, state_mask
            )
            next_state_values = self.compute_values(
                next_state_batch, None, next_state_mask
            )
            target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

            loss = self.loss_fn(
                state_action_values.unsqueeze(1), 
                target_state_action_values.unsqueeze(1)
            )

            self.policy_optimizer.zero_grad()
            loss.backward()
            for param in self.q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()

            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
            }
            self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
            self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())

            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info

    def get_ppa_value(self):
        assert self.env.initial_state_pool_max_len > 0
        number_states = len(self.env.initial_state_pool)
        state_value = np.zeros(number_states)
        info_count = np.zeros(number_states)
        for i in range(number_states):
            avg_ppa = self.env.initial_state_pool[i]["ppa"]
            state_count = self.env.initial_state_pool[i]["count"]
            # upper confidence bound
            score = (5 - avg_ppa / self.env.ppa_scale) * 20 + 1 / math.sqrt(state_count)
            state_value[i] = score
            info_count[i] = state_count
        return state_value, info_count

    def _get_env_pool_value_novelty(self, value_type):
        assert self.env.initial_state_pool_max_len > 0
        number_states = len(self.env.initial_state_pool)
        states_batch = []
        states_mask = []
        if value_type == "ppa_value":
            state_value, info_count = self.get_ppa_value()
            return state_value, info_count
        for i in range(number_states):
            states_batch.append(
                torch.tensor(
                    self.env.initial_state_pool[i]["state"]
                )
            )
            states_mask.append(
                self.env.initial_state_pool[i]["state_mask"]
            )
        if value_type == "novelty":
            state_novelty = self.compute_int_rewards(states_batch, states_mask)
            return state_novelty.cpu().numpy()
        elif value_type == "value":
            state_value = self.compute_values(
                states_batch, None, states_mask
            )
            return state_value.cpu().numpy()
        elif value_type == "average_value":
            state_value = self.compute_values(
                states_batch, None, states_mask, is_average=True
            )
            return state_value.cpu().numpy()
        else:
            raise NotImplementedError

    def log_state_mutual_distances(self, state_mutual_distances):
        fig1 = plt.figure()
        f1 = plt.imshow(state_mutual_distances)
        number_states = state_mutual_distances.shape[0]
        # Loop over data dimensions and create text annotations.
        for i in range(number_states):
            for j in range(number_states):
                text = plt.text(j, i, state_mutual_distances[i, j],
                            ha="center", va="center", color="w")
        logger.tb_logger.add_figure('state mutual distance', fig1, global_step=self.total_steps)

    def log_env_pool(self):
        total_state_num = len(self.env.initial_state_pool)
        best_ppa_state_num = 0
        diverse_state_num = 0
        for i in range(total_state_num):
            if self.env.initial_state_pool[i]["state_type"] == "best_ppa":
                best_ppa_state_num += 1
            elif self.env.initial_state_pool[i]["state_type"] == "diverse":
                diverse_state_num += 1
        logger.tb_logger.add_scalar('env pool total num', total_state_num, global_step=self.total_steps)
        logger.tb_logger.add_scalar('env pool best ppa num', best_ppa_state_num, global_step=self.total_steps)
        logger.tb_logger.add_scalar('env pool diverse num', diverse_state_num, global_step=self.total_steps)

    def run_episode(self, episode_num):
        # reset state
        episode_area = []
        episode_delay = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask']) 
            elif self.store_type == "detail":        
                self.store_detail(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], policy_info['state_ct32'], policy_info['state_ct22'], next_state_policy_info['state_ct32'], next_state_policy_info['state_ct22'], rewards_dict)
                
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            avg_ppa = rewards_dict['avg_ppa']
            episode_area.extend(rewards_dict["area"])
            episode_delay.extend(rewards_dict["delay"])
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_area, episode_delay)

    def log_rnd_stats(
        self, info
    ):
        # int reward vs rewards
        logger.tb_logger.add_scalar('batch int rewards', self.rnd_int_rewards, global_step=self.total_steps)
        logger.tb_logger.add_scalar('batch ext rewards', self.rnd_ext_rewards, global_step=self.total_steps)
        logger.tb_logger.add_scalar('rnd loss', self.rnd_loss_item, global_step=self.total_steps)

"""
    add a baseline: VERL
"""
class VERLDQNAlgorithm(RNDDQNAlgorithm):
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        start_episodes=40,
        population_size=8,
        **rnddqn_alg_kwargs
    ):
        super().__init__(
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            rnd_predictor,
            rnd_target,
            int_reward_run_mean_std,
            **rnddqn_alg_kwargs
        )
        self.start_episodes = start_episodes
        self.population_size = population_size
        self.population_q = []
        self.population_target_q = []

    def initialize_q_population(self):
        pre_trained_q_model = copy.deepcopy(self.q_policy).cpu()
        pre_trained_target_q_model = copy.deepcopy(self.target_q_policy).cpu()
        self.population_q = [
            copy.deepcopy(pre_trained_q_model) for _ in range(self.population_size)
        ]
        self.population_target_q = [
            copy.deepcopy(pre_trained_target_q_model) for _ in range(self.population_size)
        ]

    def crossover(self, parent1, parent1_target, parent2, parent2_target):
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        child1_target, child2_target = copy.deepcopy(parent1_target), copy.deepcopy(parent2_target)
        
        for param1, param2, param1_target, param2_target in zip(
            child1.parameters(), child2.parameters(),
            child1_target.parameters(), child2_target.parameters()
        ):
            mask = torch.rand_like(param1) > 0.5
            temp = param1.data[mask].clone()
            temp_target = param1_target.data[mask].clone()
            param1.data[mask] = param2.data[mask]
            param2.data[mask] = temp
            param1_target.data[mask] = param2_target.data[mask]
            param2_target.data[mask] = temp_target
        return child1, child1_target, child2, child2_target

    def gaussian_mutation(self, individual, target_individual, mutation_rate=0.1, sigma=0.1):
        mutated_individual = copy.deepcopy(individual)
        mutated_target_individual = copy.deepcopy(target_individual)
        for param1, param2 in zip(mutated_individual.parameters(), mutated_target_individual.parameters()):
            mask = torch.rand_like(param1) < mutation_rate
            noise = torch.randn_like(param1) * sigma
            param1.data[mask] += noise[mask]
            param2.data[mask] += noise[mask]
        return mutated_individual, mutated_target_individual

    ### parallel ####
    def compute_values_offline(
        self, state_batch, action_batch, state_mask, q_policy
    ):
        q_policy = q_policy.to(self.device)
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        states = []
        for i in range(batch_size):
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            states.append(state.unsqueeze(0))
        states = torch.cat(states)
        # compute image state
        with torch.no_grad():
            if action_batch is not None:
                q_values = q_policy(states.float(), state_mask=state_mask)
                q_values = q_values.reshape(-1, (int(self.int_bit_width*2))*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
                for i in range(batch_size):
                    state_action_values[i] = q_values[i, action_batch[i]]
            else:
                q_values = q_policy(states.float(), is_target=True, state_mask=state_mask)
                for i in range(batch_size):
                    state_action_values[i] = q_values[i:i+1].max(1)[0].detach()
        return state_action_values

    def evaluate_and_select(self, intermediate_populations):
        # 1. sample a batch
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_mask = torch.cat(batch.mask)
        next_state_mask = torch.cat(batch.next_state_mask)
        
        # 2. compute TD error for all populations
        td_errors = []
        for q, target_q in intermediate_populations:
            train_reward_batch = reward_batch.to(self.device)
            state_action_values = self.compute_values_offline(
                state_batch, action_batch, state_mask, q
            )
            next_state_values = self.compute_values_offline(
                next_state_batch, None, next_state_mask, target_q
            )
            target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

            td_error = torch.mean(
                (target_state_action_values - state_action_values) ** 2
            )
            td_errors.append(td_error.item())
            q.cpu()
            target_q.cpu()
        # 3. select the top k population
        sorting_indexes = np.argsort(td_errors)
        sel_top_indexes = sorting_indexes[:self.population_size]
        sel_populations = [intermediate_populations[i] for i in sel_top_indexes]

        # 4. update populations
        sel_populations_q = []
        sel_populations_target_q = []
        for q, target_q in sel_populations:
            sel_populations_q.append(q)
            sel_populations_target_q.append(target_q)
        self.population_q = sel_populations_q
        self.population_target_q = sel_populations_target_q

    def q_net_variation(self):
        # 1. generation 
        intermediate_populations = []
        # mutation
        for i in range(len(self.population_q)):
            mutated_individual, mutated_target_individual = self.gaussian_mutation(
                self.population_q[i], self.population_target_q[i]
            )
            intermediate_populations.extend(
                [
                    (self.population_q[i], self.population_target_q[i]),
                    (mutated_individual, mutated_target_individual)
                ]
            )
        # crossover
        for num in range(len(self.population_q) * 3):
            sel_index1, sel_index2 = np.random.choice(
                len(self.population_q), 2
            )
            child1, child1_target, child2, child2_target = self.crossover(
                self.population_q[sel_index1], self.population_target_q[sel_index1],
                self.population_q[sel_index2], self.population_target_q[sel_index2]
            )
            intermediate_populations.extend(
                [
                    (child1, child1_target),
                    (child2, child2_target)
                ]
            )
        # 2. evaluation and selection
        self.evaluate_and_select(intermediate_populations)

        # 3. update elite q policy model 
        cur_best_q = self.population_q[0]
        cur_best_target_q = self.population_target_q[0]
        self.q_policy = copy.deepcopy(cur_best_q).to(self.device)
        self.target_q_policy = copy.deepcopy(cur_best_target_q).to(self.device)

    def run_experiments(self):
        # 1. warm satrt up, pre-train Q and Target Q replay buffer
        for episode_num in range(self.start_episodes):
            self.run_episode(episode_num)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num)
        # 2. initialize a population of Q networks
        self.initialize_q_population()
        # 3. iterate between GA and RL
        for episode_num in range(self.total_episodes):
            # GA mutation
            self.q_net_variation()
            # RL optimization
            self.run_episode(episode_num+self.start_episodes)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num+self.start_episodes)
            # RL injection
            self.population_q[-1] = copy.deepcopy(self.q_policy).cpu()
            self.population_target_q[-1] = copy.deepcopy(self.target_q_policy).cpu()
        self.end_experiments(episode_num + self.start_episodes)

"""
    add a baseline: VERL
"""

class NStepRNDDQNAlgorithm(RNDDQNAlgorithm):
    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        is_last=False
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        self.replay_memory.push(
            is_last,
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1,-1),
            next_state_mask.reshape(1,-1)
            # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            # rewards_dict
        )

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            list_of_listtransitions = self.replay_memory.sample(self.batch_size)
            next_state_batch = []
            state_batch = []
            action_batch = []
            reward_batch = []
            state_mask = []
            next_state_mask = []

            for listtransitions in list_of_listtransitions:
                batch = Transition(*zip(*listtransitions)) # namedtuple, dict of tuple

                next_state_batch.append(torch.cat(batch.next_state))
                state_batch.append(torch.cat(batch.state))
                action_batch.append(torch.cat(batch.action))
                reward_batch.append(torch.cat(batch.reward))
                state_mask.append(torch.cat(batch.mask))
                next_state_mask.append(torch.cat(batch.next_state_mask))
   
            next_state_batch = torch.cat(next_state_batch)
            state_batch = torch.cat(state_batch)
            action_batch = torch.cat(action_batch)
            reward_batch = torch.cat(reward_batch)
            state_mask = torch.cat(state_mask)
            next_state_mask = torch.cat(next_state_mask)

            # update reward int run mean std
            self.update_reward_int_run_mean_std(
                reward_batch.cpu().numpy()
            )
            # compute reward int 
            int_rewards_batch = self.compute_int_rewards(
                next_state_batch, next_state_mask
            )
            if self.bonus_type == "noveld":
                int_rewards_last_state_batch = self.compute_int_rewards(
                    state_batch, state_mask
                )
                int_rewards_batch = int_rewards_batch - self.noveld_alpha * int_rewards_last_state_batch

            int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(
                state_batch, action_batch, state_mask
            )
            next_state_values = self.compute_values(
                next_state_batch, None, next_state_mask
            )
            target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

            loss = self.loss_fn(
                state_action_values.unsqueeze(1), 
                target_state_action_values.unsqueeze(1)
            )

            self.policy_optimizer.zero_grad()
            loss.backward()
            for param in self.q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()

            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
            }
            self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
            self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())

            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info

    def run_episode(self, episode_num):
        # reset state
        episode_area = []
        episode_delay = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
                
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])

            # store data
            is_last = False
            if (step+1) % self.n_step_num == 0:
                is_last = True
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], is_last=is_last)
            else:
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], is_last=is_last)
            # update q policy
            loss, info = self.update_q()
            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            avg_ppa = rewards_dict['avg_ppa']
            episode_area.extend(rewards_dict["area"])
            episode_delay.extend(rewards_dict["delay"])
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_area, episode_delay)

class MultiObjRNDDQNAlgorithm(RNDDQNAlgorithm):
    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the best ppa state into the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    self.env.initial_state_pool.append(
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

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        area_reward, delay_reward
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        self.replay_memory.push(
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1,-1),
            next_state_mask.reshape(1,-1),
            torch.tensor([area_reward]),
            torch.tensor([delay_reward])
            # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            # rewards_dict
        )

    def compute_values(
        self, state_batch, action_batch, state_mask, is_average=False
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        state_action_area_values = torch.zeros(batch_size, device=self.device)
        state_action_delay_values = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            # compute image state
            if action_batch is not None:
                # reshape 有问题************
                q_area, q_delay, q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i])
                q_area = q_area.reshape((int(self.int_bit_width*2))*4)
                q_delay = q_delay.reshape((int(self.int_bit_width*2))*4)
                q_values = q_values.reshape((int(self.int_bit_width*2))*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
                state_action_values[i] = q_values[action_batch[i]]
                state_action_area_values[i] = q_area[action_batch[i]]
                state_action_delay_values[i] = q_delay[action_batch[i]]
            else:
                q_area, q_delay, q_values = self.target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
                if self.multiobj_type == "pure_max":
                    state_action_values[i] = q_values.max(1)[0].detach()
                    state_action_area_values[i] = q_area.max(1)[0].detach()
                    state_action_delay_values[i] = q_delay.max(1)[0].detach()
                elif self.multiobj_type == "weight_max":
                    state_action_values[i] = q_values.max(1)[0].detach()
                    cur_q_values = q_values.reshape((int(self.int_bit_width*2))*4)
                    index = torch.argmax(cur_q_values)
                    state_action_area_values[i] = q_area.squeeze()[index].detach()
                    state_action_delay_values[i] = q_delay.squeeze()[index].detach()         
        return state_action_values, state_action_area_values, state_action_delay_values

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = MultiObjTransition(*zip(*transitions))

            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_mask = torch.cat(batch.mask)
            next_state_mask = torch.cat(batch.next_state_mask)
            area_reward_batch = torch.cat(batch.area_reward)
            delay_reward_batch = torch.cat(batch.delay_reward)
            
            # update reward int run mean std
            self.update_reward_int_run_mean_std(
                reward_batch.cpu().numpy()
            )
            # compute reward int 
            int_rewards_batch = self.compute_int_rewards(
                next_state_batch, next_state_mask
            )
            int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            # TODO: int reward 会不会有问题？
            train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            train_area_reward_batch = area_reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            train_delay_reward_batch = delay_reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values, state_action_area_values, state_action_delay_values = self.compute_values(
                state_batch, action_batch, state_mask
            )
            next_state_values, next_state_area_values, next_state_delay_values = self.compute_values(
                next_state_batch, None, next_state_mask
            )

            target_state_action_values = (next_state_values * self.gamma) + train_reward_batch
            target_state_action_area_values = (next_state_area_values * self.gamma) + train_area_reward_batch
            target_state_action_delay_values = (next_state_delay_values * self.gamma) + train_delay_reward_batch
            
            area_loss = self.loss_fn(
                state_action_area_values.unsqueeze(1), 
                target_state_action_area_values.unsqueeze(1)
            )
            delay_loss = self.loss_fn(
                state_action_delay_values.unsqueeze(1), 
                target_state_action_delay_values.unsqueeze(1)
            )
            loss = area_loss + delay_loss
            self.policy_optimizer.zero_grad()
            loss.backward()
            for param in self.q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()

            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float()),
                "q_area_values": state_action_area_values.detach().cpu().numpy(),
                "q_delay_values": state_action_delay_values.detach().cpu().numpy()
            }
            self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
            self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())

            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info

    def log_multi_obj_stats(self, rewards_dict, info):
        # log multi obj q values
        if 'q_area_values' in info.keys():
            q_area_values = np.mean(info['q_area_values'])
            q_delay_values = np.mean(info['q_delay_values'])
            logger.tb_logger.add_scalar('q_area_values', q_area_values, global_step=self.total_steps)
            logger.tb_logger.add_scalar('q_delay_values', q_delay_values, global_step=self.total_steps)

        # log multi obj reward
        logger.tb_logger.add_scalar('normalize_area_no_scale', rewards_dict['normalize_area_no_scale'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('normalize_delay_no_scale', rewards_dict['normalize_delay_no_scale'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('normalize_area', rewards_dict['normalize_area'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('normalize_delay', rewards_dict['normalize_delay'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('area_reward', rewards_dict['area_reward'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('delay_reward', rewards_dict['delay_reward'], global_step=self.total_steps)
        
    def run_episode(self, episode_num):
        # reset state
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if len(self.env.initial_state_pool) >= 2:
            state_mutual_distances = self.env.get_mutual_distance()
            self.log_state_mutual_distances(state_mutual_distances)
            self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # environment interaction, select action based on the weighted q value
            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # store data
            self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict["area_reward"], rewards_dict["delay_reward"])
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            # log multi obj stats
            self.log_multi_obj_stats(
                rewards_dict, info
            )
            self.log_rnd_stats(info)
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )

class PreferenceDrivenMultiObjRNDDQNAlgorithm(RNDDQNAlgorithm):
    def sample_weight(self):
        w = np.random.uniform()
        weight_area = w * 5
        weight_delay = 5 - weight_area

        return [weight_area, weight_delay]

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        area_reward, delay_reward, weight
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        weight = np.array(weight).reshape(1,-1)

        self.replay_memory.push(
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1,-1),
            next_state_mask.reshape(1,-1),
            torch.tensor([area_reward]),
            torch.tensor([delay_reward]),
            torch.tensor(weight)
            # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            # rewards_dict
        )

    def compute_cosine_similarity(
            self,  cur_weight_condition, q_area, q_delay
    ):
        # cat q area q delay
        concat_q = torch.cat(
            (q_area.detach(), q_delay.detach()), 1
        )
        # repeat weights
        weights = cur_weight_condition.repeat(
            q_area.shape[0], 1
        ).to(self.device)

        cos_sim = self.cos(
            weights, concat_q
        )
        return cos_sim

    def compute_values(
        self, state_batch, action_batch, state_mask, weight_batch, is_average=False
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        state_action_area_values = torch.zeros(batch_size, device=self.device)
        state_action_delay_values = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            cur_weight_condition = weight_batch[i:i+1].float()
            self.q_policy.wallace_area = int(cur_weight_condition[0,0])
            self.q_policy.wallace_delay = int(cur_weight_condition[0,1])
            self.target_q_policy.wallace_area = int(cur_weight_condition[0,0])
            self.target_q_policy.wallace_delay = int(cur_weight_condition[0,1])
            # compute image state
            if action_batch is not None:
                # reshape 有问题************
                q_area, q_delay, q_values = self.q_policy(state.unsqueeze(0).float(), cur_weight_condition, state_mask=state_mask[i])
                q_area = q_area.reshape((int(self.int_bit_width*2))*4)
                q_delay = q_delay.reshape((int(self.int_bit_width*2))*4)
                q_values = q_values.reshape((int(self.int_bit_width*2))*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
                state_action_values[i] = q_values[action_batch[i]]
                state_action_area_values[i] = q_area[action_batch[i]]
                state_action_delay_values[i] = q_delay[action_batch[i]]
            else:
                q_area, q_delay, q_values = self.target_q_policy(state.unsqueeze(0).float(), cur_weight_condition, is_target=True, state_mask=state_mask[i])
                if self.multiobj_type == "pure_max":
                    state_action_values[i] = q_values.max(1)[0].detach()
                    state_action_area_values[i] = q_area.max(1)[0].detach()
                    state_action_delay_values[i] = q_delay.max(1)[0].detach()
                elif self.multiobj_type == "weight_max":
                    state_action_values[i] = q_values.max(1)[0].detach()
                    cur_q_values = q_values.reshape((int(self.int_bit_width*2))*4)
                    index = torch.argmax(cur_q_values)
                    state_action_area_values[i] = q_area.squeeze()[index].detach()
                    state_action_delay_values[i] = q_delay.squeeze()[index].detach()
                elif self.multiobj_type == "preference_driven":
                    # compute cosine similarity
                    cosine_similarity = self.compute_cosine_similarity(
                        cur_weight_condition, q_area, q_delay
                    )
                    pd_q_values = cosine_similarity * q_values
                    state_action_values[i] = pd_q_values.max(1)[0].detach()
                    cur_q_values = pd_q_values.reshape((int(self.int_bit_width*2))*4)
                    index = torch.argmax(cur_q_values)
                    state_action_area_values[i] = q_area.squeeze()[index].detach()
                    state_action_delay_values[i] = q_delay.squeeze()[index].detach()
                    
        return state_action_values, state_action_area_values, state_action_delay_values

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = PDMultiObjTransition(*zip(*transitions))

            next_state_batch = torch.cat(batch.next_state)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_mask = torch.cat(batch.mask)
            next_state_mask = torch.cat(batch.next_state_mask)
            area_reward_batch = torch.cat(batch.area_reward)
            delay_reward_batch = torch.cat(batch.delay_reward)
            weight_batch = torch.cat(batch.weight)
            
            # update reward int run mean std
            self.update_reward_int_run_mean_std(
                reward_batch.cpu().numpy()
            )
            # compute reward int 
            int_rewards_batch = self.compute_int_rewards(
                next_state_batch, next_state_mask
            )
            if self.bonus_type == "noveld":
                int_rewards_last_state_batch = self.compute_int_rewards(
                    state_batch, state_mask
                )
                int_rewards_batch = int_rewards_batch - self.noveld_alpha * int_rewards_last_state_batch

            int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            # TODO: int reward 会不会有问题？
            train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            train_area_reward_batch = area_reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            train_delay_reward_batch = delay_reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values, state_action_area_values, state_action_delay_values = self.compute_values(
                state_batch, action_batch, state_mask, weight_batch
            )
            next_state_values, next_state_area_values, next_state_delay_values = self.compute_values(
                next_state_batch, None, next_state_mask, weight_batch
            )

            target_state_action_values = (next_state_values * self.gamma) + train_reward_batch
            target_state_action_area_values = (next_state_area_values * self.gamma) + train_area_reward_batch
            target_state_action_delay_values = (next_state_delay_values * self.gamma) + train_delay_reward_batch
            
            area_loss = self.loss_fn(
                state_action_area_values.unsqueeze(1), 
                target_state_action_area_values.unsqueeze(1)
            )
            delay_loss = self.loss_fn(
                state_action_delay_values.unsqueeze(1), 
                target_state_action_delay_values.unsqueeze(1)
            )
            loss = area_loss + delay_loss
            self.policy_optimizer.zero_grad()
            loss.backward()
            for param in self.q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()

            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float()),
                "q_area_values": state_action_area_values.detach().cpu().numpy(),
                "q_delay_values": state_action_delay_values.detach().cpu().numpy()
            }
            self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
            self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())

            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info
    
    def run_episode(self, episode_num):
        # 0. sampling a preference --> weight condition -> env 的weight，policy 的weight
        weight = self.sample_weight()
        self.env.weight_area = weight[0]
        self.env.weight_delay = weight[1]
        self.q_policy.wallace_area = weight[0]
        self.q_policy.wallace_delay = weight[1]
        self.target_q_policy.wallace_area = weight[0]
        self.target_q_policy.wallace_delay = weight[1]
        
        # 1. reset state
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # environment interaction, select action based on the weighted q value
            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps,
                weight,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, weight,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # store data
            self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict["area_reward"], rewards_dict["delay_reward"], weight)
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            # log multi obj stats
            self.log_multi_obj_stats(
                rewards_dict, info
            )
            self.log_rnd_stats(info)
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )

    def log_multi_obj_stats(self, rewards_dict, info):
        # log multi obj q values
        if 'q_area_values' in info.keys():
            q_area_values = np.mean(info['q_area_values'])
            q_delay_values = np.mean(info['q_delay_values'])
            logger.tb_logger.add_scalar('q_area_values', q_area_values, global_step=self.total_steps)
            logger.tb_logger.add_scalar('q_delay_values', q_delay_values, global_step=self.total_steps)

        # log multi obj reward
        logger.tb_logger.add_scalar('normalize_area_no_scale', rewards_dict['normalize_area_no_scale'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('normalize_delay_no_scale', rewards_dict['normalize_delay_no_scale'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('normalize_area', rewards_dict['normalize_area'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('normalize_delay', rewards_dict['normalize_delay'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('area_reward', rewards_dict['area_reward'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('delay_reward', rewards_dict['delay_reward'], global_step=self.total_steps)
        
class ThreeDRNDDQNAlgorithm(RNDDQNAlgorithm):
    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        compressor_state=np.zeros((2,int(self.int_bit_width*2)))
        compressor_state[0]=np.sum(state[0],axis=0)
        compressor_state[1]=np.sum(state[1],axis=0)
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the best ppa state into the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    self.env.initial_state_pool.append(
                        {
                            "state": compressor_state,
                            "threed_state": copy.deepcopy(state),
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

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,self.MAX_STAGE_NUM,int(self.int_bit_width*2)))
        self.replay_memory.push(
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1,-1),
            next_state_mask.reshape(1,-1)
            # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            # rewards_dict
        )

    def compute_values(
        self, state_batch, action_batch, state_mask, is_average=False
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        # 并行版本
        if action_batch is not None:
            # TODO: 核对state mask
            q_values = self.q_policy(
                state_batch.float(),
                state_mask=state_mask
            )
            for i in range(batch_size):
                state_action_values[i] = q_values[i, action_batch[i]]
        else:
            q_values = self.target_q_policy(
                state_batch.float(), 
                state_mask=state_mask
            )
            for i in range(batch_size):
                state_action_values[i] = q_values[i:i+1].max(1)[0].detach()
            
        # 串行
        # for i in range(batch_size):
        #     state = state_batch[i]
        #     # compute image state
        #     if action_batch is not None:
        #         # reshape 有问题************
        #         q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4*self.MAX_STAGE_NUM)           
        #         # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
        #         state_action_values[i] = q_values[action_batch[i]]
        #     else:
        #         q_values = self.target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
                
        #         # q_values = self.target_q_policy(state.unsqueeze(0))                
        #         # if is_average:
        #         #     q_values = (q_values + 1000).detach()
        #         #     num = torch.count_nonzero(q_values)
        #         #     state_action_values[i] = q_values.sum() / (num+1e-4)
        #         if self.is_double_q:
        #             current_q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
        #             index = torch.argmax(current_q_values)
        #             state_action_values[i] = q_values.squeeze()[index].detach()
        #         else:      
        #             state_action_values[i] = q_values.max(1)[0].detach()
        return state_action_values

    def compute_int_rewards(
        self, state_batch, state_mask
    ):
        batch_size = len(state_batch)
        int_rewards = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            state = state_batch[i]
            
            with torch.no_grad():
                predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4*self.MAX_STAGE_NUM)
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4*self.MAX_STAGE_NUM)
            # set_trace()
            int_rewards[i] = torch.sum(
                (predict_value - target_value)**2
            )
        return int_rewards
    
    def update_rnd_model(
        self, state_batch, state_mask
    ):
        loss = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            state = state_batch[i]
            predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4*self.MAX_STAGE_NUM)
            with torch.no_grad():
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4*self.MAX_STAGE_NUM)
            # set_trace()
            loss[i] = self.rnd_loss(
                predict_value, target_value
            )
        loss = torch.mean(loss)
        self.rnd_model_optimizer.zero_grad()
        loss.backward()
        for param in self.rnd_predictor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.rnd_model_optimizer.step()
        # update log
        self.rnd_loss_item = loss.item()
        return loss

    def run_experiments(self):
        for episode_num in range(self.total_episodes):
            self.run_episode(episode_num)
            if self.evaluate_freq > 0 and (episode_num+1) % self.evaluate_freq == 0:
                self.evaluate(episode_num)
        self.end_experiments(episode_num)

    def evaluate(self, episode_num):
        evaluate_num = min(self.evaluate_num, len(self.env.initial_state_pool))
        ppa_list = []
        for i in range(1, evaluate_num+1):
            test_state = copy.deepcopy(
                self.env.initial_state_pool[-i]["threed_state"]
            )
            ct32, ct22, stage_num = self.env.decompose_compressor_tree_v2(test_state)
            rewards_dict = self.env.get_reward()
            avg_area = np.mean(rewards_dict['area'])
            avg_delay = np.mean(rewards_dict['delay'])
            avg_ppa = self.env.weight_area * avg_area / self.env.wallace_area + \
                self.env.weight_delay * avg_delay / self.env.wallace_delay
            avg_ppa = avg_ppa * self.env.ppa_scale
            ppa_list.append(avg_ppa)
        
        best_ppa = np.min(ppa_list)
        logger.tb_logger.add_scalar('true best ppa found', best_ppa, global_step=episode_num)

    def log_action_stats(self, action):
        action_stage= int(int(action) // (self.action_num*int(2*self.int_bit_width)))
        action_column = int((int(action) % (self.action_num*int(2*self.int_bit_width))) // self.action_num)
        action_type = int( (int(action) % (self.action_num*int(2*self.int_bit_width)))   %  self.action_num)

        logger.tb_logger.add_scalar('3d action stage', action_stage, global_step=self.total_steps)
        logger.tb_logger.add_scalar('3d action column', action_column, global_step=self.total_steps)
        logger.tb_logger.add_scalar('3d action type', action_type, global_step=self.total_steps)

class MBRLRNDDQNAlgorithm(RNDDQNAlgorithm):
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        # model-based kwargs
        ppa_model, 
        imagined_replay_memory,
        ppa_model_path=None,
        # sr model kwargs
        is_sr_model=False,
        sr_model_path=None,
        # dsr model kwargs
        dsr_model_npy_path=None,
        OperatorList = ['*', '+', '-', '/', '^', 'ln', 'exp', 'c'],
        # imagined_env,
        warm_start_steps=500,
        # imagine sample kwargs
        num_random_sample=64,
        num_sample=20,
        depth=5,
        imagine_data_freq=25,
        train_q_steps=5,
        trajectory_step_num=5,
        model_env_iterative_epi_num=20,
        start_episodes=40,
        # train model kwargs
        train_model_freq=100,
        train_model_start_steps=200,
        train_model_finetune_steps=20,
        train_model_batch_size=256,
        model_lr=1e-3,
        # evaluate imagine model kwargs
        evaluate_imagine_state_num=5,
        evaluate_imagine_state_freq=10,
        imagine_is_softmax=False,
        # train policy kwargs
        real_data_ratio=0.2,
        num_train_per_step=4,
        # mbrl_type
        mbrl_type="inter_mbrl", # [dyna, inter_mbrl]
        use_dyna_data=True,
        # lr decay
        lr_decay=False,
        lr_decay_step=5,
        lr_decay_rate=0.96,
        lr_decay_step_finetune=1,
        **rnd_dqn_alg_kwargs
    ):
        super(MBRLRNDDQNAlgorithm, self).__init__(
            env, q_policy, target_q_policy, replay_memory,
            rnd_predictor, rnd_target, int_reward_run_mean_std,
            **rnd_dqn_alg_kwargs
        )

        # model-based kwargs
        self.ppa_model = ppa_model
        self.imagined_replay_memory = imagined_replay_memory
        self.ppa_model_path = ppa_model_path
        self.has_load_ppa_model = False
        # self.imagined_env = imagined_env
        # sr model kwargs
        self.is_sr_model = is_sr_model
        self.sr_model_path = sr_model_path
        # dsr model kwargs
        self.dsr_model_npy_path = dsr_model_npy_path
        if self.dsr_model_npy_path is not None:
            self.dsr_sequence = []
            self.dsr_length = []
            for npy_path in self.dsr_model_npy_path:
                dsr_model_npy_data = torch.load(npy_path)
                self.dsr_sequence.append(dsr_model_npy_data["sequence"])
                self.dsr_length.append(dsr_model_npy_data["length"])
            for i in range(DSRFeatureDim[self.bit_width]):
                OperatorList.append(f'var_x{i}')
            self.OperatorList = OperatorList
            self.operators = Operators(self.OperatorList, self.device)

        self.warm_start_steps = warm_start_steps
        self.num_random_sample = num_random_sample
        self.train_model_freq = train_model_freq
        self.imagine_data_freq = imagine_data_freq
        self.num_sample = num_sample
        self.depth = depth
        self.real_data_ratio = real_data_ratio
        self.num_train_per_step = num_train_per_step
        self.train_model_start_steps = train_model_start_steps
        self.train_model_finetune_steps = train_model_finetune_steps
        self.model_lr = model_lr
        self.train_q_steps = train_q_steps
        self.train_model_batch_size = train_model_batch_size
        self.imagine_is_softmax = imagine_is_softmax    
        self.mbrl_type = mbrl_type
        self.trajectory_step_num = trajectory_step_num
        self.model_env_iterative_epi_num = model_env_iterative_epi_num
        self.start_episodes = start_episodes
        self.use_dyna_data = use_dyna_data

        # lr decay
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step_finetune = lr_decay_step_finetune

        self.is_start_model = False
        self.is_first_model_train = True
        self.is_first_run_model = True

        # evaluate imagine state pool model 
        self.evaluate_imagine_state_num = evaluate_imagine_state_num
        self.evaluate_imagine_state_freq = evaluate_imagine_state_freq

        # optimizer
        self.ppa_model_optimizer = self.optimizer_class(
            self.ppa_model.parameters(),
            lr=self.model_lr
        )
        if self.lr_decay:
            self.model_lr_scheduler = lr_scheduler.StepLR(
                self.ppa_model_optimizer,
                self.lr_decay_step,
                gamma=self.lr_decay_rate
            )
        self.mse_loss = nn.MSELoss()

        self.train_model_total_steps = 0
        self.imagine_found_best_ppa = 10000

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        normalize_area, normalize_delay,
        is_model_evaluation=False
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        if is_model_evaluation:
            self.imagined_replay_memory.push(
                torch.tensor(state),
                action,
                torch.tensor(next_state),
                torch.tensor([reward]),
                mask.reshape(1,-1),
                next_state_mask.reshape(1,-1),
                torch.tensor([normalize_area]),
                torch.tensor([normalize_delay])                
            )
        else:
            self.replay_memory.push(
                torch.tensor(state),
                action,
                torch.tensor(next_state),
                torch.tensor([reward]),
                mask.reshape(1,-1),
                next_state_mask.reshape(1,-1),
                torch.tensor([normalize_area]),
                torch.tensor([normalize_delay])
                # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
                # rewards_dict
            )

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask, is_model_evaluation=False):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    if is_model_evaluation:
                        self.env.imagined_initial_state_pool.append(
                            {
                                "state": copy.deepcopy(state),
                                "area": 0,
                                "delay": 0,
                                "ppa": rewards_dict['avg_ppa'],
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "best_ppa",
                                "normalize_area": rewards_dict["normalize_area"],
                                "normalize_delay": rewards_dict["normalize_delay"]
                            }
                        )
                    else:    
                        # push the best ppa state into the initial pool
                        avg_area = np.mean(rewards_dict['area'])
                        avg_delay = np.mean(rewards_dict['delay'])
                        self.env.initial_state_pool.append(
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
        # best ppa info
        if not is_model_evaluation:
            if self.found_best_info["found_best_ppa"] > rewards_dict['avg_ppa']:
                self.found_best_info["found_best_ppa"] = rewards_dict['avg_ppa']
                self.found_best_info["found_best_state"] = copy.deepcopy(state)
                self.found_best_info["found_best_area"] = np.mean(rewards_dict['area']) 
                self.found_best_info["found_best_delay"] = np.mean(rewards_dict['delay'])
    
    def combine_batch(self, imagined_batch_transitions, real_batch_transitions):
        imagined_batch = MBRLTransition(*zip(*imagined_batch_transitions))
        real_batch = MBRLTransition(*zip(*real_batch_transitions))
        
        real_next_state_batch = torch.cat(real_batch.next_state)
        real_state_batch = torch.cat(real_batch.state)
        real_action_batch = torch.cat(real_batch.action)
        real_reward_batch = torch.cat(real_batch.reward)
        real_state_mask = torch.cat(real_batch.mask)
        real_next_state_mask = torch.cat(real_batch.next_state_mask)        

        imagined_next_state_batch = torch.cat(imagined_batch.next_state)
        imagined_state_batch = torch.cat(imagined_batch.state)
        imagined_action_batch = torch.cat(imagined_batch.action)
        imagined_reward_batch = torch.cat(imagined_batch.reward)
        imagined_state_mask = torch.cat(imagined_batch.mask)
        imagined_next_state_mask = torch.cat(imagined_batch.next_state_mask)        

        next_state_batch = torch.cat(
            (real_next_state_batch, imagined_next_state_batch)
        )
        state_batch = torch.cat(
            (real_state_batch, imagined_state_batch)
        )
        action_batch = torch.cat(
            (real_action_batch, imagined_action_batch)
        )
        reward_batch = torch.cat(
            (real_reward_batch, imagined_reward_batch)
        )
        state_mask = torch.cat(
            (real_state_mask, imagined_state_mask)
        )
        next_state_mask = torch.cat(
            (real_next_state_mask, imagined_next_state_mask)
        )
        return next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask
    
    ### parallel ####
    def compute_values(
        self, state_batch, action_batch, state_mask
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        states = []
        for i in range(batch_size):
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            states.append(state.unsqueeze(0))
        states = torch.cat(states)
        # compute image state
        if action_batch is not None:
            q_values = self.q_policy(states.float(), state_mask=state_mask)
            q_values = q_values.reshape(-1, (int(self.int_bit_width*2))*4)           
            # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
            for i in range(batch_size):
                state_action_values[i] = q_values[i, action_batch[i]]
        else:
            q_values = self.target_q_policy(states.float(), is_target=True, state_mask=state_mask)
            for i in range(batch_size):
                state_action_values[i] = q_values[i:i+1].max(1)[0].detach()
        return state_action_values
    
    ### parallel ####
    def compute_int_rewards(self, state_batch, state_mask):
        batch_size = len(state_batch)
        int_rewards = torch.zeros(batch_size, device=self.device)
        states = []
        for i in range(batch_size):
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            states.append(state.unsqueeze(0))
        states = torch.cat(states)
        with torch.no_grad():
            predict_value = self.rnd_predictor(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (int(self.int_bit_width*2))*4)
            target_value = self.rnd_target(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (int(self.int_bit_width*2))*4)
        # set_trace()
        int_rewards = torch.sum(
            (predict_value - target_value)**2, dim=1
        )
        return int_rewards

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else: 
            if not self.is_start_model or len(self.imagined_replay_memory) == 0:
                # updating q using pure real data
                transitions = self.replay_memory.sample(self.batch_size)
                batch = MBRLTransition(*zip(*transitions))
                next_state_batch = torch.cat(batch.next_state)
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_mask = torch.cat(batch.mask)
                next_state_mask = torch.cat(batch.next_state_mask)
                # update reward int run mean std
                self.update_reward_int_run_mean_std(
                    reward_batch.cpu().numpy()
                )

                # compute reward int 
                int_rewards_batch = self.compute_int_rewards(
                    next_state_batch, next_state_mask
                )
                int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = self.compute_values(
                    state_batch, action_batch, state_mask
                )
                next_state_values = self.compute_values(
                    next_state_batch, None, next_state_mask
                )
                target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                loss = self.loss_fn(
                    state_action_values.unsqueeze(1), 
                    target_state_action_values.unsqueeze(1)
                )

                self.policy_optimizer.zero_grad()
                loss.backward()
                for param in self.q_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.policy_optimizer.step()

                info = {
                    "q_values": state_action_values.detach().cpu().numpy(),
                    "target_q_values": target_state_action_values.detach().cpu().numpy(),
                    "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
                }
                self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())
            else:
                for _ in range(self.train_q_steps):
                    imagined_batch_size = min(len(self.imagined_replay_memory), int((1-self.real_data_ratio)*self.batch_size))
                    real_batch_size = self.batch_size - imagined_batch_size
                    imagined_batch_transitions = self.imagined_replay_memory.sample(imagined_batch_size)
                    real_batch_transitions = self.replay_memory.sample(real_batch_size)
                    next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask = self.combine_batch(
                        imagined_batch_transitions, real_batch_transitions
                    )
                    # update reward int run mean std
                    self.update_reward_int_run_mean_std(
                        reward_batch.cpu().numpy()
                    )

                    # compute reward int 
                    int_rewards_batch = self.compute_int_rewards(
                        next_state_batch, next_state_mask
                    )
                    int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                    train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = self.compute_values(
                        state_batch, action_batch, state_mask
                    )
                    next_state_values = self.compute_values(
                        next_state_batch, None, next_state_mask
                    )
                    target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                    loss = self.loss_fn(
                        state_action_values.unsqueeze(1), 
                        target_state_action_values.unsqueeze(1)
                    )

                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    for param in self.q_policy.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.policy_optimizer.step()

                    info = {
                        "q_values": state_action_values.detach().cpu().numpy(),
                        "target_q_values": target_state_action_values.detach().cpu().numpy(),
                        "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
                    }
                    self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                    self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())            


            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info

    def _imagine_model_data(self):
        # set_trace()
        if self.is_start_model:
            for _ in range(self.num_sample):
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
                for _ in range(self.depth):
                    action, policy_info = self.q_policy.select_action(
                        torch.tensor(state), 
                        self.total_steps, 
                        deterministic=self.deterministic,
                        is_softmax=self.is_softmax
                    )
                    # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                    next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                    _, next_state_policy_info = self.q_policy.select_action(
                        torch.tensor(next_state), self.total_steps, 
                        deterministic=self.deterministic,
                        is_softmax=self.is_softmax
                    )
                    # 2.2 store real/imagined data
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                    
                    # 2.3 update real/imagined initial state pool
                    self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                    state = copy.deepcopy(next_state)
                    
    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        # 2. sampling data
        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # 2.1 sampling real environment interaction
            is_model_evaluation = False

            """
                imagine data inter each episode
            """
            if self.is_start_model:
                if self.mbrl_type == "inter_mbrl":
                    if step <= self.real_data_ratio * self.len_per_episode:
                        is_model_evaluation = False
                    else:
                        is_model_evaluation = True
                elif self.mbrl_type == "dyna":
                    is_model_evaluation = False
            """
                imagine data inter each episode
            """

            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
            if is_model_evaluation:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
            else:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # 2.2 store real/imagined data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation) 
            elif self.store_type == "detail":        
                raise NotImplementedError
            # 2.3 update real/imagined initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

            # 2.4 update q using mixed data for many steps
            loss, info = self.update_q()

            # 2.4 update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            # 2.5 model-based rl
            if self.total_steps >= self.warm_start_steps:
                self.is_start_model = True
                # start model-based episode after warm start
                mb_info = self._run_mb_episode()
                self.log_mb_stats(mb_info)

            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

        # 2.5 sampling data using model after each episode
        if self.mbrl_type == "dyna":
            self._imagine_model_data()
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
    
    def _train_model(self):
        # set_trace()
        loss_area_total = []
        loss_delay_total = []
        lr_decay = False
        if self.is_first_model_train:
            self.is_first_model_train = False
            if self.lr_decay:
                lr_decay = True
            train_steps = self.train_model_start_steps
        else:
            train_steps = self.train_model_finetune_steps
            # if self.lr_decay:
            #     self.model_lr_scheduler = lr_scheduler.StepLR(
            #         self.ppa_model_optimizer,
            #         self.lr_decay_step_finetune,
            #         gamma=self.lr_decay_rate
            #     )
        TIMEDICT = {}
        for train_step in range(train_steps):
            # sample a batch
            transitions = self.replay_memory.sample(self.train_model_batch_size)
            batch = MBRLTransition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            normalize_area_batch = torch.cat(batch.normalize_area).float().to(self.device)
            normalize_delay_batch = torch.cat(batch.normalize_delay).float().to(self.device)

            batch_num = next_state_batch.shape[0]
            area_loss = torch.zeros(batch_num, device=self.device)
            delay_loss = torch.zeros(batch_num, device=self.device)
            states = []
            TIMEDICT["STARTBATCH"] = time.time()
            for j in range(batch_num):
                # !!! 耗时，重复调用，可工程优化
                ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, next_state_batch[j].cpu().numpy())
                ct32 = torch.tensor(np.array([ct32]))
                ct22 = torch.tensor(np.array([ct22]))
                if stage_num < self.MAX_STAGE_NUM-1:
                    zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                    ct32 = torch.cat((ct32, zeros), dim=1)
                    ct22 = torch.cat((ct22, zeros), dim=1)
                state = torch.cat((ct32, ct22), dim=0)
                states.append(state.unsqueeze(0))
            states = torch.cat(states)
            print(f"train model states nan: {states.isnan().any()}, inf: {states.isinf().any()}")
            TIMEDICT["ENDPROCESSDATA"] = time.time()
            predict_area, predict_delay = self.ppa_model(states.float())
            TIMEDICT["ENDMODEL"] = time.time()
            
            predict_area = predict_area.squeeze()
            predict_delay = predict_delay.squeeze()

            # 可能出问题一
            print(f"predict area shape: {predict_area.shape}")
            print(f"predict delay shape: {predict_delay.shape}")
            print(f"normalize area shape: {normalize_area_batch.shape}")
            print(f"normalize delay shape: {normalize_delay_batch.shape}")
            print(f"predict area nan: {predict_area.isnan().any()}, inf: {predict_area.isinf().any()}")
            print(f"predict delay nan: {predict_delay.isnan().any()}, inf: {predict_delay.isinf().any()}")
            print(f"normalize area nan: {normalize_area_batch.isnan().any()}, inf: {normalize_area_batch.isinf().any()}")
            print(f"normalize delay nan: {normalize_delay_batch.isnan().any()}, inf: {normalize_delay_batch.isinf().any()}")
            
            st = TIMEDICT["STARTBATCH"]
            for k in TIMEDICT.keys():
                if k != "STARTBATCH":
                    et = TIMEDICT[k] - st
                    print(f"time {k}: {et} seconds")
            
            area_loss = self.mse_loss(
                predict_area, normalize_area_batch
            )
            delay_loss = self.mse_loss(
                predict_delay, normalize_delay_batch
            )
            loss_area = torch.mean(area_loss)
            loss_delay = torch.mean(delay_loss)
            loss = loss_area + loss_delay
            self.ppa_model_optimizer.zero_grad()
            loss.backward()
            for param in self.ppa_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.ppa_model_optimizer.step()
            loss_area_total.append(loss_area.item())
            loss_delay_total.append(loss_delay.item())
            if lr_decay:
                self.model_lr_scheduler.step()

            logger.tb_logger.add_scalar('mbrl model area loss each step', loss_area.item(), global_step=self.train_model_total_steps)
            logger.tb_logger.add_scalar('mbrl model delay loss each step', loss_delay.item(), global_step=self.train_model_total_steps)
            self.train_model_total_steps += 1
        info = {
            "loss_area_total": loss_area_total,
            "loss_delay_total": loss_delay_total
        }
        return loss, info
    
    def _imagine_data(self):
        # 从真实环境出发
        pass

    def _run_mb_episode(self):
        mb_info = {}
        if self.is_sr_model:
            # sr model
            if self.sr_model_path is None:
                # train a sr model using collected samples
                raise NotImplementedError
            else:
                if not self.has_load_ppa_model:
                    self.has_load_ppa_model = True
                self.ppa_model = {}
                if self.sr_model_path[0].endswith("joblib"):
                    # gplearn model
                    self.ppa_model["area_model"] = joblib.load(self.sr_model_path[0])
                    self.ppa_model["delay_model"] = joblib.load(self.sr_model_path[1])
                elif self.sr_model_path[0].endswith("pkl"):
                    # dsr model
                    area_expression = Expression(self.operators, self.dsr_sequence[0], self.dsr_length[0]).to(self.device)
                    delay_expression = Expression(self.operators, self.dsr_sequence[1], self.dsr_length[1]).to(self.device)
                    area_expression.load_state_dict(torch.load(self.sr_model_path[0]))
                    delay_expression.load_state_dict(torch.load(self.sr_model_path[1]))
                    self.ppa_model["area_model"] = area_expression
                    self.ppa_model["delay_model"] = delay_expression
        else:
            # 1. train model
            if self.ppa_model_path is None:
                # execute model training as usual
                if self.total_steps % self.train_model_freq == 0:
                    model_loss, model_train_info = self._train_model()        
                    mb_info['model_loss'] = model_loss
                    mb_info['model_train_info'] = model_train_info
            else:
                if not self.has_load_ppa_model:
                    self.has_load_ppa_model = True
                    self.ppa_model.load_state_dict(torch.load(self.ppa_model_path))
        # 2. imagine data
        # if self.total_steps % self.imagine_data_freq == 0:
        #     self._imagine_data()
        return mb_info

    def log_mb_stats(self, mb_info):
        if "model_train_info" in mb_info.keys():
            logger.tb_logger.add_scalar('mbrl model area loss', np.mean(mb_info["model_train_info"]["loss_area_total"]), global_step=self.total_steps)
            logger.tb_logger.add_scalar('mbrl model delay loss', np.mean(mb_info["model_train_info"]["loss_delay_total"]), global_step=self.total_steps)

    def run_experiments(self):
        for episode_num in range(self.total_episodes):
            self.run_episode(episode_num)
            # if (episode_num+1) % self.evaluate_freq == 0 and self.evaluate_freq > 0:
            #     self.evaluate(episode_num)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num)
        self.end_experiments(episode_num)

    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        if self.store_type == "detail":
            save_data_dict["replay_buffer"] = self.replay_memory.memory
        # env initial state pool 
        if self.env.initial_state_pool_max_len > 0:
            save_data_dict["env_initial_state_pool"] = self.env.initial_state_pool
        # best state best design
        save_data_dict["found_best_info"] = self.found_best_info
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]
        
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        best_state = copy.deepcopy(self.found_best_info["found_best_state"])
        ppas_dict = self.env.get_ppa_full_delay_cons(best_state)

        # no evaluate
        # if episode_num % self.evaluate_imagine_state_freq == 0:
        #     full_ppas_dict = self.evaluate(episode_num)
        #     for k in ppas_dict.keys():
        #         ppas_dict[k].extend(full_ppas_dict[k])

        save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(self.total_steps, save_data_dict)

    def evaluate(self, episode_num):
        # TODO: evaluate 函数要改一下，去evaluate虚拟的pool
        evaluate_num = min(self.evaluate_imagine_state_num, len(self.env.imagined_initial_state_pool))
        ppa_list = []
        full_ppa_dict = {
            "area": [],
            "delay": [],
            "power": []
        }

        for i in range(1, evaluate_num+1):
            test_state = copy.deepcopy(
                self.env.imagined_initial_state_pool[-i]["state"]
            )
            # get full ppa dict
            ppas_dict = self.env.get_ppa_full_delay_cons(test_state)
            for k in ppas_dict.keys():
                full_ppa_dict[k].extend(ppas_dict[k])

            # get avg ppa
            if self.is_mac:
                initial_partial_product = MacPartialProduct[self.bit_width]
            else:
                initial_partial_product = PartialProduct[self.bit_width]
            ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
            rewards_dict = self.env.get_reward()
            avg_area = np.mean(rewards_dict['area'])
            avg_delay = np.mean(rewards_dict['delay'])
            avg_ppa = self.env.weight_area * avg_area / self.env.wallace_area + \
                self.env.weight_delay * avg_delay / self.env.wallace_delay
            avg_ppa = avg_ppa * self.env.ppa_scale
            normalize_area = self.env.ppa_scale * (avg_area / self.env.wallace_area)
            normalize_delay = self.env.ppa_scale * (avg_delay / self.env.wallace_delay)

            if avg_ppa < self.found_best_info['found_best_ppa']:
                _, state_policy_info = self.q_policy.select_action(
                    torch.tensor(test_state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
                state_mask = state_policy_info['mask']
                self.env.initial_state_pool.append(
                            {
                                "state": copy.deepcopy(test_state),
                                "area": avg_area,
                                "delay": avg_delay,
                                "ppa": avg_ppa,
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "best_ppa",
                                "normalize_area": normalize_area,
                                "normalize_delay": normalize_delay
                            }
                        )
                # update best ppa found
                self.found_best_info["found_best_ppa"] = avg_ppa
                self.found_best_info["found_best_state"] = copy.deepcopy(test_state)
                self.found_best_info["found_best_area"] = avg_area 
                self.found_best_info["found_best_delay"] = avg_delay
            ppa_list.append(avg_ppa)

        best_ppa = np.min(ppa_list)
        logger.tb_logger.add_scalar('mbrl model true best ppa found', best_ppa, global_step=episode_num)
        
        return full_ppa_dict
        
class PlanningShootingAlgorithm(MBRLRNDDQNAlgorithm):
    def update_env_initial_state_pool(self, state, rewards_dict, state_mask, is_model_evaluation=False):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if is_model_evaluation:
                    if self.imagine_found_best_ppa > rewards_dict['avg_ppa']:
                        self.env.imagined_initial_state_pool.append(
                            {
                                "state": copy.deepcopy(state),
                                "area": 0,
                                "delay": 0,
                                "ppa": rewards_dict['avg_ppa'],
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "best_ppa",
                                "normalize_area": rewards_dict["normalize_area"],
                                "normalize_delay": rewards_dict["normalize_delay"]
                            }
                        )
                        self.imagine_found_best_ppa = rewards_dict['avg_ppa']
                else:
                    if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                        # push the best ppa state into the initial pool
                        avg_area = np.mean(rewards_dict['area'])
                        avg_delay = np.mean(rewards_dict['delay'])
                        self.env.initial_state_pool.append(
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
        # best ppa info
        if not is_model_evaluation:
            if self.found_best_info["found_best_ppa"] > rewards_dict['avg_ppa']:
                self.found_best_info["found_best_ppa"] = rewards_dict['avg_ppa']
                self.found_best_info["found_best_state"] = copy.deepcopy(state)
                self.found_best_info["found_best_area"] = np.mean(rewards_dict['area']) 
                self.found_best_info["found_best_delay"] = np.mean(rewards_dict['delay'])

    def _imagine_model_data(self):
        # set_trace()
        if self.is_start_model:
            # reset imagine found best ppa
            self.imagine_found_best_ppa = 10000
            steps = 0
            for _ in range(self.num_sample):
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
                for _ in range(self.depth):
                    # self.total_steps += 1
                    steps += 1
                    action, policy_info = self.q_policy.select_action(
                        torch.tensor(state), 
                        self.total_steps, 
                        deterministic=self.deterministic,
                        is_softmax=self.imagine_is_softmax
                    )
                    # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                    next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                    _, next_state_policy_info = self.q_policy.select_action(
                        torch.tensor(next_state), self.total_steps, 
                        deterministic=self.deterministic,
                        is_softmax=self.imagine_is_softmax
                    )
                    # 2.2 store real/imagined data
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                    
                    # 2.3 update real/imagined initial state pool
                    self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                    state = copy.deepcopy(next_state)

                    avg_ppa = rewards_dict['avg_ppa']
                    logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                    logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)

    def evaluate(self, episode_num):
        # TODO: evaluate 函数要改一下，去evaluate虚拟的pool
        evaluate_num = min(self.evaluate_imagine_state_num, len(self.env.imagined_initial_state_pool))

        # last_found_best_ppa = self.found_best_info['found_best_ppa']
        true_found_best_ppa = self.found_best_info['found_best_ppa']
        last_found_best_ppa = self.env.initial_state_pool[0]["ppa"]
        cur_found_best_ppa = 10000
        newly_found_best_state = 0
        for i in range(1, evaluate_num+1):
            test_state = copy.deepcopy(
                self.env.imagined_initial_state_pool[-i]["state"]
            )
            # get avg ppa
            if self.is_mac:
                initial_partial_product = MacPartialProduct[self.bit_width]
            else:
                initial_partial_product = PartialProduct[self.bit_width]
            ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
            rewards_dict = self.env.get_reward()
            avg_area = np.mean(rewards_dict['area'])
            avg_delay = np.mean(rewards_dict['delay'])
            avg_ppa = self.env.weight_area * avg_area / self.env.wallace_area + \
                self.env.weight_delay * avg_delay / self.env.wallace_delay
            avg_ppa = avg_ppa * self.env.ppa_scale
            normalize_area = self.env.ppa_scale * (avg_area / self.env.wallace_area)
            normalize_delay = self.env.ppa_scale * (avg_delay / self.env.wallace_delay)
            if avg_ppa < cur_found_best_ppa:
                cur_found_best_ppa = avg_ppa
            
            if avg_ppa < last_found_best_ppa:
                newly_found_best_state += 1
                cur_found_best_ppa = avg_ppa
                _, state_policy_info = self.q_policy.select_action(
                    torch.tensor(test_state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
                state_mask = state_policy_info['mask']
                self.env.initial_state_pool.append(
                            {
                                "state": copy.deepcopy(test_state),
                                "area": avg_area,
                                "delay": avg_delay,
                                "ppa": avg_ppa,
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "best_ppa",
                                "normalize_area": normalize_area,
                                "normalize_delay": normalize_delay
                            }
                        )
                last_found_best_ppa = self.env.initial_state_pool[0]["ppa"]
            
            # update best ppa found
            if avg_ppa < true_found_best_ppa:
                true_found_best_ppa = avg_ppa
                self.found_best_info["found_best_ppa"] = avg_ppa
                self.found_best_info["found_best_state"] = copy.deepcopy(test_state)
                self.found_best_info["found_best_area"] = avg_area 
                self.found_best_info["found_best_delay"] = avg_delay

        logger.tb_logger.add_scalar('mbrl model true best ppa found', cur_found_best_ppa, global_step=episode_num)
        logger.tb_logger.add_scalar('mbrl model evaluate imagine state num', evaluate_num, global_step=episode_num)
        logger.tb_logger.add_scalar('mbrl model found best ppa state num', newly_found_best_state, global_step=episode_num)
        logger.tb_logger.add_scalar('mbrl model total best ppa', true_found_best_ppa, global_step=episode_num)
        
    def random_shooting(self, episode_num):
        # 1. _imagine_model_data search data randomly
        self._imagine_model_data()
        # 2. evaluate imagine model data and log ppa
        self.evaluate(episode_num)

    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if len(self.env.initial_state_pool) >= 2:
            state_mutual_distances = self.env.get_mutual_distance()
            self.log_state_mutual_distances(state_mutual_distances)
            self.log_env_pool()

        # 2. sampling data
        for step in range(self.len_per_episode):
            if not self.is_start_model:
                logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
                logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
                if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                    logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                    if info_count is not None:
                        logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                        logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
                self.total_steps += 1
                # 2.1 sampling real environment interaction
                is_model_evaluation = False

                """
                    imagine data inter each episode
                """
                """
                    imagine data inter each episode
                """
                action, policy_info = self.q_policy.select_action(
                    torch.tensor(state), 
                    self.total_steps, 
                    deterministic=self.deterministic,
                    is_softmax=self.is_softmax
                )
                logger.log(f"total steps: {self.total_steps}, action: {action}")
                # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                if is_model_evaluation:
                    next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
                else:
                    next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
                _, next_state_policy_info = self.q_policy.select_action(
                    torch.tensor(next_state), self.total_steps, 
                    deterministic=self.deterministic,
                    is_softmax=self.is_softmax
                )
                # 2.2 store real/imagined data
                if self.store_type == "simple":
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation) 
                elif self.store_type == "detail":        
                    raise NotImplementedError
                # 2.3 update real/imagined initial state pool
                self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

                # 2.4 update q using mixed data for many steps
                loss, info = self.update_q()

                # 2.4 update target q (TODO: SOFT UPDATE)
                if self.total_steps % self.target_update_freq == 0:
                    self.target_q_policy.load_state_dict(
                        self.q_policy.state_dict()
                    )
                # 2.5 model-based rl
                if self.total_steps >= self.warm_start_steps:
                    self.is_start_model = True
                    # start model-based episode after warm start
                    mb_info = self._run_mb_episode()
                    self.log_mb_stats(mb_info)

                state = copy.deepcopy(next_state)
                # reset agent 
                if self.agent_reset_freq > 0:
                    self.reset_agent()
                # log datasets
                self.log_stats(
                    loss, reward, rewards_dict,
                    next_state, action, info, policy_info
                )
                self.log_rnd_stats(info)
                
                avg_ppa = rewards_dict['avg_ppa']
                logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

        if self.is_start_model:
            # planning for each episode/searching optimal states
            self.random_shooting(episode_num)
        # 2.5 sampling data using model after each episode
        # self._imagine_model_data()
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )

    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        if self.store_type == "detail":
            save_data_dict["replay_buffer"] = self.replay_memory.memory
        # env initial state pool 
        if self.env.initial_state_pool_max_len > 0:
            save_data_dict["env_initial_state_pool"] = self.env.initial_state_pool
        # best state best design
        save_data_dict["found_best_info"] = self.found_best_info
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]
        
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        best_state = copy.deepcopy(self.found_best_info["found_best_state"])
        ppas_dict = self.env.get_ppa_full_delay_cons(best_state)
        
        save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(self.total_steps, save_data_dict)

class AlterSearchAndLearnAlgorithm(PlanningShootingAlgorithm):
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else: 
            if not self.is_start_model or len(self.imagined_replay_memory) == 0 or not self.use_dyna_data:
                # updating q using pure real data
                transitions = self.replay_memory.sample(self.batch_size)
                batch = MBRLTransition(*zip(*transitions))
                next_state_batch = torch.cat(batch.next_state)
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_mask = torch.cat(batch.mask)
                next_state_mask = torch.cat(batch.next_state_mask)
                # update reward int run mean std
                self.update_reward_int_run_mean_std(
                    reward_batch.cpu().numpy()
                )

                # compute reward int 
                int_rewards_batch = self.compute_int_rewards(
                    next_state_batch, next_state_mask
                )
                int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = self.compute_values(
                    state_batch, action_batch, state_mask
                )
                next_state_values = self.compute_values(
                    next_state_batch, None, next_state_mask
                )
                target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                loss = self.loss_fn(
                    state_action_values.unsqueeze(1), 
                    target_state_action_values.unsqueeze(1)
                )

                self.policy_optimizer.zero_grad()
                loss.backward()
                for param in self.q_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.policy_optimizer.step()

                info = {
                    "q_values": state_action_values.detach().cpu().numpy(),
                    "target_q_values": target_state_action_values.detach().cpu().numpy(),
                    "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
                }
                self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())
            else:
                for _ in range(self.train_q_steps):
                    imagined_batch_size = min(len(self.imagined_replay_memory), int((1-self.real_data_ratio)*self.batch_size))
                    real_batch_size = self.batch_size - imagined_batch_size
                    imagined_batch_transitions = self.imagined_replay_memory.sample(imagined_batch_size)
                    real_batch_transitions = self.replay_memory.sample(real_batch_size)
                    next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask = self.combine_batch(
                        imagined_batch_transitions, real_batch_transitions
                    )
                    # update reward int run mean std
                    self.update_reward_int_run_mean_std(
                        reward_batch.cpu().numpy()
                    )

                    # compute reward int 
                    int_rewards_batch = self.compute_int_rewards(
                        next_state_batch, next_state_mask
                    )
                    int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                    train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = self.compute_values(
                        state_batch, action_batch, state_mask
                    )
                    next_state_values = self.compute_values(
                        next_state_batch, None, next_state_mask
                    )
                    target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                    loss = self.loss_fn(
                        state_action_values.unsqueeze(1), 
                        target_state_action_values.unsqueeze(1)
                    )

                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    for param in self.q_policy.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.policy_optimizer.step()

                    info = {
                        "q_values": state_action_values.detach().cpu().numpy(),
                        "target_q_values": target_state_action_values.detach().cpu().numpy(),
                        "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
                    }
                    self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                    self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())            


            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info

    def run_experiments(self):
        for episode_num in range(self.start_episodes):
            if not self.is_start_model:
                # 1. warm start pure real data
                self.run_episode(episode_num)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num)
        search_type = 1 # 1 for model search -1 for real env
        for episode_num in range(self.total_episodes):
            if (episode_num+1) % self.model_env_iterative_epi_num == 0:
                search_type = -1 * search_type
            if search_type == 1:
                self.random_shooting(episode_num + self.start_episodes) # search 100/200 steps for 20 episodes
            if search_type == -1:
                self.run_episode(episode_num + self.start_episodes) # run 20 episodes
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num + self.start_episodes)
        self.end_experiments(episode_num + self.start_episodes)

    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        env_state, sel_index = self.env.reset()
        state = copy.deepcopy(env_state)
    
        # 2. sampling data
        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # 2.1 sampling real environment interaction
            is_model_evaluation = False

            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
            if is_model_evaluation:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
            else:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # 2.2 store real/imagined data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation) 
            elif self.store_type == "detail":        
                raise NotImplementedError
            # 2.3 update real/imagined initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

            # 2.4 update q using mixed data for many steps
            loss, info = self.update_q()

            # 2.4 update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            # 2.5 model-based rl
            if self.total_steps >= self.warm_start_steps:
                self.is_start_model = True
                # start model-based episode after warm start
                mb_info = self._run_mb_episode()
                self.log_mb_stats(mb_info)

            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )

"""
    V2: multi-obj no ga, with single Q
"""
class AlterSearchAndLearnV2Algorithm(AlterSearchAndLearnAlgorithm):
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        # model-based kwargs
        ppa_model, 
        imagined_replay_memory,
        **mbrl_alg_kwargs
    ):
        super(AlterSearchAndLearnV2Algorithm, self).__init__(
            env, q_policy, target_q_policy, replay_memory,
            rnd_predictor, rnd_target, int_reward_run_mean_std,
            ppa_model, imagined_replay_memory,
            **mbrl_alg_kwargs
        )

        weight_list = self.env.weight_list
        self.imagine_found_best_ppa = []
        self.found_best_info = []
        for i in range(len(weight_list)):
            # best ppa found
            self.found_best_info.append(
                {
                    "found_best_ppa": 1e5,
                    "found_best_state": None,
                    "found_best_area": 1e5,
                    "found_best_delay": 1e5
                }
            )
            self.imagine_found_best_ppa.append(10000)

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask, is_model_evaluation=False):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if is_model_evaluation:
                    for i, weights in enumerate(self.env.weight_list):
                        imagine_found_best_ppa = self.imagine_found_best_ppa[i]
                        cur_state_ppa = weights[0] * rewards_dict["normalize_area"] + weights[1] * rewards_dict["normalize_delay"]
                        if imagine_found_best_ppa > cur_state_ppa:
                            self.env.imagined_initial_state_pool[i].append(
                                {
                                    "state": copy.deepcopy(state),
                                    "area": 0,
                                    "delay": 0,
                                    "ppa": cur_state_ppa,
                                    "count": 1,
                                    "state_mask": state_mask,
                                    "state_type": "best_ppa",
                                    "normalize_area": rewards_dict["normalize_area"],
                                    "normalize_delay": rewards_dict["normalize_delay"]
                                }
                            )
                            self.imagine_found_best_ppa[i] = cur_state_ppa
                else:
                    for i, weights in enumerate(self.env.weight_list):
                        found_best_ppa = self.found_best_info[i]['found_best_ppa']
                        cur_state_ppa = weights[0] * rewards_dict["normalize_area"] + weights[1] * rewards_dict["normalize_delay"]
                        if found_best_ppa > cur_state_ppa:
                            # push the best ppa state into the initial pool
                            avg_area = np.mean(rewards_dict['area'])
                            avg_delay = np.mean(rewards_dict['delay'])
                            self.env.initial_state_pool[i].append(
                                {
                                    "state": copy.deepcopy(state),
                                    "area": avg_area,
                                    "delay": avg_delay,
                                    "ppa": cur_state_ppa,
                                    "count": 1,
                                    "state_mask": state_mask,
                                    "state_type": "best_ppa",
                                    "normalize_area": rewards_dict["normalize_area"],
                                    "normalize_delay": rewards_dict["normalize_delay"]
                                }
                            )
        # best ppa info
        if not is_model_evaluation:
            for i, weights in enumerate(self.env.weight_list):
                cur_state_ppa = weights[0] * rewards_dict["normalize_area"] + weights[1] * rewards_dict["normalize_delay"]
                if self.found_best_info[i]["found_best_ppa"] > cur_state_ppa:
                    self.found_best_info[i]["found_best_ppa"] = cur_state_ppa
                    self.found_best_info[i]["found_best_state"] = copy.deepcopy(state)
                    self.found_best_info[i]["found_best_area"] = np.mean(rewards_dict['area']) 
                    self.found_best_info[i]["found_best_delay"] = np.mean(rewards_dict['delay'])

    def _merge_ppa(self, ppas_list):
        merge_ppas_dict = {
            "area": [],
            "delay": [],
            "power": []
        }
        for ppas in ppas_list:
            for k in ppas.keys():
                merge_ppas_dict[k].extend(ppas[k])
        return merge_ppas_dict
    
    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        if self.store_type == "detail":
            save_data_dict["replay_buffer"] = self.replay_memory.memory
        # env initial state pool 
        if self.env.initial_state_pool_max_len > 0:
            save_data_dict["env_initial_state_pool"] = self.env.initial_state_pool
        # best state best design
        save_data_dict["found_best_info"] = self.found_best_info
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]
        
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        ppas_list = []
        for i in range(len(self.found_best_info)):
            best_state = copy.deepcopy(self.found_best_info[i]["found_best_state"])
            ppas_dict = self.env.get_ppa_full_delay_cons(best_state)
            ppas_list.append(ppas_dict)
        merge_ppas_dict = self._merge_ppa(ppas_list)
        save_pareto_data_dict = self.log_and_save_pareto_points(merge_ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(self.total_steps, save_data_dict)

    def _imagine_model_data(self):
        if self.is_start_model:
            # reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)):
                self.imagine_found_best_ppa[i] = 10000
                for _ in range(self.num_sample):
                    env_state, sel_index = self.env.reset(pool_index=i)
                    state = copy.deepcopy(env_state)
                    for _ in range(self.depth):
                        steps += 1
                        action, policy_info = self.q_policy.select_action(
                            torch.tensor(state), 
                            self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                        next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                        _, next_state_policy_info = self.q_policy.select_action(
                            torch.tensor(next_state), self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # 2.2 store real/imagined data
                        self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                        
                        # 2.3 update real/imagined initial state pool
                        self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                        state = copy.deepcopy(next_state)

                        avg_ppa = rewards_dict['avg_ppa']
                        logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)

    def evaluate(self, episode_num):
        # TODO: evaluate 函数要改一下，去evaluate虚拟的pool
        for j in range(len(self.env.weight_list)):
            evaluate_num = min(self.evaluate_imagine_state_num, len(self.env.imagined_initial_state_pool[j]))
            true_found_best_ppa = self.found_best_info[j]['found_best_ppa']
            last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
            cur_found_best_ppa = 10000
            newly_found_best_state = 0
            for i in range(1, evaluate_num+1):
                test_state = copy.deepcopy(
                    self.env.imagined_initial_state_pool[j][-i]["state"]
                )
                # get avg ppa
                if self.is_mac:
                    initial_partial_product = MacPartialProduct[self.bit_width]
                else:
                    initial_partial_product = PartialProduct[self.bit_width]
                ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
                rewards_dict = self.env.get_reward()
                avg_area = np.mean(rewards_dict['area'])
                avg_delay = np.mean(rewards_dict['delay'])
                avg_ppa = self.env.weight_list[j][0] * avg_area / self.env.wallace_area + \
                    self.env.weight_list[j][1] * avg_delay / self.env.wallace_delay
                avg_ppa = avg_ppa * self.env.ppa_scale
                normalize_area = self.env.ppa_scale * (avg_area / self.env.wallace_area)
                normalize_delay = self.env.ppa_scale * (avg_delay / self.env.wallace_delay)
                if avg_ppa < cur_found_best_ppa:
                    cur_found_best_ppa = avg_ppa
                
                if avg_ppa < last_found_best_ppa:
                    newly_found_best_state += 1
                    cur_found_best_ppa = avg_ppa
                    _, state_policy_info = self.q_policy.select_action(
                        torch.tensor(test_state), 0, 
                        deterministic=False,
                        is_softmax=False
                    )
                    state_mask = state_policy_info['mask']
                    self.env.initial_state_pool[j].append(
                                {
                                    "state": copy.deepcopy(test_state),
                                    "area": avg_area,
                                    "delay": avg_delay,
                                    "ppa": avg_ppa,
                                    "count": 1,
                                    "state_mask": state_mask,
                                    "state_type": "best_ppa",
                                    "normalize_area": normalize_area,
                                    "normalize_delay": normalize_delay
                                }
                            )
                    last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
                
                # update best ppa found
                if avg_ppa < true_found_best_ppa:
                    true_found_best_ppa = avg_ppa
                    self.found_best_info[j]["found_best_ppa"] = avg_ppa
                    self.found_best_info[j]["found_best_state"] = copy.deepcopy(test_state)
                    self.found_best_info[j]["found_best_area"] = avg_area 
                    self.found_best_info[j]["found_best_delay"] = avg_delay

            logger.tb_logger.add_scalar(f'mbrl model true best ppa found {j}-th env pool', cur_found_best_ppa, global_step=episode_num)
            logger.tb_logger.add_scalar(f'mbrl model evaluate imagine state num {j}-th env pool', evaluate_num, global_step=episode_num)
            logger.tb_logger.add_scalar(f'mbrl model found best ppa state num {j}-th env pool', newly_found_best_state, global_step=episode_num)
            logger.tb_logger.add_scalar(f'mbrl model total best ppa {j}-th env pool', true_found_best_ppa, global_step=episode_num)

"""
    V2GA: multi-obj ga, with single Q
"""
class AlterSearchAndLearnV2GAAlgorithm(AlterSearchAndLearnV2Algorithm):
    def evaluate_with_model(self, test_state, j):
        # 1. 首先得到image state
        if self.is_mac:
            initial_partial_product = MacPartialProduct[self.bit_width]
        else:
            initial_partial_product = PartialProduct[self.bit_width]
        logger.log(f"before decompose: {test_state}")
        ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
        logger.log(f"after decompose: {test_state}")

        if stage_num > (self.MAX_STAGE_NUM - 1):
            logger.log(f"warning!!! stage num invalid: {stage_num}")
            return 20000, 20000, 20000

        # 2. evaluate with model
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        image_state = torch.tensor(
            image_state,
            dtype=torch.float,
            device=self.device
        )
        with torch.no_grad():
            normalize_area, normalize_delay = self.ppa_model(
                image_state.unsqueeze(0)
            )
        normalize_area = normalize_area.item()
        normalize_delay = normalize_delay.item()

        avg_ppa = self.env.weight_list[j][0] * normalize_area + self.env.weight_list[j][1] * normalize_delay

        return avg_ppa, normalize_area, normalize_delay

    def evaluate_with_dsr_model(self, test_state, j):
        # 1. 首先得到image state
        if self.is_mac:
            initial_partial_product = MacPartialProduct[self.bit_width]
        else:
            initial_partial_product = PartialProduct[self.bit_width]
        logger.log(f"before decompose: {test_state}")
        ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
        logger.log(f"after decompose: {test_state}")

        if stage_num > (self.MAX_STAGE_NUM - 1):
            logger.log(f"warning!!! stage num invalid: {stage_num}")
            return 20000, 20000, 20000

        # 2. evaluate with dsr model
        sr_feature = []
        sr_feature.append(stage_num)
        ct32_num = np.sum(ct32)
        ct22_num = np.sum(ct22)
        sr_feature.append(3*ct32_num+2*ct22_num)
        sr_feature.append(ct32_num)
        sr_feature.append(ct22_num)
        
        ct32 = np.expand_dims(ct32, axis=0)
        ct22 = np.expand_dims(ct22, axis=0)
        if stage_num < self.MAX_STAGE_NUM-1: # self.MAX_STAGE_NUM 设置为4是不是有点小呢？MAX STAGE NUM 应该是用来做图片填充的
            zeros = np.zeros((1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2)))
            ct32 = np.concatenate((ct32, zeros), axis=1)
            ct22 = np.concatenate((ct22, zeros), axis=1)
        image_state = np.concatenate((ct32, ct22), axis=0) # (2, max_stage-1, num_column)        
        # column feature for ct32 / ct22
        for i in range(1, image_state.shape[2]):
            column_ct32_ct22 = np.sum(image_state[0,:,i]) + np.sum(image_state[1,:,i])
            if column_ct32_ct22 == 0:
                column_ct32_ct22 = -1
            sr_feature.append(column_ct32_ct22)
        sr_feature = np.array(sr_feature)
        sr_feature = np.expand_dims(sr_feature, axis=0)
        sr_feature = torch.tensor(sr_feature, dtype=torch.float).to(self.device)
        with torch.no_grad():
            normalize_area = self.ppa_model["area_model"](sr_feature)
            normalize_delay = self.ppa_model["delay_model"](sr_feature)
        normalize_area = normalize_area.item()
        normalize_delay = normalize_delay.item()

        avg_ppa = self.env.weight_list[j][0] * normalize_area + self.env.weight_list[j][1] * normalize_delay

        return avg_ppa, normalize_area, normalize_delay

    def update_imagine_pool(self, state, avg_ppa, normalize_area, normalize_delay, j):
        if avg_ppa < self.imagine_found_best_ppa[j]:
            self.env.imagined_initial_state_pool[j].append(
                {
                    "state": copy.deepcopy(state),
                    "area": 0,
                    "delay": 0,
                    "ppa": avg_ppa,
                    "count": 1,
                    "state_mask": None,
                    "state_type": "best_ppa",
                    "normalize_area": normalize_area,
                    "normalize_delay": normalize_delay
                }
            )
            self.imagine_found_best_ppa[j] = avg_ppa
            return True
        else:
            return False

    def random_shooting(self, episode_num):
        # 1. _imagine_model_data search data randomly
        self._imagine_model_data(episode_num)
        # 2. evaluate imagine model data and log ppa
        self.evaluate(episode_num)

    def _imagine_model_data(self, episode_num):
        if self.is_start_model:
            # reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)):
                self.imagine_found_best_ppa[i] = 10000
                for _ in range(self.num_sample):
                    env_state, sel_index = self.env.reset(pool_index=i)
                    state = copy.deepcopy(env_state)
                    for _ in range(self.depth):
                        steps += 1
                        action, policy_info = self.q_policy.select_action(
                            torch.tensor(state), 
                            self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                        next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                        _, next_state_policy_info = self.q_policy.select_action(
                            torch.tensor(next_state), self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # 2.2 store real/imagined data
                        self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                        
                        # 2.3 update real/imagined initial state pool
                        self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                        state = copy.deepcopy(next_state)

                        avg_ppa = rewards_dict['avg_ppa']
                        logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)

            # 2. GA Mutation from different env pools
            
            for i in range(int(self.num_sample * self.depth / 2)):
                env_state1, sel_index1 = self.env.reset(pool_index=0)
                print(f"select first state index: {sel_index1}")
                pool_index = np.random.choice([1,2,3])
                env_state2, sel_index2 = self.env.reset(pool_index=pool_index)
                
                crossover_states = []
                # column crossover 
                print("before column crossover")
                cc_state1, cc_state2 = self.env.column_crossover(
                    copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                )
                print("before block crossover")
                # block crossover
                bc_state1, bc_state2 = self.env.block_crossover(
                    copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                )
                if cc_state1 is not None:
                    crossover_states.append(cc_state1)
                if cc_state2 is not None:
                    crossover_states.append(cc_state2)
                if bc_state1 is not None:
                    crossover_states.append(bc_state1)
                if bc_state2 is not None:
                    crossover_states.append(bc_state2)
                print(f"crossover states num: {len(crossover_states)}")
                if len(crossover_states) > 0:
                    for j in range(len(self.imagine_found_best_ppa)):
                        GA_found_state_num = 0
                        for state in crossover_states:
                            logger.log(f"state: {state}")
                            steps += 1
                            if self.is_sr_model:
                                avg_ppa, normalize_area, normalize_delay = self.evaluate_with_dsr_model(state, j)
                            else:
                                avg_ppa, normalize_area, normalize_delay = self.evaluate_with_model(state, j)
                            is_found_better_state = self.update_imagine_pool(state, avg_ppa, normalize_area, normalize_delay, j)
                            if is_found_better_state:
                                GA_found_state_num += 1
                            logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                            logger.tb_logger.add_scalar(f'{j}-th preference avg ppa', avg_ppa, global_step=self.total_steps + steps)
                        logger.tb_logger.add_scalar(f'{j}-th preference mbrl model ga found best state num', GA_found_state_num, global_step=episode_num)
                else:
                    print(f"warning!!!no valid crossover states found next iteration") 

"""
    V22GA: multi-obj ga v22, with single Q
"""
class AlterSearchAndLearnV22GAAlgorithm(AlterSearchAndLearnV2GAAlgorithm):
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        # model-based kwargs
        ppa_model, 
        imagined_replay_memory,
        is_value_function_mutation=True,
        is_crossover_variation=True,
        **mbrl_alg_kwargs
    ):
        super(AlterSearchAndLearnV2Algorithm, self).__init__(
            env, q_policy, target_q_policy, replay_memory,
            rnd_predictor, rnd_target, int_reward_run_mean_std,
            ppa_model, imagined_replay_memory,
            **mbrl_alg_kwargs
        )

        self.is_value_function_mutation = is_value_function_mutation
        self.is_crossover_variation = is_crossover_variation

        weight_list = self.env.weight_list
        self.imagine_found_best_ppa = []
        self.imagine_found_best_ppa_ga = []
        self.found_best_info = []
        self.imagine_ga_state_pool = []
        for i in range(len(weight_list)):
            # best ppa found
            self.found_best_info.append(
                {
                    "found_best_ppa": 1e5,
                    "found_best_state": None,
                    "found_best_area": 1e5,
                    "found_best_delay": 1e5
                }
            )
            self.imagine_found_best_ppa.append(10000)
            self.imagine_found_best_ppa_ga.append(10000)
            self.imagine_ga_state_pool.append(
                deque([],maxlen=self.env.initial_state_pool_max_len)
            )

    def update_imagine_pool(self, state, avg_ppa, normalize_area, normalize_delay, j):
        if avg_ppa < self.imagine_found_best_ppa_ga[j]:
            # imagine ga pool
            self.imagine_ga_state_pool[j].append(
                {
                    "state": copy.deepcopy(state),
                    "area": 0,
                    "delay": 0,
                    "ppa": avg_ppa,
                    "count": 1,
                    "state_mask": None,
                    "state_type": "best_ppa",
                    "normalize_area": normalize_area,
                    "normalize_delay": normalize_delay
                }
            )
            self.imagine_found_best_ppa_ga[j] = avg_ppa
            return True
        else:
            return False
        
    def _imagine_model_data(self, episode_num):
        if self.is_start_model:
            # 0. reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)): 
                self.imagine_found_best_ppa[i] = 10000
            # 1. imagine with model
            if self.is_value_function_mutation:
                for _ in range(self.num_sample):
                    pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
                    env_state, sel_index = self.env.reset(pool_index=pool_index)
                    # env_state, sel_index = self.env.reset(pool_index=0)
                    state = copy.deepcopy(env_state)
                    for _ in range(self.depth):
                        steps += 1
                        action, policy_info = self.q_policy.select_action(
                            torch.tensor(state), 
                            self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                        next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                        _, next_state_policy_info = self.q_policy.select_action(
                            torch.tensor(next_state), self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # 2.2 store real/imagined data
                        self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                        
                        # 2.3 update real/imagined initial state pool
                        self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                        state = copy.deepcopy(next_state)

                        avg_ppa = rewards_dict['avg_ppa']
                        logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)
            if self.is_crossover_variation:
                for i in range(len(self.imagine_found_best_ppa_ga)): 
                    self.imagine_found_best_ppa_ga[i] = 10000
                # 2. GA Mutation from different env pools
                for i in range(int(self.num_sample * self.depth / 2)):
                    env_state1, sel_index1 = self.env.reset(pool_index=0)
                    print(f"select first state index: {sel_index1}")
                    pool_index = np.random.choice([1,2,3])
                    env_state2, sel_index2 = self.env.reset(pool_index=pool_index)
                    
                    crossover_states = []
                    # column crossover 
                    print("before column crossover")
                    cc_state1, cc_state2 = self.env.column_crossover(
                        copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                    )
                    print("before block crossover")
                    # block crossover
                    bc_state1, bc_state2 = self.env.block_crossover(
                        copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                    )
                    if cc_state1 is not None:
                        crossover_states.append(cc_state1)
                    if cc_state2 is not None:
                        crossover_states.append(cc_state2)
                    if bc_state1 is not None:
                        crossover_states.append(bc_state1)
                    if bc_state2 is not None:
                        crossover_states.append(bc_state2)
                    print(f"crossover states num: {len(crossover_states)}")
                    if len(crossover_states) > 0:
                        for j in range(len(self.imagine_found_best_ppa)):
                            GA_found_state_num = 0
                            for state in crossover_states:
                                logger.log(f"state: {state}")
                                steps += 1
                                if self.is_sr_model:
                                    avg_ppa, normalize_area, normalize_delay = self.evaluate_with_dsr_model(state, j)
                                else:
                                    avg_ppa, normalize_area, normalize_delay = self.evaluate_with_model(state, j)
                                is_found_better_state = self.update_imagine_pool(state, avg_ppa, normalize_area, normalize_delay, j)
                                if is_found_better_state:
                                    GA_found_state_num += 1
                                logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                                logger.tb_logger.add_scalar(f'{j}-th preference avg ppa', avg_ppa, global_step=self.total_steps + steps)
                            logger.tb_logger.add_scalar(f'{j}-th preference mbrl model ga found best state num', GA_found_state_num, global_step=self.total_steps + steps)
                    else:
                        print(f"warning!!!no valid crossover states found next iteration") 

    def evaluate_ga(self, episode_num):
        # TODO: evaluate 函数要改一下，去evaluate虚拟的pool
        for j in range(len(self.env.weight_list)):
            evaluate_num = min(self.evaluate_imagine_state_num, len(self.imagine_ga_state_pool[j]))
            true_found_best_ppa = self.found_best_info[j]['found_best_ppa']
            last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
            cur_found_best_ppa = 10000
            newly_found_best_state = 0
            for i in range(1, evaluate_num+1):
                test_state = copy.deepcopy(
                    self.imagine_ga_state_pool[j][-i]["state"]
                )
                # get avg ppa
                if self.is_mac:
                    initial_partial_product = MacPartialProduct[self.bit_width]
                else:
                    initial_partial_product = PartialProduct[self.bit_width]
                ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
                rewards_dict = self.env.get_reward()
                avg_area = np.mean(rewards_dict['area'])
                avg_delay = np.mean(rewards_dict['delay'])
                avg_ppa = self.env.weight_list[j][0] * avg_area / self.env.wallace_area + \
                    self.env.weight_list[j][1] * avg_delay / self.env.wallace_delay
                avg_ppa = avg_ppa * self.env.ppa_scale
                normalize_area = self.env.ppa_scale * (avg_area / self.env.wallace_area)
                normalize_delay = self.env.ppa_scale * (avg_delay / self.env.wallace_delay)
                if avg_ppa < cur_found_best_ppa:
                    cur_found_best_ppa = avg_ppa
                
                if avg_ppa < last_found_best_ppa:
                    newly_found_best_state += 1
                    cur_found_best_ppa = avg_ppa
                    _, state_policy_info = self.q_policy.select_action(
                        torch.tensor(test_state), 0, 
                        deterministic=False,
                        is_softmax=False
                    )
                    state_mask = state_policy_info['mask']
                    self.env.initial_state_pool[j].append(
                                {
                                    "state": copy.deepcopy(test_state),
                                    "area": avg_area,
                                    "delay": avg_delay,
                                    "ppa": avg_ppa,
                                    "count": 1,
                                    "state_mask": state_mask,
                                    "state_type": "best_ppa",
                                    "normalize_area": normalize_area,
                                    "normalize_delay": normalize_delay
                                }
                            )
                    last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
                
                # update best ppa found
                if avg_ppa < true_found_best_ppa:
                    true_found_best_ppa = avg_ppa
                    self.found_best_info[j]["found_best_ppa"] = avg_ppa
                    self.found_best_info[j]["found_best_state"] = copy.deepcopy(test_state)
                    self.found_best_info[j]["found_best_area"] = avg_area 
                    self.found_best_info[j]["found_best_delay"] = avg_delay

            logger.tb_logger.add_scalar(f'GA mbrl model true best ppa found {j}-th env pool', cur_found_best_ppa, global_step=episode_num)
            logger.tb_logger.add_scalar(f'GA mbrl model evaluate imagine state num {j}-th env pool', evaluate_num, global_step=episode_num)
            logger.tb_logger.add_scalar(f'GA mbrl model found best ppa state num {j}-th env pool', newly_found_best_state, global_step=episode_num)
            logger.tb_logger.add_scalar(f'GA mbrl model total best ppa {j}-th env pool', true_found_best_ppa, global_step=episode_num)

    def random_shooting(self, episode_num):
        # 1. _imagine_model_data search data randomly
        self._imagine_model_data(episode_num)
        # 2. evaluate imagine model data and log ppa
        if self.is_value_function_mutation:
            self.evaluate(episode_num)
        # 2. evaluate imagine ga data and log ppa
        if self.is_crossover_variation:
            self.evaluate_ga(episode_num)

    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
        # pool_index = 0
        env_state, sel_index = self.env.reset(pool_index=pool_index)
        state = copy.deepcopy(env_state)
    
        # 2. sampling data
        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # 2.1 sampling real environment interaction
            is_model_evaluation = False

            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
            if is_model_evaluation:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
            else:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # 2.2 store real/imagined data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation) 
            elif self.store_type == "detail":        
                raise NotImplementedError
            # 2.3 update real/imagined initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

            # 2.4 update q using mixed data for many steps
            loss, info = self.update_q()

            # 2.4 update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            # 2.5 model-based rl
            if self.total_steps >= self.warm_start_steps:
                self.is_start_model = True
                # start model-based episode after warm start
                mb_info = self._run_mb_episode()
                self.log_mb_stats(mb_info)

            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )

"""
    nips rebuttal add a ea baseline using our ea operator
"""
class AblationOnlyGA(AlterSearchAndLearnV22GAAlgorithm):
    def run_experiments(self):
        for episode_num in range(self.total_episodes):
            if episode_num < self.start_episodes:
                # 1. warm start
                self.run_episode(episode_num) # run env with random mutation
                if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                    self.end_experiments(episode_num)
            else:
                # 2. GA crossover
                self.run_episode_ga(episode_num)
                if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                    self.end_experiments(episode_num)
        self.end_experiments(episode_num)

    def evaluate_ga(self, crossover_states, episode_num):
        for state in crossover_states:
            test_state = copy.deepcopy(
                state
            )
            # get avg ppa
            if self.is_mac:
                initial_partial_product = MacPartialProduct[self.bit_width]
            else:
                initial_partial_product = PartialProduct[self.bit_width]
            ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
            
            if stage_num > (self.MAX_STAGE_NUM - 1):
                logger.log(f"warning!!! stage num invalid: {stage_num}")
                continue
            
            rewards_dict = self.env.get_reward()
            avg_area = np.mean(rewards_dict['area'])
            avg_delay = np.mean(rewards_dict['delay'])

            for j in range(len(self.env.weight_list)):
                true_found_best_ppa = self.found_best_info[j]['found_best_ppa']
                last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
                cur_found_best_ppa = 10000
                newly_found_best_state = 0
                avg_ppa = self.env.weight_list[j][0] * avg_area / self.env.wallace_area + \
                    self.env.weight_list[j][1] * avg_delay / self.env.wallace_delay
                avg_ppa = avg_ppa * self.env.ppa_scale
                normalize_area = self.env.ppa_scale * (avg_area / self.env.wallace_area)
                normalize_delay = self.env.ppa_scale * (avg_delay / self.env.wallace_delay)
                if avg_ppa < cur_found_best_ppa:
                    cur_found_best_ppa = avg_ppa
                    
                if avg_ppa < last_found_best_ppa:
                    newly_found_best_state += 1
                    cur_found_best_ppa = avg_ppa
                    _, state_policy_info = self.q_policy.select_action(
                        torch.tensor(test_state), 0, 
                        deterministic=False,
                        is_softmax=False
                    )
                    state_mask = state_policy_info['mask']
                    self.env.initial_state_pool[j].append(
                                {
                                    "state": copy.deepcopy(test_state),
                                    "area": avg_area,
                                    "delay": avg_delay,
                                    "ppa": avg_ppa,
                                    "count": 1,
                                    "state_mask": state_mask,
                                    "state_type": "best_ppa",
                                    "normalize_area": normalize_area,
                                    "normalize_delay": normalize_delay
                                }
                            )
                    last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
                    
                # update best ppa found
                if avg_ppa < true_found_best_ppa:
                    true_found_best_ppa = avg_ppa
                    self.found_best_info[j]["found_best_ppa"] = avg_ppa
                    self.found_best_info[j]["found_best_state"] = copy.deepcopy(test_state)
                    self.found_best_info[j]["found_best_area"] = avg_area 
                    self.found_best_info[j]["found_best_delay"] = avg_delay

                logger.tb_logger.add_scalar(f'GA mbrl model true best ppa found {j}-th env pool', cur_found_best_ppa, global_step=episode_num)
                logger.tb_logger.add_scalar(f'GA mbrl model found best ppa state num {j}-th env pool', newly_found_best_state, global_step=episode_num)
                logger.tb_logger.add_scalar(f'GA mbrl model total best ppa {j}-th env pool', true_found_best_ppa, global_step=episode_num)

    def run_episode_ga(self, episode_num):
        for step in range(self.len_per_episode):
            env_state1, sel_index1 = self.env.reset(pool_index=0)    
            pool_index = np.random.choice([1,2,3])
            env_state2, sel_index2 = self.env.reset(pool_index=pool_index)
            
            crossover_type = np.random.choice([0,1])
            crossover_states = []
            if crossover_type == 0:
                # column crossover
                cc_state1, cc_state2 = self.env.column_crossover(
                    copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                )
            elif crossover_type == 1:
                # block crossover
                cc_state1, cc_state2 = self.env.block_crossover(
                    copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                )
            if cc_state1 is not None:
                crossover_states.append(cc_state1)
            if cc_state2 is not None:
                crossover_states.append(cc_state2)
            print(f"crossover states num: {len(crossover_states)}")
            if len(crossover_states) > 0:
                self.evaluate_ga(crossover_states, episode_num)

    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
        # pool_index = 0
        env_state, sel_index = self.env.reset(pool_index=pool_index)
        state = copy.deepcopy(env_state)
    
        # 2. sampling data
        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            self.total_steps += 1
            # 2.1 sampling real environment interaction
            is_model_evaluation = False

            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
            if is_model_evaluation:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
            else:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # 2.2 store real/imagined data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation) 
            elif self.store_type == "detail":        
                raise NotImplementedError
            # 2.3 update real/imagined initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

            loss = 0
            info = {}

            state = copy.deepcopy(next_state)
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

class AblationNoModel(AblationOnlyGA):
    def run_experiments(self):
        search_type = 1
        for episode_num in range(self.total_episodes):
            if episode_num < self.start_episodes:
                # 1. warm start
                self.run_episode(episode_num) # run env with random mutation
                if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                    self.end_experiments(episode_num)
            else:
                # 2. iterate between q policy and ga crossover
                if search_type == 1:
                    # 1) GA crossover
                    self.run_episode_ga(episode_num)
                elif search_type == -1:
                    # 2) q agent 
                    self.run_episode(episode_num)
                if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                    self.end_experiments(episode_num)
                search_type *= -1
        self.end_experiments(episode_num)

"""
    V22GARandom: multi-obj ga v22, with single Q; random column search and ga global exploration random selection
"""
class AlterSearchAndLearnV22RandomGAAlgorithm(AlterSearchAndLearnV22GAAlgorithm):
    def random_shooting(self, episode_num):
        # 1. _imagine_model_data search data randomly
        # set random q policy
        self.q_policy.EPS_END = 1
        self.q_policy.EPS_START = 2
        self._imagine_model_data(episode_num)
        self.q_policy.EPS_END = 0.1
        self.q_policy.EPS_START = 0.9
        # set policy back
        # 2. evaluate imagine model data and log ppa
        if self.is_value_function_mutation:
            self.evaluate(episode_num)
        # 2. evaluate imagine ga data and log ppa
        if self.is_crossover_variation:
            self.evaluate_ga(episode_num)    

"""
    V23GA: multi-obj ga v23, with single Q; q function local search and ga global exploration random selection
"""
class AlterSearchAndLearnV23GAAlgorithm(AlterSearchAndLearnV22GAAlgorithm):
    def _imagine_model_data(self, episode_num):
        if self.is_start_model:
            # 0. reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)): 
                self.imagine_found_best_ppa[i] = 10000
            
            imagine_type = np.random.choice([0,1])
            if imagine_type == 0:
                # 1. imagine with model
                if self.is_value_function_mutation:
                    for _ in range(self.num_sample):
                        pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
                        env_state, sel_index = self.env.reset(pool_index=pool_index)
                        state = copy.deepcopy(env_state)
                        for _ in range(self.depth):
                            steps += 1
                            action, policy_info = self.q_policy.select_action(
                                torch.tensor(state), 
                                self.total_steps, 
                                deterministic=self.deterministic,
                                is_softmax=self.imagine_is_softmax
                            )
                            # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                            next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                            _, next_state_policy_info = self.q_policy.select_action(
                                torch.tensor(next_state), self.total_steps, 
                                deterministic=self.deterministic,
                                is_softmax=self.imagine_is_softmax
                            )
                            # 2.2 store real/imagined data
                            self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                            
                            # 2.3 update real/imagined initial state pool
                            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                            state = copy.deepcopy(next_state)

                            avg_ppa = rewards_dict['avg_ppa']
                            logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                            logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)
            elif imagine_type == 1:
                if self.is_crossover_variation:
                    for i in range(len(self.imagine_found_best_ppa_ga)): 
                        self.imagine_found_best_ppa_ga[i] = 10000
                    # 2. GA Mutation from different env pools
                    for i in range(int(self.num_sample * self.depth / 2)):
                        env_state1, sel_index1 = self.env.reset(pool_index=0)
                        print(f"select first state index: {sel_index1}")
                        pool_index = np.random.choice([1,2,3])
                        env_state2, sel_index2 = self.env.reset(pool_index=pool_index)
                        
                        crossover_states = []
                        # column crossover 
                        print("before column crossover")
                        cc_state1, cc_state2 = self.env.column_crossover(
                            copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                        )
                        print("before block crossover")
                        # block crossover
                        bc_state1, bc_state2 = self.env.block_crossover(
                            copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                        )
                        if cc_state1 is not None:
                            crossover_states.append(cc_state1)
                        if cc_state2 is not None:
                            crossover_states.append(cc_state2)
                        if bc_state1 is not None:
                            crossover_states.append(bc_state1)
                        if bc_state2 is not None:
                            crossover_states.append(bc_state2)
                        print(f"crossover states num: {len(crossover_states)}")
                        if len(crossover_states) > 0:
                            for j in range(len(self.imagine_found_best_ppa)):
                                GA_found_state_num = 0
                                for state in crossover_states:
                                    logger.log(f"state: {state}")
                                    steps += 1
                                    if self.is_sr_model:
                                        avg_ppa, normalize_area, normalize_delay = self.evaluate_with_dsr_model(state, j)
                                    else:
                                        avg_ppa, normalize_area, normalize_delay = self.evaluate_with_model(state, j)
                                    is_found_better_state = self.update_imagine_pool(state, avg_ppa, normalize_area, normalize_delay, j)
                                    if is_found_better_state:
                                        GA_found_state_num += 1
                                    logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                                    logger.tb_logger.add_scalar(f'{j}-th preference avg ppa', avg_ppa, global_step=self.total_steps + steps)
                                logger.tb_logger.add_scalar(f'{j}-th preference mbrl model ga found best state num', GA_found_state_num, global_step=self.total_steps + steps)
                        else:
                            print(f"warning!!!no valid crossover states found next iteration") 
        return imagine_type

    def random_shooting(self, episode_num):
        # 1. _imagine_model_data search data randomly
        imagine_type = self._imagine_model_data(episode_num)
        # 2. evaluate imagine model data and log ppa
        if imagine_type == 0:
            if self.is_value_function_mutation:
                self.evaluate(episode_num)
        elif imagine_type == 1:
            # 2. evaluate imagine ga data and log ppa
            if self.is_crossover_variation:
                self.evaluate_ga(episode_num)

"""
    V22GA ablation study: multi-obj ga v22, with single Q, no q no ga; 
"""
class AlterSearchAndLearnV22NoQNoGAAlgorithm(AlterSearchAndLearnV22GAAlgorithm):
    def _imagine_model_data(self, episode_num):
        if self.is_start_model:
            # 0. reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)): 
                self.imagine_found_best_ppa[i] = 10000
            # 1. imagine with model
            if self.is_value_function_mutation:
                for _ in range(self.num_sample):
                    pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
                    env_state, sel_index = self.env.reset(pool_index=pool_index)
                    state = copy.deepcopy(env_state)
                    for _ in range(self.depth):
                        steps += 1
                        action, policy_info = self.q_policy.select_action(
                            torch.tensor(state), 
                            self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                        next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                        _, next_state_policy_info = self.q_policy.select_action(
                            torch.tensor(next_state), self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # 2.2 store real/imagined data
                        self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                        
                        # 2.3 update real/imagined initial state pool
                        self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                        state = copy.deepcopy(next_state)

                        avg_ppa = rewards_dict['avg_ppa']
                        logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)

    def random_shooting(self, episode_num):
        # set random policy
        self.q_policy.EPS_END = 1
        self.q_policy.EPS_START = 2
        # 1. _imagine_model_data search data randomly
        self._imagine_model_data(episode_num)
        # 2. evaluate imagine model data and log ppa
        if self.is_value_function_mutation:
            self.evaluate(episode_num)
        # set policy back
        self.q_policy.EPS_END = 0.1
        self.q_policy.EPS_START = 0.9

"""
    V22GA ablation study: multi-obj ga v22, with single Q, no hybrid; 
"""
class AlterSearchAndLearnV22NoHybridGAAlgorithm(AlterSearchAndLearnV22GAAlgorithm):
    def update_imagine_pool(self, state, avg_ppa, normalize_area, normalize_delay, j):
        if avg_ppa < self.found_best_info[j]['found_best_ppa']:
            # imagine ga pool
            self.env.initial_state_pool[j].append(
                {
                    "state": copy.deepcopy(state),
                    "area": 0,
                    "delay": 0,
                    "ppa": avg_ppa,
                    "count": 1,
                    "state_mask": None,
                    "state_type": "best_ppa",
                    "normalize_area": normalize_area,
                    "normalize_delay": normalize_delay
                }
            )
            # self.imagine_found_best_ppa_ga[j] = avg_ppa
            return True
        else:
            return False
        
    def _imagine_model_data(self, episode_num):
        if self.is_start_model:
            # 0. reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)): 
                self.imagine_found_best_ppa[i] = 10000
            # 1. imagine with model
            if self.is_value_function_mutation:
                for _ in range(self.num_sample):
                    pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
                    env_state, sel_index = self.env.reset(pool_index=pool_index)
                    state = copy.deepcopy(env_state)
                    for _ in range(self.depth):
                        steps += 1
                        action, policy_info = self.q_policy.select_action(
                            torch.tensor(state), 
                            self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                        next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                        _, next_state_policy_info = self.q_policy.select_action(
                            torch.tensor(next_state), self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # 2.2 store real/imagined data
                        self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True) 
                        
                        # 2.3 update real/imagined initial state pool
                        self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=False)

                        state = copy.deepcopy(next_state)

                        avg_ppa = rewards_dict['avg_ppa']
                        logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)
            if self.is_crossover_variation:
                for i in range(len(self.imagine_found_best_ppa_ga)): 
                    self.imagine_found_best_ppa_ga[i] = 10000
                # 2. GA Mutation from different env pools
                for i in range(int(self.num_sample * self.depth / 2)):
                    env_state1, sel_index1 = self.env.reset(pool_index=0)
                    print(f"select first state index: {sel_index1}")
                    pool_index = np.random.choice([1,2,3])
                    env_state2, sel_index2 = self.env.reset(pool_index=pool_index)
                    
                    crossover_states = []
                    # column crossover 
                    print("before column crossover")
                    cc_state1, cc_state2 = self.env.column_crossover(
                        copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                    )
                    print("before block crossover")
                    # block crossover
                    bc_state1, bc_state2 = self.env.block_crossover(
                        copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                    )
                    if cc_state1 is not None:
                        crossover_states.append(cc_state1)
                    if cc_state2 is not None:
                        crossover_states.append(cc_state2)
                    if bc_state1 is not None:
                        crossover_states.append(bc_state1)
                    if bc_state2 is not None:
                        crossover_states.append(bc_state2)
                    print(f"crossover states num: {len(crossover_states)}")
                    if len(crossover_states) > 0:
                        for j in range(len(self.imagine_found_best_ppa)):
                            GA_found_state_num = 0
                            for state in crossover_states:
                                logger.log(f"state: {state}")
                                steps += 1
                                if self.is_sr_model:
                                    avg_ppa, normalize_area, normalize_delay = self.evaluate_with_dsr_model(state, j)
                                else:
                                    avg_ppa, normalize_area, normalize_delay = self.evaluate_with_model(state, j)
                                is_found_better_state = self.update_imagine_pool(state, avg_ppa, normalize_area, normalize_delay, j)
                                if is_found_better_state:
                                    GA_found_state_num += 1
                                logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                                logger.tb_logger.add_scalar(f'{j}-th preference avg ppa', avg_ppa, global_step=self.total_steps + steps)
                            logger.tb_logger.add_scalar(f'{j}-th preference mbrl model ga found best state num', GA_found_state_num, global_step=self.total_steps + steps)
                    else:
                        print(f"warning!!!no valid crossover states found next iteration") 

    def random_shooting(self, episode_num):
        # 1. _imagine_model_data search data randomly
        self._imagine_model_data(episode_num)
        # # 2. evaluate imagine model data and log ppa
        # if self.is_value_function_mutation:
        #     self.evaluate(episode_num)
        # # 2. evaluate imagine ga data and log ppa
        # if self.is_crossover_variation:
        #     self.evaluate_ga(episode_num)

"""
    V3GA: multi-obj ga, with multi Q
"""
class AlterSearchAndLearnV3GAAlgorithm(AlterSearchAndLearnV2GAAlgorithm):
    def __init__(
        self,
        env,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        # model-based kwargs
        ppa_model, 
        imagined_replay_memory,
        **mbrl_alg_kwargs
    ):
        super(AlterSearchAndLearnV3GAAlgorithm, self).__init__(
            env, q_policy, target_q_policy, replay_memory,
            rnd_predictor, rnd_target, int_reward_run_mean_std,
            ppa_model, imagined_replay_memory,
            **mbrl_alg_kwargs
        )
        self.policy_optimizer = []
        self.q_policy = [
            copy.deepcopy(self.q_policy) for _ in range(len(self.env.weight_list))
        ]
        self.target_q_policy = [
            copy.deepcopy(self.target_q_policy) for _ in range(len(self.env.weight_list))
        ]
        for i in range(len(self.env.weight_list)):
            self.policy_optimizer.append(
                self.optimizer_class(
                    self.q_policy[i].parameters(),
                    lr=self.q_net_lr
                )
            )

    def evaluate(self, episode_num):
        # TODO: evaluate 函数要改一下，去evaluate虚拟的pool
        for j in range(len(self.env.weight_list)):
            evaluate_num = min(self.evaluate_imagine_state_num, len(self.env.imagined_initial_state_pool[j]))
            true_found_best_ppa = self.found_best_info[j]['found_best_ppa']
            last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
            cur_found_best_ppa = 10000
            newly_found_best_state = 0
            for i in range(1, evaluate_num+1):
                test_state = copy.deepcopy(
                    self.env.imagined_initial_state_pool[j][-i]["state"]
                )
                # get avg ppa
                if self.is_mac:
                    initial_partial_product = MacPartialProduct[self.bit_width]
                else:
                    initial_partial_product = PartialProduct[self.bit_width]
                ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
                rewards_dict = self.env.get_reward()
                avg_area = np.mean(rewards_dict['area'])
                avg_delay = np.mean(rewards_dict['delay'])
                avg_ppa = self.env.weight_list[j][0] * avg_area / self.env.wallace_area + \
                    self.env.weight_list[j][1] * avg_delay / self.env.wallace_delay
                avg_ppa = avg_ppa * self.env.ppa_scale
                normalize_area = self.env.ppa_scale * (avg_area / self.env.wallace_area)
                normalize_delay = self.env.ppa_scale * (avg_delay / self.env.wallace_delay)
                if avg_ppa < cur_found_best_ppa:
                    cur_found_best_ppa = avg_ppa
                
                if avg_ppa < last_found_best_ppa:
                    newly_found_best_state += 1
                    cur_found_best_ppa = avg_ppa
                    _, state_policy_info = self.q_policy[j].select_action(
                        torch.tensor(test_state), 0, 
                        deterministic=False,
                        is_softmax=False
                    )
                    state_mask = state_policy_info['mask']
                    self.env.initial_state_pool[j].append(
                                {
                                    "state": copy.deepcopy(test_state),
                                    "area": avg_area,
                                    "delay": avg_delay,
                                    "ppa": avg_ppa,
                                    "count": 1,
                                    "state_mask": state_mask,
                                    "state_type": "best_ppa",
                                    "normalize_area": normalize_area,
                                    "normalize_delay": normalize_delay
                                }
                            )
                    last_found_best_ppa = self.env.initial_state_pool[j][0]["ppa"]
                
                # update best ppa found
                if avg_ppa < true_found_best_ppa:
                    true_found_best_ppa = avg_ppa
                    self.found_best_info[j]["found_best_ppa"] = avg_ppa
                    self.found_best_info[j]["found_best_state"] = copy.deepcopy(test_state)
                    self.found_best_info[j]["found_best_area"] = avg_area 
                    self.found_best_info[j]["found_best_delay"] = avg_delay

            logger.tb_logger.add_scalar(f'mbrl model true best ppa found {j}-th env pool', cur_found_best_ppa, global_step=episode_num)
            logger.tb_logger.add_scalar(f'mbrl model evaluate imagine state num {j}-th env pool', evaluate_num, global_step=episode_num)
            logger.tb_logger.add_scalar(f'mbrl model found best ppa state num {j}-th env pool', newly_found_best_state, global_step=episode_num)
            logger.tb_logger.add_scalar(f'mbrl model total best ppa {j}-th env pool', true_found_best_ppa, global_step=episode_num)

    def log_stats(
        self, loss, reward, rewards_dict,
        next_state, action, info, policy_info, action_column=0
    ):
        try:
            loss = loss.item()
            for i in range(len(self.env.weight_list)):
                q_values = np.mean(info[i]['q_values'])
                target_q_values = np.mean(info[i]['target_q_values'])
                positive_rewards_number = info[i]['positive_rewards_number']
                # log q values info 
                logger.tb_logger.add_scalar(f'q_values_{i}', q_values, global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'target_q_values_{i}', target_q_values, global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'positive_rewards_number_{i}', positive_rewards_number, global_step=self.total_steps)
        except:
            loss = loss
            q_values = 0.
            target_q_values = 0.
            positive_rewards_number = 0.
            # log q values info 
            logger.tb_logger.add_scalar('q_values', q_values, global_step=self.total_steps)
            logger.tb_logger.add_scalar('target_q_values', target_q_values, global_step=self.total_steps)
            logger.tb_logger.add_scalar('positive_rewards_number', positive_rewards_number, global_step=self.total_steps)

        logger.tb_logger.add_scalar('train loss', loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar('reward', reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('legal num column', rewards_dict['legal_num_column'], global_step=self.total_steps)
        if "legal_num_stage" in rewards_dict.keys():
            logger.tb_logger.add_scalar('legal num stage', rewards_dict['legal_num_stage'], global_step=self.total_steps)  
        if "legal_num_column_pp3" in rewards_dict.keys():
            logger.tb_logger.add_scalar('legal_num_column_pp3', rewards_dict['legal_num_column_pp3'], global_step=self.total_steps)  
            logger.tb_logger.add_scalar('legal_num_column_pp0', rewards_dict['legal_num_column_pp0'], global_step=self.total_steps)  
            
        if len(action.shape) <= 2:
            logger.tb_logger.add_scalar('action_index', action, global_step=self.total_steps)
        if policy_info is not None:
            logger.tb_logger.add_scalar('stage_num', policy_info["stage_num"], global_step=self.total_steps)
            logger.tb_logger.add_scalar('eps_threshold', policy_info["eps_threshold"], global_step=self.total_steps)
        
        logger.tb_logger.add_scalar('action_column', action_column, global_step=self.total_steps)
        
        for i in range(len(self.found_best_info)):
            logger.tb_logger.add_scalar(f'best ppa {i}-th weight', self.found_best_info[i]["found_best_ppa"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best area {i}-th weight', self.found_best_info[i]["found_best_area"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best delay {i}-th weight', self.found_best_info[i]["found_best_delay"], global_step=self.total_steps)

        # log state 
        # ct32, ct22 = self.build_state_dict(next_state)
        # logger.tb_logger.add_scalars(
        #     'compressor 32', ct32, global_step=self.total_steps)
        # logger.tb_logger.add_scalars(
        #     'compressor 22', ct22, global_step=self.total_steps)
        # logger.tb_logger.add_image(
        #     'state image', next_state, global_step=self.total_steps, dataformats='HW'
        # )

        # log wallace area wallace delay
        logger.tb_logger.add_scalar('wallace area', self.env.wallace_area, global_step=self.total_steps)
        logger.tb_logger.add_scalar('wallace delay', self.env.wallace_delay, global_step=self.total_steps)
        
        # log env initial state pool
        # if self.env.initial_state_pool_max_len > 0:
        #     avg_area, avg_delay = self.get_env_pool_log()
        #     logger.tb_logger.add_scalars(
        #         'env state pool area', avg_area, global_step=self.total_steps)
        #     logger.tb_logger.add_scalars(
        #         'env state pool delay', avg_delay, global_step=self.total_steps)

        # log mask stats
        self.log_mask_stats(policy_info)
        self.log_action_stats(action)

    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        pool_index = np.random.choice([i for i in range(len(self.env.weight_list))])
        env_state, sel_index = self.env.reset(pool_index=pool_index)
        state = copy.deepcopy(env_state)
    
        # 2. sampling data
        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # 2.1 sampling real environment interaction
            is_model_evaluation = False

            action, policy_info = self.q_policy[0].select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
            if is_model_evaluation:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
            else:
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
            _, next_state_policy_info = self.q_policy[0].select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # 2.2 store real/imagined data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], rewards_dict, is_model_evaluation=is_model_evaluation) 
            elif self.store_type == "detail":        
                raise NotImplementedError
            # 2.3 update real/imagined initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

            # 2.4 update q using mixed data for many steps
            loss, info = self.update_q()

            # 2.4 update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                for i in range(len(self.env.weight_list)):
                    self.target_q_policy[i].load_state_dict(
                        self.q_policy[i].state_dict()
                    )
            # 2.5 model-based rl
            if self.total_steps >= self.warm_start_steps:
                self.is_start_model = True
                # start model-based episode after warm start
                mb_info = self._run_mb_episode()
                self.log_mb_stats(mb_info)

            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

        # update target q
        for i in range(len(self.env.weight_list)):
            self.target_q_policy[i].load_state_dict(
                self.q_policy[i].state_dict()
            )

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        normalize_area, normalize_delay, rewards_dict,
        is_model_evaluation=False
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        if is_model_evaluation:
            self.imagined_replay_memory.push(
                torch.tensor(state),
                action,
                torch.tensor(next_state),
                torch.tensor([reward]),
                mask.reshape(1,-1),
                next_state_mask.reshape(1,-1),
                torch.tensor([normalize_area]),
                torch.tensor([normalize_delay]),
                torch.tensor([rewards_dict["area_reward"]]),
                torch.tensor([rewards_dict["delay_reward"]])        
            )
        else:
            self.replay_memory.push(
                torch.tensor(state),
                action,
                torch.tensor(next_state),
                torch.tensor([reward]),
                mask.reshape(1,-1),
                next_state_mask.reshape(1,-1),
                torch.tensor([normalize_area]),
                torch.tensor([normalize_delay]),
                torch.tensor([rewards_dict["area_reward"]]),
                torch.tensor([rewards_dict["delay_reward"]])
            )

    ### parallel ####
    def compute_values(
        self, state_batch, action_batch, state_mask, policy_index
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        states = []
        for i in range(batch_size):
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            states.append(state.unsqueeze(0))
        states = torch.cat(states)
        # compute image state
        if action_batch is not None:
            q_values = self.q_policy[policy_index](states.float(), state_mask=state_mask)
            q_values = q_values.reshape(-1, (int(self.int_bit_width*2))*4)           
            # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
            for i in range(batch_size):
                state_action_values[i] = q_values[i, action_batch[i]]
        else:
            q_values = self.target_q_policy[policy_index](states.float(), is_target=True, state_mask=state_mask)
            for i in range(batch_size):
                state_action_values[i] = q_values[i:i+1].max(1)[0].detach()
        return state_action_values

    def combine_batch(self, imagined_batch_transitions, real_batch_transitions):
        imagined_batch = MBRLMultiObjTransition(*zip(*imagined_batch_transitions))
        real_batch = MBRLMultiObjTransition(*zip(*real_batch_transitions))
        
        real_next_state_batch = torch.cat(real_batch.next_state)
        real_state_batch = torch.cat(real_batch.state)
        real_action_batch = torch.cat(real_batch.action)
        real_reward_batch = torch.cat(real_batch.reward)
        real_state_mask = torch.cat(real_batch.mask)
        real_next_state_mask = torch.cat(real_batch.next_state_mask)
        real_area_reward_batch = torch.cat(real_batch.area_reward)
        real_delay_reward_batch = torch.cat(real_batch.delay_reward)

        imagined_next_state_batch = torch.cat(imagined_batch.next_state)
        imagined_state_batch = torch.cat(imagined_batch.state)
        imagined_action_batch = torch.cat(imagined_batch.action)
        imagined_reward_batch = torch.cat(imagined_batch.reward)
        imagined_state_mask = torch.cat(imagined_batch.mask)
        imagined_next_state_mask = torch.cat(imagined_batch.next_state_mask)        
        imagined_area_reward_batch = torch.cat(imagined_batch.area_reward)
        imagined_delay_reward_batch = torch.cat(imagined_batch.delay_reward)

        next_state_batch = torch.cat(
            (real_next_state_batch, imagined_next_state_batch)
        )
        state_batch = torch.cat(
            (real_state_batch, imagined_state_batch)
        )
        action_batch = torch.cat(
            (real_action_batch, imagined_action_batch)
        )
        reward_batch = torch.cat(
            (real_reward_batch, imagined_reward_batch)
        )
        state_mask = torch.cat(
            (real_state_mask, imagined_state_mask)
        )
        next_state_mask = torch.cat(
            (real_next_state_mask, imagined_next_state_mask)
        )
        area_reward_batch = torch.cat(
            (real_area_reward_batch, imagined_area_reward_batch)
        )
        delay_reward_batch = torch.cat(
            (real_delay_reward_batch, imagined_delay_reward_batch)
        )
        return next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask, area_reward_batch, delay_reward_batch
   
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else: 
            if not self.is_start_model or len(self.imagined_replay_memory) == 0 or not self.use_dyna_data:
                # updating q using pure real data
                transitions = self.replay_memory.sample(self.batch_size)
                batch = MBRLMultiObjTransition(*zip(*transitions))
                next_state_batch = torch.cat(batch.next_state)
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_mask = torch.cat(batch.mask)
                next_state_mask = torch.cat(batch.next_state_mask)
                area_reward_batch = torch.cat(batch.area_reward)
                delay_reward_batch = torch.cat(batch.delay_reward)
                # update reward int run mean std
                self.update_reward_int_run_mean_std(
                    reward_batch.cpu().numpy()
                )
                # compute reward int 
                int_rewards_batch = self.compute_int_rewards(
                    next_state_batch, next_state_mask
                )
                int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                info = []
                for i in range(len(self.env.weight_list)):
                    train_reward_batch = self.env.weight_list[i][0] * area_reward_batch + self.env.weight_list[i][1] * delay_reward_batch
                    train_reward_batch = train_reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = self.compute_values(
                        state_batch, action_batch, state_mask, i
                    )
                    next_state_values = self.compute_values(
                        next_state_batch, None, next_state_mask, i
                    )
                    target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                    loss = self.loss_fn(
                        state_action_values.unsqueeze(1), 
                        target_state_action_values.unsqueeze(1)
                    )

                    self.policy_optimizer[i].zero_grad()
                    loss.backward()
                    for param in self.q_policy[i].parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.policy_optimizer[i].step()

                    info.append(
                        {
                            "q_values": state_action_values.detach().cpu().numpy(),
                            "target_q_values": target_state_action_values.detach().cpu().numpy(),
                            "positive_rewards_number": torch.sum(torch.gt(train_reward_batch.cpu(), 0).float())
                        }
                    )
                self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())
            else:
                for _ in range(self.train_q_steps):
                    imagined_batch_size = min(len(self.imagined_replay_memory), int((1-self.real_data_ratio)*self.batch_size))
                    real_batch_size = self.batch_size - imagined_batch_size
                    imagined_batch_transitions = self.imagined_replay_memory.sample(imagined_batch_size)
                    real_batch_transitions = self.replay_memory.sample(real_batch_size)
                    next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask, area_reward_batch, delay_reward_batch = self.combine_batch(
                        imagined_batch_transitions, real_batch_transitions
                    )
                    # update reward int run mean std
                    self.update_reward_int_run_mean_std(
                        reward_batch.cpu().numpy()
                    )

                    # compute reward int 
                    int_rewards_batch = self.compute_int_rewards(
                        next_state_batch, next_state_mask
                    )
                    int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                    info = []
                    for i in range(len(self.env.weight_list)):
                        train_reward_batch = self.env.weight_list[i][0] * area_reward_batch + self.env.weight_list[i][1] * delay_reward_batch
                        train_reward_batch = train_reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                        # columns of actions taken. These are the actions which would've been taken
                        # for each batch state according to policy_net
                        state_action_values = self.compute_values(
                            state_batch, action_batch, state_mask, i
                        )
                        next_state_values = self.compute_values(
                            next_state_batch, None, next_state_mask, i
                        )
                        target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                        loss = self.loss_fn(
                            state_action_values.unsqueeze(1), 
                            target_state_action_values.unsqueeze(1)
                        )

                        self.policy_optimizer[i].zero_grad()
                        loss.backward()
                        for param in self.q_policy[i].parameters():
                            param.grad.data.clamp_(-1, 1)
                        self.policy_optimizer[i].step()
                        info.append(
                            {
                                "q_values": state_action_values.detach().cpu().numpy(),
                                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                                "positive_rewards_number": torch.sum(torch.gt(train_reward_batch.cpu(), 0).float())
                            }
                        )
                    self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                    self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())
            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info

    def _imagine_model_data(self, episode_num):
        if self.is_start_model:
            # reset imagine found best ppa
            steps = 0
            for i in range(len(self.imagine_found_best_ppa)):
                self.imagine_found_best_ppa[i] = 10000
                for _ in range(self.num_sample):
                    env_state, sel_index = self.env.reset(pool_index=i)
                    state = copy.deepcopy(env_state)
                    for _ in range(self.depth):
                        steps += 1
                        action, policy_info = self.q_policy[i].select_action(
                            torch.tensor(state), 
                            self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                        next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                        _, next_state_policy_info = self.q_policy[i].select_action(
                            torch.tensor(next_state), self.total_steps, 
                            deterministic=self.deterministic,
                            is_softmax=self.imagine_is_softmax
                        )
                        # 2.2 store real/imagined data
                        self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], rewards_dict, is_model_evaluation=True) 
                        
                        # 2.3 update real/imagined initial state pool
                        self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                        state = copy.deepcopy(next_state)

                        avg_ppa = rewards_dict['avg_ppa']
                        logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps + steps)

            # 2. GA Mutation from different env pools
            for i in range(int(self.num_sample * self.depth / 2)):
                indexes_list = [i for i in range(len(self.env.weight_list))]
                sel_pool_indexes = np.random.choice(indexes_list, size=2, replace=False)
                env_state1, sel_index1 = self.env.reset(pool_index=sel_pool_indexes[0])
                print(f"select first state index: {sel_index1}")
                env_state2, sel_index2 = self.env.reset(pool_index=sel_pool_indexes[1])
                
                crossover_states = []
                # column crossover 
                print("before column crossover")
                cc_state1, cc_state2 = self.env.column_crossover(
                    copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                )
                print("before block crossover")
                # block crossover
                bc_state1, bc_state2 = self.env.block_crossover(
                    copy.deepcopy(env_state1), copy.deepcopy(env_state2)
                )
                if cc_state1 is not None:
                    crossover_states.append(cc_state1)
                if cc_state2 is not None:
                    crossover_states.append(cc_state2)
                if bc_state1 is not None:
                    crossover_states.append(bc_state1)
                if bc_state2 is not None:
                    crossover_states.append(bc_state2)
                print(f"crossover states num: {len(crossover_states)}")
                if len(crossover_states) > 0:
                    for j in range(len(self.imagine_found_best_ppa)):
                        GA_found_state_num = 0
                        for state in crossover_states:
                            logger.log(f"state: {state}")
                            steps += 1
                            avg_ppa, normalize_area, normalize_delay = self.evaluate_with_model(state, j)
                            is_found_better_state = self.update_imagine_pool(state, avg_ppa, normalize_area, normalize_delay, j)
                            if is_found_better_state:
                                GA_found_state_num += 1
                            logger.log(f"total steps: {self.total_steps + steps}, avg ppa: {avg_ppa}")
                            logger.tb_logger.add_scalar(f'{j}-th preference avg ppa', avg_ppa, global_step=self.total_steps + steps)
                        logger.tb_logger.add_scalar(f'{j}-th preference mbrl model ga found best state num', GA_found_state_num, global_step=episode_num)
                else:
                    print(f"warning!!!no valid crossover states found next iteration") 

    def _train_model(self):
        # set_trace()
        loss_area_total = []
        loss_delay_total = []
        lr_decay = False
        if self.is_first_model_train:
            self.is_first_model_train = False
            if self.lr_decay:
                lr_decay = True
            train_steps = self.train_model_start_steps
        else:
            train_steps = self.train_model_finetune_steps
            # if self.lr_decay:
            #     self.model_lr_scheduler = lr_scheduler.StepLR(
            #         self.ppa_model_optimizer,
            #         self.lr_decay_step_finetune,
            #         gamma=self.lr_decay_rate
            #     )
        for train_step in range(train_steps):
            # sample a batch
            transitions = self.replay_memory.sample(self.train_model_batch_size)
            batch = MBRLMultiObjTransition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            normalize_area_batch = torch.cat(batch.normalize_area).float().to(self.device)
            normalize_delay_batch = torch.cat(batch.normalize_delay).float().to(self.device)

            batch_num = next_state_batch.shape[0]
            area_loss = torch.zeros(batch_num, device=self.device)
            delay_loss = torch.zeros(batch_num, device=self.device)
            states = []
            for j in range(batch_num):
                # !!! 耗时，重复调用，可工程优化
                ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, next_state_batch[j].cpu().numpy())
                ct32 = torch.tensor(np.array([ct32]))
                ct22 = torch.tensor(np.array([ct22]))
                if stage_num < self.MAX_STAGE_NUM-1:
                    zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                    ct32 = torch.cat((ct32, zeros), dim=1)
                    ct22 = torch.cat((ct22, zeros), dim=1)
                state = torch.cat((ct32, ct22), dim=0)
                states.append(state.unsqueeze(0))
            states = torch.cat(states)
            predict_area, predict_delay = self.ppa_model(states.float())
            predict_area = predict_area.squeeze()
            predict_delay = predict_delay.squeeze()

            # 可能出问题一
            print(f"predict area shape: {predict_area.shape}")
            print(f"predict delay shape: {predict_delay.shape}")
            print(f"normalize area shape: {normalize_area_batch.shape}")
            print(f"normalize delay shape: {normalize_delay_batch.shape}")
            
            area_loss = self.mse_loss(
                predict_area, normalize_area_batch
            )
            delay_loss = self.mse_loss(
                predict_delay, normalize_delay_batch
            )
            loss_area = torch.mean(area_loss)
            loss_delay = torch.mean(delay_loss)
            loss = loss_area + loss_delay
            self.ppa_model_optimizer.zero_grad()
            loss.backward()
            for param in self.ppa_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.ppa_model_optimizer.step()
            loss_area_total.append(loss_area.item())
            loss_delay_total.append(loss_delay.item())
            if lr_decay:
                self.model_lr_scheduler.step()

            logger.tb_logger.add_scalar('mbrl model area loss each step', loss_area.item(), global_step=self.train_model_total_steps)
            logger.tb_logger.add_scalar('mbrl model delay loss each step', loss_delay.item(), global_step=self.train_model_total_steps)
            self.train_model_total_steps += 1
        info = {
            "loss_area_total": loss_area_total,
            "loss_delay_total": loss_delay_total
        }
        return loss, info

class TrajectorySearchAlgorithm(MBRLRNDDQNAlgorithm):
    def _train_model(self):
        # set_trace()
        loss_area_total = []
        loss_delay_total = []
        lr_decay = False
        if self.is_first_model_train:
            self.is_first_model_train = False
            if self.lr_decay:
                lr_decay = True
            train_steps = self.train_model_start_steps
        else:
            train_steps = self.train_model_finetune_steps
            # if self.lr_decay:
            #     self.model_lr_scheduler = lr_scheduler.StepLR(
            #         self.ppa_model_optimizer,
            #         self.lr_decay_step_finetune,
            #         gamma=self.lr_decay_rate
            #     )
        for train_step in range(train_steps):
            # sample a batch
            transitions = self.replay_memory.sample(self.train_model_batch_size)
            next_state_batch = []
            normalize_area_batch = []
            normalize_delay_batch = []
            for listtransitions in transitions:
                batch = MBRLTransition(*zip(*listtransitions))
                next_state_batch.append(torch.cat(batch.next_state))
                normalize_area_batch.append(torch.cat(batch.normalize_area))
                normalize_delay_batch.append(torch.cat(batch.normalize_delay))
            next_state_batch = torch.cat(next_state_batch)
            normalize_area_batch = torch.cat(normalize_area_batch).float().to(self.device)
            normalize_delay_batch = torch.cat(normalize_delay_batch).float().to(self.device)

            batch_num = next_state_batch.shape[0]
            area_loss = torch.zeros(batch_num, device=self.device)
            delay_loss = torch.zeros(batch_num, device=self.device)
            states = []
            for j in range(batch_num):
                # !!! 耗时，重复调用，可工程优化
                ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, next_state_batch[j].cpu().numpy())
                ct32 = torch.tensor(np.array([ct32]))
                ct22 = torch.tensor(np.array([ct22]))
                if stage_num < self.MAX_STAGE_NUM-1:
                    zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                    ct32 = torch.cat((ct32, zeros), dim=1)
                    ct22 = torch.cat((ct22, zeros), dim=1)
                state = torch.cat((ct32, ct22), dim=0)
                states.append(state.unsqueeze(0))
            states = torch.cat(states)
            predict_area, predict_delay = self.ppa_model(states.float())
            predict_area = predict_area.squeeze()
            predict_delay = predict_delay.squeeze()

            # 可能出问题一
            print(f"predict area shape: {predict_area.shape}")
            print(f"predict delay shape: {predict_delay.shape}")
            print(f"normalize area shape: {normalize_area_batch.shape}")
            print(f"normalize delay shape: {normalize_delay_batch.shape}")
            
            area_loss = self.mse_loss(
                predict_area, normalize_area_batch
            )
            delay_loss = self.mse_loss(
                predict_delay, normalize_delay_batch
            )
            loss_area = torch.mean(area_loss)
            loss_delay = torch.mean(delay_loss)
            loss = loss_area + loss_delay
            self.ppa_model_optimizer.zero_grad()
            loss.backward()
            for param in self.ppa_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.ppa_model_optimizer.step()
            loss_area_total.append(loss_area.item())
            loss_delay_total.append(loss_delay.item())
            if lr_decay:
                self.model_lr_scheduler.step()

            logger.tb_logger.add_scalar('mbrl model area loss each step', loss_area.item(), global_step=self.train_model_total_steps)
            logger.tb_logger.add_scalar('mbrl model delay loss each step', loss_delay.item(), global_step=self.train_model_total_steps)
            self.train_model_total_steps += 1
        info = {
            "loss_area_total": loss_area_total,
            "loss_delay_total": loss_delay_total
        }
        return loss, info
    
    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        normalize_area, normalize_delay,
        is_model_evaluation=False, 
        is_last=False
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,int(self.int_bit_width*2)))
        next_state = np.reshape(next_state, (1,2,int(self.int_bit_width*2)))
        if is_model_evaluation:
            self.imagined_replay_memory.push(
                is_last,
                torch.tensor(state),
                action,
                torch.tensor(next_state),
                torch.tensor([reward]),
                mask.reshape(1,-1),
                next_state_mask.reshape(1,-1),
                torch.tensor([normalize_area]),
                torch.tensor([normalize_delay])                
            )
        else:
            self.replay_memory.push(
                is_last,
                torch.tensor(state),
                action,
                torch.tensor(next_state),
                torch.tensor([reward]),
                mask.reshape(1,-1),
                next_state_mask.reshape(1,-1),
                torch.tensor([normalize_area]),
                torch.tensor([normalize_delay])
            )

    def _imagine_model_data(self, root_state, sel_index):
        # set_trace()
        # reset imagine found best ppa
        self.imagine_found_best_ppa = 10000
        best_trajectory_actions = []
        best_trajectory = []
        best_ppa_found = 10000       
        for _ in range(self.num_sample):
            # set root state
            self.env.set(root_state, sel_index)
            state = copy.deepcopy(root_state)
            # sample a trajectory
            trajectory_actions = []
            trajectory = {
                "state": [],
                "next_state": [],
                "action": [],
                "mask": [],
                "next_mask": []
            }
            trajectory_best_ppa = 10000
            for step in range(self.trajectory_step_num):
                # self.total_steps += 1
                action, policy_info = self.q_policy.select_action(
                    torch.tensor(state), 
                    self.total_steps, 
                    deterministic=self.deterministic,
                    is_softmax=self.imagine_is_softmax
                )
                trajectory_actions.append(action)

                # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=True, ppa_model=self.ppa_model)
                _, next_state_policy_info = self.q_policy.select_action(
                    torch.tensor(next_state), self.total_steps, 
                    deterministic=self.deterministic,
                    is_softmax=self.imagine_is_softmax
                )
                # 2.2 store real/imagined data
                is_last = False
                if (step + 1) % self.trajectory_step_num == 0:
                    is_last = True
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True, is_last=is_last) 
                else:
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=True, is_last=is_last)
                trajectory["state"].append(state)
                trajectory["next_state"].append(next_state)
                trajectory["action"].append(action)
                trajectory["mask"].append(policy_info['mask'])
                trajectory["next_mask"].append(next_state_policy_info['mask'])
                # 2.3 update real/imagined initial state pool
                self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=True)

                state = copy.deepcopy(next_state)

                avg_ppa = rewards_dict['avg_ppa']
                if avg_ppa < trajectory_best_ppa:
                    trajectory_best_ppa = avg_ppa
                logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
                logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps)
            
            if trajectory_best_ppa < best_ppa_found:
                best_trajectory_actions = trajectory_actions
                best_ppa_found = trajectory_best_ppa
                best_trajectory = trajectory

        return best_trajectory_actions, best_trajectory

    def combine_batch(self, imagined_batch_transitions, real_batch_transitions):
        # get imagine batches
        imagined_next_state_batch = []
        imagined_state_batch = []
        imagined_action_batch = []
        imagined_reward_batch = []
        imagined_state_mask = []
        imagined_next_state_mask = []
        for listtransitions in imagined_batch_transitions:
            batch = MBRLTransition(*zip(*listtransitions)) # namedtuple, dict of tuple
            imagined_next_state_batch.append(torch.cat(batch.next_state))
            imagined_state_batch.append(torch.cat(batch.state))
            imagined_action_batch.append(torch.cat(batch.action))
            imagined_reward_batch.append(torch.cat(batch.reward))
            imagined_state_mask.append(torch.cat(batch.mask))
            imagined_next_state_mask.append(torch.cat(batch.next_state_mask))
        imagined_next_state_batch = torch.cat(imagined_next_state_batch)
        imagined_state_batch = torch.cat(imagined_state_batch)
        imagined_action_batch = torch.cat(imagined_action_batch)
        imagined_reward_batch = torch.cat(imagined_reward_batch)
        imagined_state_mask = torch.cat(imagined_state_mask)
        imagined_next_state_mask = torch.cat(imagined_next_state_mask)

        # get real batches
        real_next_state_batch = []
        real_state_batch = []
        real_action_batch = []
        real_reward_batch = []
        real_state_mask = []
        real_next_state_mask = []
        for listtransitions in real_batch_transitions:
            batch = MBRLTransition(*zip(*listtransitions)) # namedtuple, dict of tuple
            real_next_state_batch.append(torch.cat(batch.next_state))
            real_state_batch.append(torch.cat(batch.state))
            real_action_batch.append(torch.cat(batch.action))
            real_reward_batch.append(torch.cat(batch.reward))
            real_state_mask.append(torch.cat(batch.mask))
            real_next_state_mask.append(torch.cat(batch.next_state_mask))
        real_next_state_batch = torch.cat(real_next_state_batch)
        real_state_batch = torch.cat(real_state_batch)
        real_action_batch = torch.cat(real_action_batch)
        real_reward_batch = torch.cat(real_reward_batch)
        real_state_mask = torch.cat(real_state_mask)
        real_next_state_mask = torch.cat(real_next_state_mask)

        # combine real and imagined batch
        next_state_batch = torch.cat(
            (real_next_state_batch, imagined_next_state_batch)
        )
        state_batch = torch.cat(
            (real_state_batch, imagined_state_batch)
        )
        action_batch = torch.cat(
            (real_action_batch, imagined_action_batch)
        )
        reward_batch = torch.cat(
            (real_reward_batch, imagined_reward_batch)
        )
        state_mask = torch.cat(
            (real_state_mask, imagined_state_mask)
        )
        next_state_mask = torch.cat(
            (real_next_state_mask, imagined_next_state_mask)
        )
        return next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask
    
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else: 
            if not self.is_start_model or len(self.imagined_replay_memory) == 0:
                # updating q using pure real data
                list_of_listtransitions = self.replay_memory.sample(self.batch_size)
                next_state_batch = []
                state_batch = []
                action_batch = []
                reward_batch = []
                state_mask = []
                next_state_mask = []
                for listtransitions in list_of_listtransitions:
                    batch = MBRLTransition(*zip(*listtransitions)) # namedtuple, dict of tuple

                    next_state_batch.append(torch.cat(batch.next_state))
                    state_batch.append(torch.cat(batch.state))
                    action_batch.append(torch.cat(batch.action))
                    reward_batch.append(torch.cat(batch.reward))
                    state_mask.append(torch.cat(batch.mask))
                    next_state_mask.append(torch.cat(batch.next_state_mask))
    
                next_state_batch = torch.cat(next_state_batch)
                state_batch = torch.cat(state_batch)
                action_batch = torch.cat(action_batch)
                reward_batch = torch.cat(reward_batch)
                state_mask = torch.cat(state_mask)
                next_state_mask = torch.cat(next_state_mask)

                # update reward int run mean std
                self.update_reward_int_run_mean_std(
                    reward_batch.cpu().numpy()
                )

                # compute reward int 
                int_rewards_batch = self.compute_int_rewards(
                    next_state_batch, next_state_mask
                )
                int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = self.compute_values(
                    state_batch, action_batch, state_mask
                )
                next_state_values = self.compute_values(
                    next_state_batch, None, next_state_mask
                )
                target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                loss = self.loss_fn(
                    state_action_values.unsqueeze(1), 
                    target_state_action_values.unsqueeze(1)
                )

                self.policy_optimizer.zero_grad()
                loss.backward()
                for param in self.q_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.policy_optimizer.step()

                info = {
                    "q_values": state_action_values.detach().cpu().numpy(),
                    "target_q_values": target_state_action_values.detach().cpu().numpy(),
                    "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
                }
                self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())
            else:
                for _ in range(self.train_q_steps):
                    imagined_batch_size = min(len(self.imagined_replay_memory), int((1-self.real_data_ratio)*self.batch_size))
                    real_batch_size = self.batch_size - imagined_batch_size
                    imagined_batch_transitions = self.imagined_replay_memory.sample(imagined_batch_size)
                    real_batch_transitions = self.replay_memory.sample(real_batch_size)
                    next_state_batch, state_batch, action_batch, reward_batch, state_mask, next_state_mask = self.combine_batch(
                        imagined_batch_transitions, real_batch_transitions
                    )
                    # update reward int run mean std
                    self.update_reward_int_run_mean_std(
                        reward_batch.cpu().numpy()
                    )

                    # compute reward int 
                    int_rewards_batch = self.compute_int_rewards(
                        next_state_batch, next_state_mask
                    )
                    int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
                    train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    state_action_values = self.compute_values(
                        state_batch, action_batch, state_mask
                    )
                    next_state_values = self.compute_values(
                        next_state_batch, None, next_state_mask
                    )
                    target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

                    loss = self.loss_fn(
                        state_action_values.unsqueeze(1), 
                        target_state_action_values.unsqueeze(1)
                    )

                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    for param in self.q_policy.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.policy_optimizer.step()

                    info = {
                        "q_values": state_action_values.detach().cpu().numpy(),
                        "target_q_values": target_state_action_values.detach().cpu().numpy(),
                        "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
                    }
                    self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
                    self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())            


            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_state_batch, next_state_mask
                )
        return loss, info
    
    def run_episode(self, episode_num):
        # 1. reset state
        state_value = 0.
        info_count = None
        env_state, sel_index = self.env.reset()
        state = copy.deepcopy(env_state)

        if not self.is_start_model:
            # 2. sampling data using Q policy
            for step in range(self.len_per_episode):
                logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
                logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
                if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                    logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                    if info_count is not None:
                        logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                        logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
                self.total_steps += 1
                # 2.1 sampling real environment interaction
                is_model_evaluation = False

                action, policy_info = self.q_policy.select_action(
                    torch.tensor(state), 
                    self.total_steps, 
                    deterministic=self.deterministic,
                    is_softmax=self.is_softmax
                )
                logger.log(f"total steps: {self.total_steps}, action: {action}")
                # environment interaction 前5步采集环境真实评估数据，后20步采集模型估计数据
                if is_model_evaluation:
                    next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation, ppa_model=self.ppa_model)
                else:
                    next_state, reward, rewards_dict = self.env.step(action, is_model_evaluation=is_model_evaluation)
                _, next_state_policy_info = self.q_policy.select_action(
                    torch.tensor(next_state), self.total_steps, 
                    deterministic=self.deterministic,
                    is_softmax=self.is_softmax
                )
                # 2.2 store real/imagined data
                # store data
                is_last = False
                if (step+1) % self.trajectory_step_num == 0:
                    is_last = True
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation, is_last=is_last)
                else:
                    self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_model_evaluation=is_model_evaluation, is_last=is_last)
                # 2.3 update real/imagined initial state pool
                self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'], is_model_evaluation=is_model_evaluation)

                # 2.4 update q using mixed data for many steps
                loss, info = self.update_q()

                # 2.4 update target q (TODO: SOFT UPDATE)
                if self.total_steps % self.target_update_freq == 0:
                    self.target_q_policy.load_state_dict(
                        self.q_policy.state_dict()
                    )

                state = copy.deepcopy(next_state)
                # reset agent 
                if self.agent_reset_freq > 0:
                    self.reset_agent()
                # log datasets
                self.log_stats(
                    loss, reward, rewards_dict,
                    next_state, action, info, policy_info
                )
                self.log_rnd_stats(info)
                
                avg_ppa = rewards_dict['avg_ppa']
                logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        else:
            # num_steps = int(self.len_per_episode / self.trajectory_step_num)
            # for _ in range(num_steps):
            # a. searching local trajectories using model
            local_trajectory_actions, local_trajectory = self._imagine_model_data(state, sel_index)

            # b. acting executing action sequences with the real environment
            self.env.set(state, sel_index)
            for step, action in enumerate(local_trajectory_actions):
                self.total_steps += 1
                next_state, reward, rewards_dict = self.env.step(action)

                state = local_trajectory["state"][step]
                next_state = local_trajectory["next_state"][step]
                mask = local_trajectory["mask"][step]
                next_mask = local_trajectory["next_mask"][step]
                
                is_last = False
                if (step+1) % self.trajectory_step_num == 0:
                    is_last = True
                    self.store(state, next_state, action, reward, mask, next_mask, rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_last=is_last)
                else:
                    self.store(state, next_state, action, reward, mask, next_mask, rewards_dict['normalize_area'], rewards_dict['normalize_delay'], is_last=is_last)
                self.update_env_initial_state_pool(next_state, rewards_dict, next_mask)
                avg_ppa = rewards_dict['avg_ppa']
                logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
                
            # c. updating Q 
            loss, info = self.update_q()
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)

            # d. log stats
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, None
            )
            self.log_rnd_stats(info)
        # 2.5 model-based rl train model
        if self.total_steps >= self.warm_start_steps:
            self.is_start_model = True
            # start model-based episode after warm start
            mb_info = self._run_mb_episode()
            self.log_mb_stats(mb_info)
        # evaluate imagine states
        if episode_num % self.evaluate_imagine_state_freq == 0:
            self.evaluate(episode_num)
            
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )

class RNDDQNAlgorithmWithPPAModel(RNDDQNAlgorithm):
    def run_experiments(self):
        for episode_num in range(self.total_episodes):
            self.run_episode(episode_num)
            if (episode_num+1) % self.evaluate_freq == 0:
                self.evaluate(episode_num)
        self.end_experiments(episode_num)

    def evaluate(self, episode_num):
        evaluate_num = min(self.evaluate_num, len(self.env.initial_state_pool))
        ppa_list = []
        for i in range(1, evaluate_num+1):
            test_state = copy.deepcopy(
                self.env.initial_state_pool[-i]["state"]
            )
            if self.is_mac:
                initial_partial_product = MacPartialProduct[self.bit_width]
            else:
                initial_partial_product = PartialProduct[self.bit_width]
            ct32, ct22, partial_products, stage_num = self.env.decompose_compressor_tree(initial_partial_product[:-1], test_state)
            rewards_dict = self.env.get_reward()
            avg_area = np.mean(rewards_dict['area'])
            avg_delay = np.mean(rewards_dict['delay'])
            avg_ppa = self.env.weight_area * avg_area / self.env.wallace_area + \
                self.env.weight_delay * avg_delay / self.env.wallace_delay
            avg_ppa = avg_ppa * self.env.ppa_scale
            ppa_list.append(avg_ppa)
        
        best_ppa = np.min(ppa_list)
        logger.tb_logger.add_scalar('true best ppa found', best_ppa, global_step=episode_num)
        
class RNDSeqDQNAlgorithm(RNDDQNAlgorithm):
    def compute_values(
        self, state_batch, action_batch, state_mask, is_average=False
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            seq_state = self.q_policy._process_state(
                 state_batch[i].cpu().numpy(), state_mask[i].numpy(), ct32, ct22
            )
            seq_state = torch.tensor(seq_state).unsqueeze(0).to(self.device)
            if action_batch is not None:
                # reshape 有问题************
                q_values = self.q_policy(seq_state.float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(seq_state.float(), is_target=True, state_mask=state_mask[i])
                # q_values = self.target_q_policy(state.unsqueeze(0))                
                if is_average:
                    q_values = (q_values + 1000).detach()
                    num = torch.count_nonzero(q_values)
                    state_action_values[i] = q_values.sum() / (num+1e-4)
                else:      
                    state_action_values[i] = q_values.max(1)[0].detach()
        return state_action_values

class RNDColumnDQNAlgorithm(RNDDQNAlgorithm):
    def compute_values(
        self, state_batch, action_batch, state_mask, is_average=False
    ):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            if action_batch is not None:
                q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2)))           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
                # q_values = self.target_q_policy(state.unsqueeze(0))                
                if is_average:
                    q_values = (q_values + 1000).detach()
                    num = torch.count_nonzero(q_values)
                    state_action_values[i] = q_values.sum() / (num+1e-4)
                else:      
                    state_action_values[i] = q_values.max(1)[0].detach()
        return state_action_values

    def run_episode(self, episode_num):
        # reset state
        state_value = 0.
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            self.total_steps += 1
            # environment interaction
            action, action_column, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # store data
            self.store(state, next_state, action_column, reward, policy_info['mask'], next_state_policy_info['mask'])
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, 
                policy_info, action_column=action_column
            )
            self.log_rnd_stats(info)
            avg_ppa = rewards_dict['avg_ppa']
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )


class RNDDQNAlgorithmWithPower(RNDDQNAlgorithm):
    def __init__(
            self,
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            rnd_predictor,
            rnd_target,
            int_reward_run_mean_std,
            **dqn_alg_kwargs):
        super().__init__(env, q_policy, target_q_policy, replay_memory, rnd_predictor, rnd_target, int_reward_run_mean_std, **dqn_alg_kwargs)

        self.pareto_pointset = {
            "area": [],
            "delay": [],
            "power": [],
            "state": [],
        }
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
            "found_best_power": 1e5,
        }

    def run_episode(self, episode_num):
        # reset state
        episode_area = []
        episode_delay = []
        episode_power = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "novelty_driven":
                state_novelty = self._get_env_pool_value_novelty("novelty")
                env_state, sel_index = self.env.reset(state_novelty=state_novelty)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["softmax_value_driven", "value_driven"]:
                state_value = self._get_env_pool_value_novelty("value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy in ["average_softmax_value_driven", "average_value_driven"]:
                state_value = self._get_env_pool_value_novelty("average_value")
                env_state, sel_index = self.env.reset(state_value=state_value)     
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "ppa_driven":
                state_value, info_count = self._get_env_pool_value_novelty("ppa_value")
                env_state, sel_index = self.env.reset(state_value=state_value)
                state = copy.deepcopy(env_state)
            elif self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
        
        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                torch.tensor(state), 
                self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                torch.tensor(next_state), self.total_steps, 
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask']) 
            elif self.store_type == "detail":        
                self.store_detail(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], policy_info['state_ct32'], policy_info['state_ct22'], next_state_policy_info['state_ct32'], next_state_policy_info['state_ct22'], rewards_dict)
                
            # update initial state pool
            self.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            state = copy.deepcopy(next_state)
            # reset agent 
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict,
                next_state, action, info, policy_info
            )
            self.log_rnd_stats(info)
            avg_ppa = rewards_dict['avg_ppa']
            episode_area.extend(rewards_dict["area"])
            episode_delay.extend(rewards_dict["delay"])
            episode_power.extend(rewards_dict["power"])
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
        # update target q
        self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_area, episode_delay, episode_power)

    def log_rnd_stats(
        self, info
    ):
        # int reward vs rewards
        logger.tb_logger.add_scalar('batch int rewards', self.rnd_int_rewards, global_step=self.total_steps)
        logger.tb_logger.add_scalar('batch ext rewards', self.rnd_ext_rewards, global_step=self.total_steps)
        logger.tb_logger.add_scalar('rnd loss', self.rnd_loss_item, global_step=self.total_steps)

    def process_and_log_pareto(self, episode_num, episode_area, episode_delay, episode_power):
        # 1. compute pareto pointset
        area_list, delay_list, power_list = episode_area, episode_delay, episode_power
        area_list.extend(self.pareto_pointset["area"])
        delay_list.extend(self.pareto_pointset["delay"])
        power_list.extend(self.pareto_pointset["power"])
        data_points = pd.DataFrame(
            {
                "area": area_list,
                "delay": delay_list,
                "power": power_list,
            }
        )
        pareto_mask = paretoset(data_points, sense=["min", "min", "min"])
        pareto_points = data_points[pareto_mask]
        new_pareto_area_list = pareto_points["area"].values.tolist()
        new_pareto_delay_list = pareto_points["delay"].values.tolist()
        new_pareto_power_list = pareto_points["power"].values.tolist()
        self.pareto_pointset["area"] = new_pareto_area_list
        self.pareto_pointset["delay"] = new_pareto_delay_list
        self.pareto_pointset["power"] = new_pareto_power_list
        
        # 2. compute hypervolume given pareto set and reference point
        pareto_point_array = self._combine()
        hv = hypervolume(pareto_point_array)
        hv_value = hv.compute(self.reference_point)
        logger.tb_logger.add_scalar('hypervolume', hv_value, global_step=episode_num)
        logger.log(f"episode {episode_num}, hypervolume: {hv_value}")

        # 3. log pareto points
        x = new_pareto_area_list
        y = new_pareto_delay_list
        z = new_pareto_power_list
        fig1 = plt.figure()
        f1 = plt.scatter(x, y, c='r')
        fig2 = plt.figure()
        f2 = plt.scatter(x, z, c='r')
        fig3 = plt.figure()
        f3 = plt.scatter(y, z, c='r')
        logger.tb_logger.add_figure('pareto points area-delay', fig1, global_step=episode_num)
        logger.tb_logger.add_figure('pareto points area-power', fig2, global_step=episode_num)
        logger.tb_logger.add_figure('pareto points delay-power', fig3, global_step=episode_num)

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the best ppa state into the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    avg_power = np.mean(rewards_dict['power'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "power": avg_power,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa",
                            "normalize_area": rewards_dict["normalize_area"],
                            "normalize_delay": rewards_dict["normalize_delay"],
                            "normalize_power": rewards_dict["normalize_power"],
                        }
                    )
            elif self.env.store_state_type == "leq":
                if self.found_best_info['found_best_ppa'] >= rewards_dict['avg_ppa']:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    avg_power = np.mean(rewards_dict['power'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "power": avg_power,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa"
                        }
                    )
            elif self.env.store_state_type == "ppa_with_diversity":
                # store best ppa state
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    avg_power = np.mean(rewards_dict['power'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "power": avg_power,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa"
                        }
                    )
                # store better diversity state
                elif rewards_dict['avg_ppa'] < self.env.initial_state_pool[0]["ppa"]:
                    number_states = len(self.env.initial_state_pool)
                    worse_state_distances = np.zeros(number_states-1)
                    cur_state_distances = np.zeros(number_states)
                    for i in range(number_states):
                        cur_state_dis = np.linalg.norm(
                            state - self.env.initial_state_pool[i]["state"],
                            ord=2
                        )
                        cur_state_distances[i] = cur_state_dis
                        if i == 0:
                            continue
                        worse_state_dis = np.linalg.norm(
                            self.env.initial_state_pool[0]["state"] - self.env.initial_state_pool[i]["state"],
                            ord=2
                        )
                        worse_state_distances[i-1] = worse_state_dis
 
                    worse_state_min_distance = np.min(worse_state_distances)
                    cur_state_min_distance = np.min(cur_state_distances)
                    if cur_state_min_distance > worse_state_min_distance:
                        # push the diverse state into the pool
                        avg_area = np.mean(rewards_dict['area'])
                        avg_delay = np.mean(rewards_dict['delay'])
                        avg_power = np.mean(rewards_dict['power'])
                        self.env.initial_state_pool.append(
                            {
                                "state": copy.deepcopy(state),
                                "area": avg_area,
                                "delay": avg_delay,
                                "power": avg_power,
                                "ppa": rewards_dict['avg_ppa'],
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "diverse"
                            }
                        )
        if self.found_best_info["found_best_ppa"] > rewards_dict['avg_ppa']:
            self.found_best_info["found_best_ppa"] = rewards_dict['avg_ppa']
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict['area']) 
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict['delay'])
            self.found_best_info["found_best_power"] = np.mean(rewards_dict['power'])

    def log_and_save_pareto_points(self, ppas_dict, episode_num):
        save_data_dict = {}
        # save ppa_csv
        save_data_dict["testing_full_ppa"] = ppas_dict
        # compute pareto points
        area_list = ppas_dict["area"]
        delay_list = ppas_dict["delay"]
        power_list = ppas_dict["power"]
        data_points = pd.DataFrame(
            {
                "area": area_list,
                "delay": delay_list,
                "power": power_list,
            }
        )
        pareto_mask = paretoset(data_points, sense=["min", "min", "min"])
        pareto_points = data_points[pareto_mask]
        true_pareto_area_list = pareto_points["area"].values.tolist()
        true_pareto_delay_list = pareto_points["delay"].values.tolist()
        true_pareto_power_list = pareto_points["power"].values.tolist()

        combine_array = []
        for i in range(len(true_pareto_area_list)):
            point = [true_pareto_area_list[i], true_pareto_delay_list[i], true_pareto_power_list[i]]
            combine_array.append(point)
        hv = hypervolume(combine_array)
        hv_value = hv.compute(self.reference_point)
        # save hypervolume and log hypervolume
        save_data_dict["testing_hypervolume"] = hv_value
        logger.tb_logger.add_scalar('testing hypervolume', hv_value, global_step=episode_num)
        
        # save pareto points and log pareto points
        x = true_pareto_area_list
        y = true_pareto_delay_list
        z = true_pareto_power_list
        fig1 = plt.figure()
        f1 = plt.scatter(x, y, c='r')
        fig2 = plt.figure()
        f2 = plt.scatter(x, z, c='r')
        fig3 = plt.figure()
        f3 = plt.scatter(y, z, c='r')
        logger.tb_logger.add_figure('testing pareto points area-delay', fig1, global_step=episode_num)
        logger.tb_logger.add_figure('testing pareto points area-power', fig2, global_step=episode_num)
        logger.tb_logger.add_figure('testing pareto points delay-power', fig3, global_step=episode_num)

        # 3d scatter
        pareto_points = np.asarray([[x[i], y[i], z[i]] for i in range(len(x))])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], color='r', label='Pareto Points', alpha=0.6)
        hull = ConvexHull(pareto_points)
        pareto_points = hull.points
        tri = Delaunay(pareto_points)
        ax.plot_trisurf(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], triangles=tri.simplices, color='red', alpha=0.3)
        plt.savefig("./test.png")

        logger.tb_logger.add_figure('testing pareto points', fig, global_step=episode_num)

        save_data_dict["testing_pareto_points_area"] = true_pareto_area_list
        save_data_dict["testing_pareto_points_delay"] = true_pareto_delay_list
        save_data_dict["testing_pareto_points_power"] = true_pareto_power_list
        
        return save_data_dict
    
    def _combine(self):
        combine_array = []
        for i in range(len(self.pareto_pointset["area"])):
            point = [self.pareto_pointset["area"][i], self.pareto_pointset["delay"][i], self.pareto_pointset["power"][i]]
            combine_array.append(point)
        return np.array(combine_array)
    
    def log_stats(
        self, loss, reward, rewards_dict,
        next_state, action, info, policy_info, action_column=0
    ):
        logger.tb_logger.add_scalar("ppa/avg_power", np.mean(rewards_dict["power"]), global_step=self.total_steps)
        logger.tb_logger.add_scalar("ppa/avg_delay", np.mean(rewards_dict["delay"]), global_step=self.total_steps)
        logger.tb_logger.add_scalar("ppa/avg_area", np.mean(rewards_dict["area"]), global_step=self.total_steps)
        logger.tb_logger.add_scalar("ppa/avg_ppa", rewards_dict["avg_ppa"], global_step=self.total_steps)

        try:
            loss = loss.item()
            q_values = np.mean(info['q_values'])
            target_q_values = np.mean(info['target_q_values'])
            positive_rewards_number = info['positive_rewards_number']
            logger.tb_logger.add_histogram('q_values_hist', info['q_values'], global_step=self.total_steps)
            logger.tb_logger.add_histogram('target_q_values_hist', info["target_q_values"], global_step=self.total_steps)
        except:
            loss = loss
            q_values = 0.
            target_q_values = 0.
            positive_rewards_number = 0.
        
        logger.tb_logger.add_scalar('train loss', loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar('reward', reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar('avg ppa', rewards_dict['avg_ppa'], global_step=self.total_steps)
        logger.tb_logger.add_scalar('legal num column', rewards_dict['legal_num_column'], global_step=self.total_steps)
        if "legal_num_stage" in rewards_dict.keys():
            logger.tb_logger.add_scalar('legal num stage', rewards_dict['legal_num_stage'], global_step=self.total_steps)
        if "legal_num_column_pp3" in rewards_dict.keys():
            logger.tb_logger.add_scalar('legal_num_column_pp3', rewards_dict['legal_num_column_pp3'], global_step=self.total_steps)
            logger.tb_logger.add_scalar('legal_num_column_pp0', rewards_dict['legal_num_column_pp0'], global_step=self.total_steps)
            
        if len(action.shape) <= 2:
            logger.tb_logger.add_scalar('action_index', action, global_step=self.total_steps)
        if policy_info is not None:
            logger.tb_logger.add_scalar('stage_num', policy_info["stage_num"], global_step=self.total_steps)
            logger.tb_logger.add_scalar('eps_threshold', policy_info["eps_threshold"], global_step=self.total_steps)
        
        logger.tb_logger.add_scalar('action_column', action_column, global_step=self.total_steps)
        logger.tb_logger.add_scalar('positive_rewards_number', positive_rewards_number, global_step=self.total_steps)
        
        try:
            for i in range(len(self.found_best_info)):
                logger.tb_logger.add_scalar(f'best_info/best ppa {i}-th weight', self.found_best_info[i]["found_best_ppa"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'best_info/best area {i}-th weight', self.found_best_info[i]["found_best_area"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'best_info/best delay {i}-th weight', self.found_best_info[i]["found_best_delay"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f'best_info/best power {i}-th weight', self.found_best_info[i]["found_best_power"], global_step=self.total_steps)
        except:
            logger.tb_logger.add_scalar(f'best_info/best ppa', self.found_best_info["found_best_ppa"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best_info/best area', self.found_best_info["found_best_area"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best_info/best delay', self.found_best_info["found_best_delay"], global_step=self.total_steps)
            logger.tb_logger.add_scalar(f'best_info/best power', self.found_best_info["found_best_power"], global_step=self.total_steps)
            
        # log q values info
        logger.tb_logger.add_scalar('q_values', q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('target_q_values', target_q_values, global_step=self.total_steps)

        if "internal_power" in rewards_dict.keys():
            try:
                logger.tb_logger.add_scalar(f"detailed_power/avg internal_power", np.mean(rewards_dict["internal_power"]), global_step=self.total_steps)
                logger.tb_logger.add_scalar(f"detailed_power/avg switching_power", np.mean(rewards_dict["switching_power"]), global_step=self.total_steps)
                logger.tb_logger.add_scalar(f"detailed_power/avg leakage_power", np.mean(rewards_dict["leakage_power"]), global_step=self.total_steps)
            except:
                logger.tb_logger.add_scalar(f"detailed_power/internal_power", rewards_dict["internal_power"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f"detailed_power/leakage_power", rewards_dict["leakage_power"], global_step=self.total_steps)
                logger.tb_logger.add_scalar(f"detailed_power/switching_power", rewards_dict["switching_power"], global_step=self.total_steps)


        # log state
        # ct32, ct22 = self.build_state_dict(next_state)
        # logger.tb_logger.add_scalars(
        #     'compressor 32', ct32, global_step=self.total_steps)
        # logger.tb_logger.add_scalars(
        #     'compressor 22', ct22, global_step=self.total_steps)
        # logger.tb_logger.add_image(
        #     'state image', next_state, global_step=self.total_steps, dataformats='HW'
        # )

        # log wallace area wallace delay
        logger.tb_logger.add_scalar('wallace area', self.env.wallace_area, global_step=self.total_steps)
        logger.tb_logger.add_scalar('wallace delay', self.env.wallace_delay, global_step=self.total_steps)
        logger.tb_logger.add_scalar('wallace power', self.env.wallace_power, global_step=self.total_steps)

        logger.tb_logger.add_scalar('weight area', self.env.weight_area, global_step=self.total_steps)
        logger.tb_logger.add_scalar('weight delay', self.env.weight_delay, global_step=self.total_steps)
        logger.tb_logger.add_scalar('weight power', self.env.weight_power, global_step=self.total_steps)
        
        # log env initial state pool
        # if self.env.initial_state_pool_max_len > 0:
        #     avg_area, avg_delay = self.get_env_pool_log()
        #     logger.tb_logger.add_scalars(
        #         'env state pool area', avg_area, global_step=self.total_steps)
        #     logger.tb_logger.add_scalars(
        #         'env state pool delay', avg_delay, global_step=self.total_steps)

        # log mask stats
        self.log_mask_stats(policy_info)
        self.log_action_stats(action)

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the best ppa state into the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    avg_power = np.mean(rewards_dict['power'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "power": avg_power,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa",
                            "normalize_area": rewards_dict["normalize_area"],
                            "normalize_delay": rewards_dict["normalize_delay"],
                            "normalize_power": rewards_dict["normalize_power"],
                        }
                    )
            elif self.env.store_state_type == "leq":
                if self.found_best_info['found_best_ppa'] >= rewards_dict['avg_ppa']:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    avg_power = np.mean(rewards_dict['power'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "power": avg_power,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa"
                        }
                    )
            elif self.env.store_state_type == "ppa_with_diversity":
                # store best ppa state
                if self.found_best_info['found_best_ppa'] > rewards_dict['avg_ppa']:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict['area'])
                    avg_delay = np.mean(rewards_dict['delay'])
                    avg_power = np.mean(rewards_dict['power'])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "power": avg_power,
                            "ppa": rewards_dict['avg_ppa'],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa"
                        }
                    )
                # store better diversity state
                elif rewards_dict['avg_ppa'] < self.env.initial_state_pool[0]["ppa"]:
                    number_states = len(self.env.initial_state_pool)
                    worse_state_distances = np.zeros(number_states-1)
                    cur_state_distances = np.zeros(number_states)
                    for i in range(number_states):
                        cur_state_dis = np.linalg.norm(
                            state - self.env.initial_state_pool[i]["state"],
                            ord=2
                        )
                        cur_state_distances[i] = cur_state_dis
                        if i == 0:
                            continue
                        worse_state_dis = np.linalg.norm(
                            self.env.initial_state_pool[0]["state"] - self.env.initial_state_pool[i]["state"],
                            ord=2
                        )
                        worse_state_distances[i-1] = worse_state_dis
 
                    worse_state_min_distance = np.min(worse_state_distances)
                    cur_state_min_distance = np.min(cur_state_distances)
                    if cur_state_min_distance > worse_state_min_distance:
                        # push the diverse state into the pool
                        avg_area = np.mean(rewards_dict['area'])
                        avg_delay = np.mean(rewards_dict['delay'])
                        avg_power = np.mean(rewards_dict['power'])
                        self.env.initial_state_pool.append(
                            {
                                "state": copy.deepcopy(state),
                                "area": avg_area,
                                "delay": avg_delay,
                                "power": avg_power,
                                "ppa": rewards_dict['avg_ppa'],
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "diverse"
                            }
                        )
        if self.found_best_info["found_best_ppa"] > rewards_dict['avg_ppa']:
            self.found_best_info["found_best_ppa"] = rewards_dict['avg_ppa']
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict['area']) 
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict['delay'])
            self.found_best_info["found_best_power"] = np.mean(rewards_dict['power'])
    
    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        if self.store_type == "detail":
            save_data_dict["replay_buffer"] = self.replay_memory.memory
        # env initial state pool 
        if self.env.initial_state_pool_max_len > 0:
            save_data_dict["env_initial_state_pool"] = self.env.initial_state_pool
        # best state best design
        save_data_dict["found_best_info"] = self.found_best_info
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]
        save_data_dict["pareto_power_points"] = self.pareto_pointset["power"]
        
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        best_state = copy.deepcopy(self.found_best_info["found_best_state"])
        ppas_dict = self.env.get_ppa_full_delay_cons(best_state)
        save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(self.total_steps, save_data_dict)

        # save q policy model
        q_policy_state_dict = self.target_q_policy.state_dict()
        logger.save_itr_params(self.total_steps, q_policy_state_dict)