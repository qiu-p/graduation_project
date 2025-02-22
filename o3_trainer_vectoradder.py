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
import pandas as pd
from paretoset import paretoset
from pygmo import hypervolume

from o0_logger import logger
from o5_utils_vector_adder import Transition, MBRLTransition, MultiObjTransition
from o0_global_const import PartialProduct

from ipdb import set_trace

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
        end_exp_freq=25
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
        # self.bit_width = env.bit_width
        self.num = env.num
        self.width = env.width
        self.initial_partial_product = np.full(self.width,self.num)
        # self.int_bit_width = env.int_bit_width
        # self.initial_partial_product = PartialProduct[self.bit_width][:-1]

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
        state = np.reshape(state, (1,2,self.width))
        next_state = np.reshape(next_state, (1,2,self.width))
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
        state = np.reshape(state, (1,2,self.width))
        next_state = np.reshape(next_state, (1,2,self.width))
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
        for i in range(batch_size):
            # compute image state
            ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            # compute image state
            if action_batch is not None:
                # reshape 有问题************
                q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((self.width)*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((self.width)*4)
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
                # q_values = self.target_q_policy(state.unsqueeze(0))                
                # if is_average:
                #     q_values = (q_values + 1000).detach()
                #     num = torch.count_nonzero(q_values)
                #     state_action_values[i] = q_values.sum() / (num+1e-4)
                if self.is_double_q:
                    current_q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((self.width)*4)
                    index = torch.argmax(current_q_values)
                    state_action_values[i] = q_values.squeeze()[index].detach()
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
        logger.tb_logger.add_scalar('stage_num', policy_info["stage_num"], global_step=self.total_steps)
        logger.tb_logger.add_scalar('eps_threshold', policy_info["eps_threshold"], global_step=self.total_steps)
        logger.tb_logger.add_scalar('action_column', action_column, global_step=self.total_steps)
        logger.tb_logger.add_scalar('positive_rewards_number', positive_rewards_number, global_step=self.total_steps)
        
        logger.tb_logger.add_scalar('best ppa', self.found_best_info["found_best_ppa"], global_step=self.total_steps)
        logger.tb_logger.add_scalar('best area', self.found_best_info["found_best_area"], global_step=self.total_steps)
        logger.tb_logger.add_scalar('best delay', self.found_best_info["found_best_delay"], global_step=self.total_steps)

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
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            
            predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4)
            with torch.no_grad():
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4)
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
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            
            with torch.no_grad():
                predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4)
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4)
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
        state = np.reshape(state, (1,2,self.width))
        next_state = np.reshape(next_state, (1,2,self.width))
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
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            # compute image state
            if action_batch is not None:
                # reshape 有问题************
                q_area, q_delay, q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i])
                q_area = q_area.reshape((self.width)*4)
                q_delay = q_delay.reshape((self.width)*4)
                q_values = q_values.reshape((self.width)*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((self.width)*4)
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
                    cur_q_values = q_values.reshape((self.width)*4)
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

class ThreeDRNDDQNAlgorithm(RNDDQNAlgorithm):
    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        compressor_state=np.zeros((2,self.width))
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
        state = np.reshape(state, (1,2,self.MAX_STAGE_NUM,self.width))
        next_state = np.reshape(next_state, (1,2,self.MAX_STAGE_NUM,self.width))
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
        #         q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((self.width)*4*self.MAX_STAGE_NUM)           
        #         # q_values = self.q_policy(state.unsqueeze(0)).reshape((self.width)*4)
        #         state_action_values[i] = q_values[action_batch[i]]
        #     else:
        #         q_values = self.target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
                
        #         # q_values = self.target_q_policy(state.unsqueeze(0))                
        #         # if is_average:
        #         #     q_values = (q_values + 1000).detach()
        #         #     num = torch.count_nonzero(q_values)
        #         #     state_action_values[i] = q_values.sum() / (num+1e-4)
        #         if self.is_double_q:
        #             current_q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((self.width)*4)
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
                predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4*self.MAX_STAGE_NUM)
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4*self.MAX_STAGE_NUM)
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
            predict_value = self.rnd_predictor(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4*self.MAX_STAGE_NUM)
            with torch.no_grad():
                target_value = self.rnd_target(state.unsqueeze(0).float(), is_target=self.is_target, state_mask=state_mask[i]).reshape((self.width)*4*self.MAX_STAGE_NUM)
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
        # imagined_env,
        warm_start_steps=500,
        # imagine sample kwargs
        num_random_sample=64,
        num_sample=20,
        depth=5,
        imagine_data_freq=25,
        train_q_steps=5,
        # train model kwargs
        train_model_freq=100,
        train_model_start_steps=200,
        train_model_finetune_steps=20,
        train_model_batch_size=256,
        model_lr=1e-3,
        # evaluate imagine model kwargs
        evaluate_imagine_state_num=5,
        evaluate_imagine_state_freq=10,
        # train policy kwargs
        real_data_ratio=0.2,
        num_train_per_step=4,
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
        # self.imagined_env = imagined_env

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

    def store(
        self, state, next_state, 
        action, reward, mask, next_state_mask,
        normalize_area, normalize_delay,
        is_model_evaluation=False
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1,2,self.width))
        next_state = np.reshape(next_state, (1,2,self.width))
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
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            states.append(state.unsqueeze(0))
        states = torch.cat(states)
        # compute image state
        if action_batch is not None:
            q_values = self.q_policy(states.float(), state_mask=state_mask)
            q_values = q_values.reshape(-1, (self.width)*4)           
            # q_values = self.q_policy(state.unsqueeze(0)).reshape((self.width)*4)
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
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            states.append(state.unsqueeze(0))
        states = torch.cat(states)
        with torch.no_grad():
            predict_value = self.rnd_predictor(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (self.width)*4)
            target_value = self.rnd_target(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (self.width)*4)
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
                if step <= self.real_data_ratio * self.len_per_episode:
                    is_model_evaluation = False
                else:
                    is_model_evaluation = True
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
        # self._imagine_model_data()
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
        for train_step in range(train_steps):
            # sample a batch
            transitions = self.replay_memory.sample(self.train_model_batch_size)
            batch = MBRLTransition(*zip(*transitions))
            next_state_batch = torch.cat(batch.next_state)
            normalize_area_batch = torch.cat(batch.normalize_area).float().to(self.device)
            normalize_delay_batch = torch.cat(batch.normalize_delay).float().to(self.device)
            
            area_loss = torch.zeros(self.train_model_batch_size, device=self.device)
            delay_loss = torch.zeros(self.train_model_batch_size, device=self.device)
            states = []
            for j in range(self.train_model_batch_size):
                # !!! 耗时，重复调用，可工程优化
                ct32, ct22, pp, stage_num = self.env.decompose_compressor_tree(self.initial_partial_product, next_state_batch[j].cpu().numpy())
                ct32 = torch.tensor(np.array([ct32]))
                ct22 = torch.tensor(np.array([ct22]))
                if stage_num < self.MAX_STAGE_NUM-1:
                    zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
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
    
    def _imagine_data(self):
        # 从真实环境出发
        pass

    def _run_mb_episode(self):
        mb_info = {}
        # 1. train model
        if self.total_steps % self.train_model_freq == 0:
            model_loss, model_train_info = self._train_model()        
            mb_info['model_loss'] = model_loss
            mb_info['model_train_info'] = model_train_info
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

        if episode_num % self.evaluate_imagine_state_freq == 0:
            full_ppas_dict = self.evaluate(episode_num)
            for k in ppas_dict.keys():
                ppas_dict[k].extend(full_ppas_dict[k])

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
                q_values = self.q_policy(seq_state.float(), state_mask=state_mask[i]).reshape((self.width)*4)           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((self.width)*4)
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
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, self.width)
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            if action_batch is not None:
                q_values = self.q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((self.width))           
                # q_values = self.q_policy(state.unsqueeze(0)).reshape((self.width)*4)
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
