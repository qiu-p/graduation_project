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
import logging
import json
import random

from o0_logger import logger
from o0_rtl_tasks import EvaluateWorker
from o0_state import State, SimpleState
from o1_environment_refine import RefineEnv, RefineEnvMultiAgent
from o2_policy_refine import DeepQPolicy, MaskDeepQPolicy
from o5_utils_refine import (
    Transition,
    ReplayMemory,
    MaskTransition,
    MaskReplayMemory,
    RunningMeanStd,
    MARLTransition,
    MARLReplayMemory,
    PowerMaskTransition,
    PowerMaskReplayMemory
)

from ipdb import set_trace

from ysh_logger import get_logger

class DQNAlgorithm:
    def __init__(
        self,
        env: RefineEnv,
        q_policy: DeepQPolicy,
        target_q_policy: DeepQPolicy,
        replay_memory: ReplayMemory,
        is_target=False,
        deterministic=False,
        is_softmax=False,
        optimizer_class="RMSprop",
        q_net_lr=1e-2,
        batch_size=64,
        gamma=0.8,
        target_update_freq=10,
        len_per_episode=25,
        total_episodes=400,
        # is double q
        is_double_q=False,
        # agent reset
        agent_reset_freq=0,
        agent_reset_type="xavier",
        device="cpu",
        # pareto
        reference_point=[2600, 1.8],
        # multiobj
        multiobj_type="pure_max",  # [pure_max, weight_max]
        # store type
        store_type="simple",  # simple or detail
        # end_exp_log
        end_exp_freq=25,
        log_level=0,
        is_adder_only=False, # 只优化 adder
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
        self.device = device
        self.is_target = is_target
        self.is_double_q = is_double_q
        self.store_type = store_type
        self.multiobj_type = multiobj_type
        self.end_exp_freq = end_exp_freq

        self.is_adder_only = is_adder_only

        self.log_level = log_level
        # optimizer
        # TODO: lr, lrdecay, gamma
        # TODO: double q
        self.q_net_lr = q_net_lr

        if isinstance(optimizer_class, str):
                optimizer_class = eval("optim." + optimizer_class)
                self.optimizer_class = optimizer_class
        if not self.is_adder_only:
            self.policy_optimizer = optimizer_class(
                self.q_policy.parameters(), lr=self.q_net_lr
            )

        # loss function
        self.loss_fn = nn.SmoothL1Loss()

        # total steps
        self.total_steps = 0

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_evaluate_worker": None,
        }
        # agent reset
        self.agent_reset_freq = agent_reset_freq
        self.agent_reset_type = agent_reset_type

        self.deterministic = deterministic
        self.is_softmax = is_softmax

        # pareto pointset
        self.pareto_pointset = {}
        for target_keys in self.env.opt_target_label:
            self.pareto_pointset[target_keys] = []

        self.reference_point = reference_point
        # configure figure
        plt.switch_backend("agg")

    def store(  # visual
        self, state: State, next_state: State, action, reward, mask, next_state_mask
    ):
        self.replay_memory.push(
            state,
            action,
            next_state,
            torch.tensor([reward]),
            mask.reshape(1, -1),
            next_state_mask.reshape(1, -1),
        )

    def store_detail(
        self,
        state: State,
        next_state: State,
        action,
        reward,
        mask,
        next_state_mask,
        state_ct32,
        state_ct22,
        next_state_ct32,
        next_state_ct22,
        rewards_dict,
    ):
        self.replay_memory.push(
            state,
            action,
            next_state,
            torch.tensor([reward]),
            mask.reshape(1, -1),
            next_state_mask.reshape(1, -1),
            state_ct32,
            state_ct22,
            next_state_ct32,
            next_state_ct22,
            rewards_dict,
        )

    # fmt: off
    def compute_values(self, state_batch, action_batch, state_mask, is_average=False):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            # compute image state
            state: State = state_batch[i]
            ct32, ct22 = state.archive()
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            image_state = torch.cat((ct32, ct22), dim=0)
            mask = state_mask[i]
            if action_batch is not None:
                q_values = self.q_policy(image_state.unsqueeze(0).float(), state_mask=mask).reshape(4 * state.get_pp_len())
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(image_state.unsqueeze(0).float(), state_mask=mask, is_target=True)
                if self.is_double_q:
                    raise NotImplementedError
                else:
                    state_action_values[i] = q_values.max(1)[0].detach()
        return state_action_values
    # fmt: on

    # fmt: off
    def update_q(self): # visual
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            next_state_batch = batch.next_state
            state_batch = batch.state
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_mask = torch.cat(batch.mask)
            next_state_mask = torch.cat(batch.next_state_mask)
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(state_batch, action_batch, state_mask)
            next_state_values = self.compute_values(next_state_batch, None, next_state_mask)
            target_state_action_values = next_state_values * self.gamma + reward_batch.to(self.device)

            loss = self.loss_fn(state_action_values.unsqueeze(1), target_state_action_values.unsqueeze(1))

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
    # fmt: on

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
        if not self.is_adder_only:
            q_policy_state_dict = self.target_q_policy.state_dict()
            logger.save_itr_params(self.total_steps, q_policy_state_dict)

    def run_experiments(self):
        for episode_num in range(self.total_episodes):
            self.run_episode(episode_num)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num)
        self.end_experiments(episode_num)

    def get_mask_stats(self, mask):
        number_column = int(len(mask) / 4)
        valid_number_each_column = np.zeros(number_column)
        for i in range(number_column):
            cur_column_mask = mask[4 * i : 4 * (i + 1)]
            valid_number_each_column[i] = torch.sum(cur_column_mask)
        counter = Counter(valid_number_each_column)
        return counter

    def log_mask_stats(self, policy_info):
        if policy_info is not None:
            # log mask average
            logger.tb_logger.add_scalar(
                "mask avg valid number",
                torch.sum(policy_info["mask"]),
                global_step=self.total_steps,
            )
            logger.tb_logger.add_scalar(
                "simple mask avg valid number",
                torch.sum(policy_info["simple_mask"]),
                global_step=self.total_steps,
            )

            mask_counter = self.get_mask_stats(policy_info["mask"])
            simple_mask_counter = self.get_mask_stats(policy_info["simple_mask"])
            for k in mask_counter.keys():
                logger.tb_logger.add_scalar(
                    f"mask number {k}", mask_counter[k], global_step=self.total_steps
                )
            for k in simple_mask_counter.keys():
                logger.tb_logger.add_scalar(
                    f"simple mask number {k}",
                    simple_mask_counter[k],
                    global_step=self.total_steps,
                )

            if "num_valid_action" in policy_info.keys():
                logger.tb_logger.add_scalar(
                    "number valid action",
                    policy_info["num_valid_action"],
                    global_step=self.total_steps,
                )

            if "state_ct32" in policy_info.keys():
                if len(policy_info["state_ct32"].shape) == 2:
                    logger.tb_logger.add_image(
                        "state ct32",
                        np.array(policy_info["state_ct32"]),
                        global_step=self.total_steps,
                        dataformats="HW",
                    )
                    logger.tb_logger.add_image(
                        "state ct22",
                        np.array(policy_info["state_ct22"]),
                        global_step=self.total_steps,
                        dataformats="HW",
                    )
                    logger.tb_logger.add_image(
                        "state ct32 sum ct22",
                        np.array(policy_info["state_ct32"])
                        + np.array(policy_info["state_ct22"]),
                        global_step=self.total_steps,
                        dataformats="HW",
                    )
                elif len(policy_info["state_ct32"].shape) == 3:
                    logger.tb_logger.add_image(
                        "state ct32",
                        np.array(policy_info["state_ct32"]),
                        global_step=self.total_steps,
                        dataformats="CHW",
                    )
                    logger.tb_logger.add_image(
                        "state ct22",
                        np.array(policy_info["state_ct22"]),
                        global_step=self.total_steps,
                        dataformats="CHW",
                    )
                    logger.tb_logger.add_image(
                        "state ct32 sum ct22",
                        np.concatenate(
                            (
                                np.array(policy_info["state_ct32"]),
                                np.array(policy_info["state_ct22"]),
                            ),
                            axis=0,
                        ),
                        global_step=self.total_steps,
                        dataformats="CHW",
                    )

    def log_action_stats(self, action):
        pass

    def log_stats(
        self,
        loss,
        reward,
        rewards_dict,
        next_state,
        action,
        info,
        policy_info,
        action_column=0,
    ):
        try:
            loss = loss.item()
            q_values = np.mean(info["q_values"])
            target_q_values = np.mean(info["target_q_values"])
            positive_rewards_number = info["positive_rewards_number"]
        except:
            loss = loss
            q_values = 0.0
            target_q_values = 0.0
            positive_rewards_number = 0.0

        logger.tb_logger.add_scalar("train loss", loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar("reward", reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar(
            "avg ppa", rewards_dict["avg_ppa"], global_step=self.total_steps
        )
        logger.tb_logger.add_scalar(
            "legal num column",
            rewards_dict["legal_num_column"],
            global_step=self.total_steps,
        )
        if "legal_num_stage" in rewards_dict.keys():
            logger.tb_logger.add_scalar(
                "legal num stage",
                rewards_dict["legal_num_stage"],
                global_step=self.total_steps,
            )
        if "legal_num_column_pp3" in rewards_dict.keys():
            logger.tb_logger.add_scalar(
                "legal_num_column_pp3",
                rewards_dict["legal_num_column_pp3"],
                global_step=self.total_steps,
            )
            logger.tb_logger.add_scalar(
                "legal_num_column_pp0",
                rewards_dict["legal_num_column_pp0"],
                global_step=self.total_steps,
            )

        if len(action.shape) <= 2:
            logger.tb_logger.add_scalar(
                "action_index", action, global_step=self.total_steps
            )
        if policy_info is not None:
            logger.tb_logger.add_scalar(
                "stage_num", policy_info["stage_num"], global_step=self.total_steps
            )
            logger.tb_logger.add_scalar(
                "eps_threshold",
                policy_info["eps_threshold"],
                global_step=self.total_steps,
            )

        logger.tb_logger.add_scalar(
            "action_column", action_column, global_step=self.total_steps
        )
        logger.tb_logger.add_scalar(
            "positive_rewards_number",
            positive_rewards_number,
            global_step=self.total_steps,
        )

        try:
            for i in range(len(self.found_best_info)):
                logger.tb_logger.add_scalar(
                    f"best ppa {i}-th weight",
                    self.found_best_info[i]["found_best_ppa"],
                    global_step=self.total_steps,
                )
                logger.tb_logger.add_scalar(
                    f"best area {i}-th weight",
                    self.found_best_info[i]["found_best_area"],
                    global_step=self.total_steps,
                )
                logger.tb_logger.add_scalar(
                    f"best delay {i}-th weight",
                    self.found_best_info[i]["found_best_delay"],
                    global_step=self.total_steps,
                )
        except:
            logger.tb_logger.add_scalar(
                f"best ppa",
                self.found_best_info["found_best_ppa"],
                global_step=self.total_steps,
            )
            logger.tb_logger.add_scalar(
                f"best area",
                self.found_best_info["found_best_area"],
                global_step=self.total_steps,
            )
            logger.tb_logger.add_scalar(
                f"best delay",
                self.found_best_info["found_best_delay"],
                global_step=self.total_steps,
            )

        # log q values info
        logger.tb_logger.add_scalar("q_values", q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar(
            "target_q_values", target_q_values, global_step=self.total_steps
        )

        # log wallace area wallace delay
        logger.tb_logger.add_scalar(
            "wallace area", self.env.wallace_area, global_step=self.total_steps
        )
        logger.tb_logger.add_scalar(
            "wallace delay", self.env.wallace_delay, global_step=self.total_steps
        )

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

    # fmt: off
    def update_env_initial_state_pool(self, state:State, evaluate_worker:EvaluateWorker):
        ppa = self.env.get_ppa(evaluate_worker.consult_ppa())
        if self.env.initial_state_pool_max_len > 0:
            if self.env.store_state_type == "less":
                if self.found_best_info['found_best_ppa'] > ppa:
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "evaluate_worker": copy.deepcopy(evaluate_worker),
                            "count": 1,
                        }
                    )
            else:
                raise NotImplementedError

        if self.found_best_info["found_best_ppa"] > ppa:
            self.found_best_info["found_best_ppa"] = ppa
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_evaluate_worker"] = copy.deepcopy(evaluate_worker)
    # fmt: on

    def reset_agent(self):  # visual
        if self.total_steps % self.agent_reset_freq == 0:
            self.q_policy.partially_reset(reset_type=self.agent_reset_type)
            self.target_q_policy.partially_reset(reset_type=self.agent_reset_type)

    # fmt: off
    def process_and_log_pareto(self, episode_num, episode_opt_target_value:dict):
        # 1. compute pareto pointset
        for ppa_key in episode_opt_target_value.keys():
            episode_opt_target_value[ppa_key].extend(self.pareto_pointset[ppa_key])
        data_points = pd.DataFrame(episode_opt_target_value)
        pareto_mask = paretoset(data_points, sense=["min"] * len(episode_opt_target_value.keys()))
        pareto_points = data_points[pareto_mask]
        new_pareto_list = {}

        for ppa_key in episode_opt_target_value.keys():
            new_pareto_list[ppa_key] = pareto_points[ppa_key].values.tolist()
            self.pareto_pointset[ppa_key] = new_pareto_list[ppa_key]

        pareto_point_array = np.zeros([len(self.pareto_pointset[ppa_key]), len(episode_opt_target_value.keys())])
        for ppa_index, ppa_key in enumerate(episode_opt_target_value.keys()):
            for point_index in range(len(new_pareto_list[ppa_key])):
                pareto_point_array[point_index][ppa_index] = new_pareto_list[ppa_key][point_index]

        hv = hypervolume(pareto_point_array)
        hv_value = hv.compute(self.reference_point)
        logger.tb_logger.add_scalar(
            'hypervolume', hv_value, global_step=episode_num)
        logger.log(f"episode {episode_num}, hypervolume: {hv_value}")

        if len(self.env.opt_target_label) == 3 and len(pareto_point_array) > 4:
            fig1 = plt.figure()
            x, y, z = pareto_point_array.T
            fig1 = plt.figure()
            f1 = plt.scatter(x, y, c='r')
            fig2 = plt.figure()
            f2 = plt.scatter(x, z, c='r')
            fig3 = plt.figure()
            f3 = plt.scatter(y, z, c='r')
            logger.tb_logger.add_figure(f'pareto points area-delay', fig1, global_step=episode_num)
            logger.tb_logger.add_figure(f'pareto points area-power', fig2, global_step=episode_num)
            logger.tb_logger.add_figure(f'pareto points delay-power', fig3, global_step=episode_num)

            # 3d scatter
            pareto_points = np.asarray([[x[i], y[i], z[i]] for i in range(len(x))])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], color='r', label='Pareto Points', alpha=0.6)
            hull = ConvexHull(pareto_points)
            pareto_points = hull.points
            tri = Delaunay(pareto_points)
            ax.plot_trisurf(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], triangles=tri.simplices, color='red', alpha=0.3)

            logger.tb_logger.add_figure('pareto points 3d', fig, global_step=episode_num)
    # fmt: on

    # fmt: off
    def log_and_save_pareto_points(self, ppas_dict, episode_num):
        save_data_dict = {}
        # save ppa_csv
        save_data_dict["testing_full_ppa"] = ppas_dict
        # compute pareto points
        data_points = pd.DataFrame(ppas_dict)
        pareto_mask = paretoset(data_points, sense=["min"] * len(self.env.opt_target_label))
        pareto_points = data_points[pareto_mask]
        true_pareto_list = {}
        for ppa_key in self.env.opt_target_label:
            true_pareto_list[ppa_key] = pareto_points[ppa_key].values.tolist()

        combine_array = []
        for i in range(len(true_pareto_list[ppa_key])):
            point = []
            for ppa_key in self.env.opt_target_label:
                point.append(true_pareto_list[ppa_key][i])
            combine_array.append(point)
        hv = hypervolume(combine_array)
        hv_value = hv.compute(self.reference_point)
        # save hypervolume and log hypervolume
        save_data_dict["testing_hypervolume"] = hv_value
        logger.tb_logger.add_scalar(
            'testing hypervolume', hv_value, global_step=episode_num)

        # save pareto points and log pareto points
        if len(self.env.opt_target_label) == 3 and len(true_pareto_list) > 4:
            fig1 = plt.figure()
            x, y, z = np.asarray(combine_array).T
            fig1 = plt.figure()
            f1 = plt.scatter(x, y, c='r')
            fig2 = plt.figure()
            f2 = plt.scatter(x, z, c='r')
            fig3 = plt.figure()
            f3 = plt.scatter(y, z, c='r')
            logger.tb_logger.add_figure(f'testing pareto points area-delay', fig1, global_step=episode_num)
            logger.tb_logger.add_figure(f'testing pareto points area-power', fig2, global_step=episode_num)
            logger.tb_logger.add_figure(f'testing pareto points delay-power', fig3, global_step=episode_num)

            # 3d scatter
            pareto_points = np.asarray([[x[i], y[i], z[i]] for i in range(len(x))])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], color='r', label='Pareto Points', alpha=0.6)
            hull = ConvexHull(pareto_points)
            pareto_points = hull.points
            tri = Delaunay(pareto_points)
            ax.plot_trisurf(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], triangles=tri.simplices, color='red', alpha=0.3)

            logger.tb_logger.add_figure('testing pareto points 3d', fig, global_step=episode_num)
        save_data_dict["testing_pareto_points"] = true_pareto_list
        return save_data_dict
    # fmt: on

    # fmt: off
    def run_episode(self, episode_num): # visual
        raise NotImplementedError
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
                self.store(state, next_state, action, reward,
                           policy_info['mask'], next_state_policy_info['mask'])
            elif self.store_type == "detail":
                self.store_detail(state, next_state, action, reward, policy_info['mask'], next_state_policy_info['mask'], policy_info['state_ct32'],
                                  policy_info['state_ct22'], next_state_policy_info['state_ct32'], next_state_policy_info['state_ct22'], rewards_dict)

            # update initial state pool
            self.update_env_initial_state_pool(
                next_state, rewards_dict, next_state_policy_info['mask'])
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
    # fmt: on


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
        bonus_type="rnd",  # rnd/noveld
        noveld_alpha=0.1,
        # n step q-learning
        n_step_num=5,
        **dqn_alg_kwargs,
    ):
        super().__init__(
            env, q_policy, target_q_policy, replay_memory, **dqn_alg_kwargs
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
        if not self.is_adder_only:
        # optimizer
            self.rnd_model_optimizer = self.optimizer_class(
                self.rnd_predictor.parameters(), lr=self.rnd_lr
            )
        # loss func
        self.rnd_loss = nn.MSELoss()
        # log
        self.rnd_loss_item = 0.0
        self.rnd_int_rewards = 0.0
        self.rnd_ext_rewards = 0.0
        # cosine similarity
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def update_reward_int_run_mean_std(self, rewards):
        mean, std, count = np.mean(rewards), np.std(rewards), len(rewards)
        self.int_reward_run_mean_std.update_from_moments(mean, std**2, count)

    # fmt: off
    def update_rnd_model(
        self, state_batch, state_mask
    ):
        loss = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            state: State = state_batch[i]
            ct32, ct22 = state.archive()
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            image_state = torch.cat((ct32, ct22), dim=0)
            mask = state_mask[i]

            predict_value = self.rnd_predictor(image_state.unsqueeze(0).float(), is_target=self.is_target, state_mask=mask).reshape(state.get_pp_len() * 4)
            with torch.no_grad():
                target_value = self.rnd_target(image_state.unsqueeze(0).float(), is_target=self.is_target, state_mask=mask).reshape(state.get_pp_len() * 4)
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
    # fmt: on

    def compute_int_rewards(self, state_batch, state_mask):
        batch_size = len(state_batch)
        int_rewards = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            state: State = state_batch[i]
            ct32, ct22 = state.archive()
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            image_state = torch.cat((ct32, ct22), dim=0)
            mask = state_mask[i]

            with torch.no_grad():
                predict_value = self.rnd_predictor(
                    image_state.unsqueeze(0).float(),
                    is_target=self.is_target,
                    state_mask=mask,
                ).reshape(state.get_pp_len() * 4)
                target_value = self.rnd_target(
                    image_state.unsqueeze(0).float(),
                    is_target=self.is_target,
                    state_mask=mask,
                ).reshape(state.get_pp_len() * 4)
            # set_trace()
            int_rewards[i] = torch.sum((predict_value - target_value) ** 2)
        return int_rewards

    # fmt: off
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            next_state_batch = batch.next_state
            state_batch = batch.state
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
                int_rewards_batch = int_rewards_batch - \
                    self.noveld_alpha * int_rewards_last_state_batch

            int_rewards_batch = int_rewards_batch / \
                torch.tensor(
                    np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            train_reward_batch = reward_batch.to(
                self.device) + self.int_reward_scale * int_rewards_batch
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(
                state_batch, action_batch, state_mask
            )
            next_state_values = self.compute_values(
                next_state_batch, None, next_state_mask
            )
            target_state_action_values = (
                next_state_values * self.gamma) + train_reward_batch

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
    # fmt: on

    def get_ppa_value(self):
        assert self.env.initial_state_pool_max_len > 0
        number_states = len(self.env.initial_state_pool)
        state_value = np.zeros(number_states)
        info_count = np.zeros(number_states)
        for i in range(number_states):
            avg_ppa = self.env.get_ppa(
                self.env.initial_state_pool[i]["evaluate_worker"]
            )
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
            states_batch.append(torch.tensor(self.env.initial_state_pool[i]["state"]))
            states_mask.append(self.env.initial_state_pool[i]["state_mask"])
        if value_type == "novelty":
            state_novelty = self.compute_int_rewards(states_batch, states_mask)
            return state_novelty.cpu().numpy()
        elif value_type == "value":
            state_value = self.compute_values(states_batch, None, states_mask)
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
                text = plt.text(
                    j,
                    i,
                    state_mutual_distances[i, j],
                    ha="center",
                    va="center",
                    color="w",
                )
        logger.tb_logger.add_figure(
            "state mutual distance", fig1, global_step=self.total_steps
        )

    def log_env_pool(self):
        total_state_num = len(self.env.initial_state_pool)
        logger.tb_logger.add_scalar(
            "env pool total num", total_state_num, global_step=self.total_steps
        )

    # fmt: off
    def log_stats(self, loss, action, step_info_dict, q_info):
        # 方便画图的小函数
        def __tb_add_fig(data, label):
            fig = plt.figure()
            plt.cla()
            plt.imshow(data)
            plt.colorbar()
            plt.tight_layout()
            logger.tb_logger.add_figure(label, fig, self.total_steps)

        # q_info and loss
        try:
            loss = loss.item()
            q_values = np.mean(q_info['q_values'])
            target_q_values = np.mean(q_info['target_q_values'])
            positive_rewards_number = q_info['positive_rewards_number']
            logger.tb_logger.add_histogram('q_values_hist', q_info['q_values'], global_step=self.total_steps)
            logger.tb_logger.add_histogram('target_q_values_hist', q_info["target_q_values"], global_step=self.total_steps)
        except:
            loss = loss
            q_values = 0.
            target_q_values = 0.
            positive_rewards_number = 0.
        logger.tb_logger.add_scalar('q_info/train loss', loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/q_values', q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/target_q_values', target_q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/positive_rewards_number', positive_rewards_number, global_step=self.total_steps)

        # step info
        next_state: State = step_info_dict["next_state"]
        reward = step_info_dict["reward"]
        next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]

        logger.tb_logger.add_scalar('action/value', action, global_step=self.total_steps)
        logger.tb_logger.add_scalar('action/column', action // 4, global_step=self.total_steps)
        logger.tb_logger.add_scalar('action/type', action % 4, global_step=self.total_steps)
        logger.tb_logger.add_scalar('reward', reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar('stage num', next_state.get_stage_num(), global_step=self.total_steps)

        ct_decomposed = next_state.archive()
        __tb_add_fig(ct_decomposed[0], "state/ct32")
        __tb_add_fig(ct_decomposed[1], "state/ct22")

        # simulation info
        ppa_dict = next_evaluate_worker.consult_ppa()
        for ppa_key in ppa_dict:
            logger.tb_logger.add_scalar(f'ppa/{ppa_key}', ppa_dict[ppa_key], global_step=self.total_steps)
        logger.tb_logger.add_scalar(f'ppa/ppa', self.env.get_ppa(ppa_dict), global_step=self.total_steps)
        power_mask_32, power_mask_22 = next_evaluate_worker.consult_compressor_power_mask(next_state.archive())
        power_mask_total = power_mask_32 + power_mask_22
        __tb_add_fig(power_mask_32, "power mask/ct32")
        __tb_add_fig(power_mask_22, "power mask/ct22")
        __tb_add_fig(power_mask_total, "power mask/total")

        # pp routing info
        if self.env.use_routing_optimize:
            evaluate_worker_no_routing: EvaluateWorker = step_info_dict["evaluate_worker_no_routing"]
            power_mask_32_no_routing, power_mask_22_no_routing = evaluate_worker_no_routing.consult_compressor_power_mask(next_state.archive())
            power_mask_total_no_routing = power_mask_32_no_routing + power_mask_22_no_routing
            __tb_add_fig(power_mask_total_no_routing - power_mask_total, "routing_info/power incr mask")
            ppa_dict_no_routing = evaluate_worker_no_routing.consult_ppa()
            for ppa_key in ppa_dict_no_routing.keys():
                logger.tb_logger.add_scalar(f'routing_info/{ppa_key} incr', ppa_dict_no_routing[ppa_key] - ppa_dict[ppa_key], global_step=self.total_steps)

        # found best info
        logger.tb_logger.add_scalar(f'found_best_info/found_best_ppa', self.found_best_info["found_best_ppa"], global_step=self.total_steps)
        found_best_evaluate_worker: EvaluateWorker = self.found_best_info["found_best_evaluate_worker"]
        found_best_ppa_dict = found_best_evaluate_worker.consult_ppa()
        for key in found_best_ppa_dict.keys():
            logger.tb_logger.add_scalar(f'found_best_info/found_best_{key}', found_best_ppa_dict[key], global_step=self.total_steps)

        # optimization ratio
        initial_evaluate_worker: EvaluateWorker = self.env.initial_evaluate_worker
        initial_ppa_dict = initial_evaluate_worker.consult_ppa()
        for key in initial_ppa_dict.keys():
            initial_value = initial_ppa_dict[key]
            cur_value = ppa_dict[key]
            cur_incr = 100 * (initial_value - cur_value) / initial_value
            logger.tb_logger.add_scalar(f'optimization ratio (%) current/{key}', cur_incr, global_step=self.total_steps)

            best_value = found_best_ppa_dict[key]
            best_incr = 100 * (initial_value - best_value) / initial_value
            logger.tb_logger.add_scalar(f'optimization ratio (%) found best/{key}', best_incr, global_step=self.total_steps)

        if self.env.use_routing_optimize:
            initial_evaluate_worker_no_routing = self.env.initial_evaluate_worker_no_routing
            initial_ppa_dict_no_routing = initial_evaluate_worker_no_routing.consult_ppa()
            for key in initial_ppa_dict.keys():
                initial_value = initial_ppa_dict_no_routing[key]
                cur_value = ppa_dict[key]
                cur_incr = 100 * (initial_value - cur_value) / initial_value
                logger.tb_logger.add_scalar(f'optimization ratio (% no routing) current/{key}', cur_incr, global_step=self.total_steps)

                best_value = found_best_ppa_dict[key]
                best_incr = 100 * (initial_value - best_value) / initial_value
                logger.tb_logger.add_scalar(f'optimization ratio (% no routing) found best/{key}', best_incr, global_step=self.total_steps)
    # fmt: on

    # fmt: off
    def run_episode(self, episode_num):
        # reset state
        episode_opt_target_value = {}
        for ppa_key in self.env.opt_target_label:
            episode_opt_target_value[ppa_key] = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
            else:
                raise NotImplementedError

        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar(
                'env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)

            get_logger(logger_name='ysh').info('total_steps: {}'.format(self.total_steps))
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            step_info_dict = self.env.step(action)
            next_state: State = step_info_dict["next_state"]
            reward = step_info_dict["reward"]
            next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]
            next_state_mask = next_state.mask_with_legality()

            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], torch.tensor(next_state_mask))
            elif self.store_type == "detail":
                raise NotImplementedError

            # update initial state pool
            self.update_env_initial_state_pool(next_state, next_evaluate_worker)
            # update q policy
            loss, q_info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            # reset agent
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(loss, action, step_info_dict, q_info)
            self.log_rnd_stats(q_info)

            next_ppa_dict = next_evaluate_worker.consult_ppa()
            for ppa_key in self.env.opt_target_label:
                episode_opt_target_value[ppa_key].append(next_ppa_dict[ppa_key])
            avg_ppa = self.env.get_ppa(next_ppa_dict)
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

            state = copy.deepcopy(next_state)
        # update target q
        self.target_q_policy.load_state_dict(
            self.q_policy.state_dict()
        )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_opt_target_value)
    # fmt: on

    def log_rnd_stats(self, info):
        # int reward vs rewards
        logger.tb_logger.add_scalar(
            "batch int rewards", self.rnd_int_rewards, global_step=self.total_steps
        )
        logger.tb_logger.add_scalar(
            "batch ext rewards", self.rnd_ext_rewards, global_step=self.total_steps
        )
        logger.tb_logger.add_scalar(
            "rnd loss", self.rnd_loss_item, global_step=self.total_steps
        )


class MaskDQNAlgorithm(DQNAlgorithm):
    def __init__(
        self,
        env: RefineEnv,
        q_policy: MaskDeepQPolicy,
        target_q_policy: MaskDeepQPolicy,
        replay_memory: MaskReplayMemory,
        **dqn_alg_kwargs,
    ):
        super().__init__(
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            **dqn_alg_kwargs,
        )
        self.q_policy = q_policy
        self.target_q_policy = target_q_policy

    # fmt: off
    def log_stats(self, loss, action, step_info_dict, q_info):
        # 方便画图的小函数
        def __tb_add_fig(data, label):
            fig = plt.figure()
            plt.cla()
            plt.imshow(data)
            plt.colorbar()
            plt.tight_layout()
            logger.tb_logger.add_figure(label, fig, self.total_steps)
        def __tb_add_bar(data, label):
            fig = plt.figure()
            plt.cla()
            x = np.asarray(data)
            plt.bar(range(len(x)), x, alpha=0.5)
            plt.plot(range(len(x)), x, "--o")
            plt.tight_layout()
            logger.tb_logger.add_figure(label, fig, self.total_steps)
        # q_info and loss
        try:
            loss = loss.item()
            q_values = np.mean(q_info['q_values'])
            target_q_values = np.mean(q_info['target_q_values'])
            positive_rewards_number = q_info['positive_rewards_number']
            if self.log_level > 1:
                logger.tb_logger.add_histogram('q_values_hist', q_info['q_values'], global_step=self.total_steps)
                logger.tb_logger.add_histogram('target_q_values_hist', q_info["target_q_values"], global_step=self.total_steps)
        except:
            loss = loss
            q_values = 0.
            target_q_values = 0.
            positive_rewards_number = 0.
        logger.tb_logger.add_scalar('q_info/train loss', loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/q_values', q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/target_q_values', target_q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/positive_rewards_number', positive_rewards_number, global_step=self.total_steps)

        # step info
        next_state: State = step_info_dict["next_state"]
        reward = step_info_dict["reward"]
        next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]

        logger.tb_logger.add_scalar('action/value', action, global_step=self.total_steps)
        logger.tb_logger.add_scalar('action/column', action // self.q_policy.action_type_num, global_step=self.total_steps)
        logger.tb_logger.add_scalar('action/type', action % self.q_policy.action_type_num, global_step=self.total_steps)
        logger.tb_logger.add_scalar('reward', reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar('stage num', next_state.get_stage_num(), global_step=self.total_steps)

        if self.log_level > 1:
            ct_decomposed = next_state.archive()
            __tb_add_fig(ct_decomposed[0], "state/ct32")
            __tb_add_fig(ct_decomposed[1], "state/ct22")
            # comap info
            __tb_add_bar(next_state.compressor_map[0], "state/comp32")
            __tb_add_bar(next_state.compressor_map[1], "state/comp22")

        # simulation info
        ppa_dict = next_evaluate_worker.consult_ppa()
        for ppa_key in ppa_dict:
            logger.tb_logger.add_scalar(f'ppa/{ppa_key}', ppa_dict[ppa_key], global_step=self.total_steps)
        logger.tb_logger.add_scalar(f'ppa/ppa', self.env.get_ppa(ppa_dict), global_step=self.total_steps)
        if self.log_level > 1:
            power_mask_32, power_mask_22 = next_evaluate_worker.consult_compressor_power_mask(next_state.archive())
            power_mask_total = power_mask_32 + power_mask_22
            __tb_add_fig(power_mask_32, "power mask/ct32")
            __tb_add_fig(power_mask_22, "power mask/ct22")
            __tb_add_fig(power_mask_total, "power mask/total")

        # pp routing info
        if self.log_level > 1:
            if self.env.use_routing_optimize:
                evaluate_worker_no_routing: EvaluateWorker = step_info_dict["evaluate_worker_no_routing"]
                power_mask_32_no_routing, power_mask_22_no_routing = evaluate_worker_no_routing.consult_compressor_power_mask(next_state.archive())
                power_mask_total_no_routing = power_mask_32_no_routing + power_mask_22_no_routing
                __tb_add_fig(power_mask_total_no_routing - power_mask_total, "routing_info/power incr mask")
                ppa_dict_no_routing = evaluate_worker_no_routing.consult_ppa()
                for ppa_key in ppa_dict_no_routing.keys():
                    logger.tb_logger.add_scalar(f'routing_info/{ppa_key} incr', ppa_dict_no_routing[ppa_key] - ppa_dict[ppa_key], global_step=self.total_steps)

        # found best info
        logger.tb_logger.add_scalar(f'found_best_info/found_best_ppa', self.found_best_info["found_best_ppa"], global_step=self.total_steps)
        found_best_evaluate_worker: EvaluateWorker = self.found_best_info["found_best_evaluate_worker"]
        found_best_ppa_dict = found_best_evaluate_worker.consult_ppa()
        for key in found_best_ppa_dict.keys():
            logger.tb_logger.add_scalar(f'found_best_info/found_best_{key}', found_best_ppa_dict[key], global_step=self.total_steps)

        # optimization ratio
        initial_evaluate_worker: EvaluateWorker = self.env.initial_evaluate_worker
        initial_ppa_dict = initial_evaluate_worker.consult_ppa()
        for key in initial_ppa_dict.keys():
            initial_value = initial_ppa_dict[key]
            cur_value = ppa_dict[key]
            cur_incr = 100 * (initial_value - cur_value) / initial_value
            logger.tb_logger.add_scalar(f'optimization ratio (%) current/{key}', cur_incr, global_step=self.total_steps)

            best_value = found_best_ppa_dict[key]
            best_incr = 100 * (initial_value - best_value) / initial_value
            logger.tb_logger.add_scalar(f'optimization ratio (%) found best/{key}', best_incr, global_step=self.total_steps)

        if self.env.use_routing_optimize:
            initial_evaluate_worker_no_routing = self.env.initial_evaluate_worker_no_routing
            initial_ppa_dict_no_routing = initial_evaluate_worker_no_routing.consult_ppa()
            for key in initial_ppa_dict.keys():
                initial_value = initial_ppa_dict_no_routing[key]
                cur_value = ppa_dict[key]
                cur_incr = 100 * (initial_value - cur_value) / initial_value
                logger.tb_logger.add_scalar(f'optimization ratio (% no routing) current/{key}', cur_incr, global_step=self.total_steps)

                best_value = found_best_ppa_dict[key]
                best_incr = 100 * (initial_value - best_value) / initial_value
                logger.tb_logger.add_scalar(f'optimization ratio (% no routing) found best/{key}', best_incr, global_step=self.total_steps)
    # fmt: on

    # fmt: off
    def log_state_mutual_distances(self, state_mutual_distances):
        fig1 = plt.figure()
        f1 = plt.imshow(state_mutual_distances)
        number_states = state_mutual_distances.shape[0]
        # Loop over data dimensions and create text annotations.
        for i in range(number_states):
            for j in range(number_states):
                text = plt.text(j, i, state_mutual_distances[i, j], ha="center", va="center", color="w")
        logger.tb_logger.add_figure("state mutual distance", fig1, global_step=self.total_steps)
    # fmt: on

    # fmt: off
    def log_env_pool(self):
        total_state_num = len(self.env.initial_state_pool)
        logger.tb_logger.add_scalar("env pool total num", total_state_num, global_step=self.total_steps)
    # fmt: on

    # fmt: off
    def run_episode(self, episode_num):
        # reset state
        episode_opt_target_value = {}
        for ppa_key in self.env.opt_target_label:
            episode_opt_target_value[ppa_key] = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
            else:
                raise NotImplementedError

        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar(
                'env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)

            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            step_info_dict = self.env.step(action)
            next_state: State = step_info_dict["next_state"]
            reward = step_info_dict["reward"]
            next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]

            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward)
            elif self.store_type == "detail":
                raise NotImplementedError

            # update initial state pool
            self.update_env_initial_state_pool(next_state, next_evaluate_worker)
            # update q policy
            loss, q_info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(self.q_policy.state_dict())
            # reset agent
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(loss, action, step_info_dict, q_info)

            next_ppa_dict = next_evaluate_worker.consult_ppa()
            for ppa_key in self.env.opt_target_label:
                episode_opt_target_value[ppa_key].append(next_ppa_dict[ppa_key])
            avg_ppa = self.env.get_ppa(next_ppa_dict)
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

            state = copy.deepcopy(next_state)
        # update target q
        self.target_q_policy.load_state_dict(
            self.q_policy.state_dict()
        )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_opt_target_value)
    # fmt: on

    # fmt: off
    def store(self, state: State, next_state: State, action, reward):
        self.replay_memory.push(
            torch.tensor(state.archive(True)),
            torch.tensor(next_state.archive(True)),
            action,
            torch.tensor([reward]))
    # fmt: on
    
    # fmt: off
    def compute_values(self, state_batch, action_batch):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            image_state: State = state_batch[i]
            if action_batch is not None:
                q_values = self.q_policy(image_state.unsqueeze(0).float()).reshape(self.q_policy.num_classes)
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(image_state.unsqueeze(0).float())
                if self.is_double_q:
                    raise NotImplementedError
                else:
                    state_action_values[i] = q_values.max(1)[0].detach()
        return state_action_values
    # fmt: on

    # fmt: off
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.0
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = MaskTransition(*zip(*transitions))
            state_batch = batch.state
            next_state_batch = batch.next_state
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.compute_values(state_batch, action_batch)
            next_state_values = self.compute_values(next_state_batch, None)
            target_state_action_values = next_state_values * self.gamma + reward_batch.to(self.device)

            loss = self.loss_fn(
                state_action_values.unsqueeze(1),
                target_state_action_values.unsqueeze(1),
            )
            self.policy_optimizer.zero_grad()
            loss.backward()
            for param in self.q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_optimizer.step()
            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float()),
            }
        return loss, info
    # fmt: on


class MaskRNDDQNAlgorithm(MaskDQNAlgorithm):
    def __init__(
        self,
        env: RefineEnv,
        q_policy: MaskDeepQPolicy,
        target_q_policy: MaskDeepQPolicy,
        replay_memory: MaskReplayMemory,
        rnd_predictor: MaskDeepQPolicy,
        rnd_target: MaskDeepQPolicy,
        int_reward_run_mean_std: RunningMeanStd,
        rnd_lr=3e-4,
        update_rnd_freq=10,
        int_reward_scale=1,
        evaluate_freq=5,
        evaluate_num=5,
        bonus_type="rnd",  # rnd/noveld
        noveld_alpha=0.1,
        # n step q-learning
        n_step_num=5,
        **dqn_alg_kwargs,
    ):
        super().__init__(
            env, q_policy, target_q_policy, replay_memory, **dqn_alg_kwargs
        )
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
            self.rnd_predictor.parameters(), lr=self.rnd_lr
        )
        # loss func
        self.rnd_loss = nn.MSELoss()
        # log
        self.rnd_loss_item = 0.0
        self.rnd_int_rewards = 0.0
        self.rnd_ext_rewards = 0.0
        # cosine similarity
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        self.timer = {
            "pre": [],
            "action": [],
            "step": [],
            "update": [],
            "log": [],
        }

    def update_reward_int_run_mean_std(self, rewards):
        mean, std, count = np.mean(rewards), np.std(rewards), len(rewards)
        self.int_reward_run_mean_std.update_from_moments(mean, std**2, count)

    # fmt: off
    def update_rnd_model(self, state_batch):
        loss = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            image_state: State = state_batch[i]

            predict_value = self.rnd_predictor(image_state.unsqueeze(0).float()).reshape(self.q_policy.num_classes)
            with torch.no_grad():
                target_value = self.rnd_target(image_state.unsqueeze(0).float()).reshape(self.q_policy.num_classes)
            # set_trace()
            loss[i] = self.rnd_loss(predict_value, target_value)
        loss = torch.mean(loss)
        self.rnd_model_optimizer.zero_grad()
        loss.backward()
        for param in self.rnd_predictor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.rnd_model_optimizer.step()
        # update log
        self.rnd_loss_item = loss.item()
        return loss
    # fmt: on

    # fmt: off
    def compute_int_rewards(self, state_batch):
        batch_size = len(state_batch)
        int_rewards = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            image_state: State = state_batch[i]
            with torch.no_grad():
                predict_value = self.rnd_predictor(image_state.unsqueeze(0).float()).reshape(self.q_policy.num_classes)
                target_value = self.rnd_target(image_state.unsqueeze(0).float()).reshape(self.q_policy.num_classes)
            int_rewards[i] = torch.sum((predict_value - target_value) ** 2)
        return int_rewards

    # fmt: off
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = MaskTransition(*zip(*transitions))

            next_state_batch = batch.next_state
            state_batch = batch.state
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            # update reward int run mean std
            self.update_reward_int_run_mean_std(reward_batch.cpu().numpy())
            # compute reward int
            int_rewards_batch = self.compute_int_rewards(next_state_batch)
            if self.bonus_type == "noveld":
                int_rewards_last_state_batch = self.compute_int_rewards(state_batch)
                int_rewards_batch = int_rewards_batch - \
                    self.noveld_alpha * int_rewards_last_state_batch

            int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(state_batch, action_batch)
            next_state_values = self.compute_values(next_state_batch, None)
            target_state_action_values = (next_state_values * self.gamma) + train_reward_batch

            loss = self.loss_fn(state_action_values.unsqueeze(1), target_state_action_values.unsqueeze(1))

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
                rnd_loss = self.update_rnd_model(next_state_batch)
        return loss, info
    # fmt: on

    # fmt: off
    def run_episode(self, episode_num):
        # reset state
        episode_opt_target_value = {}
        for ppa_key in self.env.opt_target_label:
            episode_opt_target_value[ppa_key] = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
            else:
                raise NotImplementedError

        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            start_time = time.time()

            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar('env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)

            pre_time = time.time()
            logging.critical(f"pre time: {pre_time - start_time}")
            self.timer["pre"].append(pre_time - start_time)
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            action_time = time.time()
            logging.critical(f"select_action time: {action_time - pre_time}")
            self.timer["action"].append(action_time - pre_time)

            logger.log(f"total steps: {self.total_steps}, action: {action}")
            step_info_dict = self.env.step(action)
            next_state: State = step_info_dict["next_state"]
            reward = step_info_dict["reward"]
            next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]

            step_time = time.time()
            logging.critical(f"step time: {step_time - action_time}")
            self.timer["step"].append(step_time - action_time)
            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward)
            elif self.store_type == "detail":
                raise NotImplementedError

            # update initial state pool
            self.update_env_initial_state_pool(next_state, next_evaluate_worker)
            # update q policy
            loss, q_info = self.update_q()

            update_time = time.time()
            logging.critical(f"update time: {update_time - step_time}")
            self.timer["update"].append(update_time - step_time)
            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(self.q_policy.state_dict())
            # reset agent
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(loss, action, step_info_dict, q_info)            

            next_ppa_dict = next_evaluate_worker.consult_ppa()
            for ppa_key in self.env.opt_target_label:
                episode_opt_target_value[ppa_key].append(next_ppa_dict[ppa_key])
            avg_ppa = self.env.get_ppa(next_ppa_dict)
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")
            self.log_rnd_stats(q_info)

            log_time = time.time()
            logging.critical(f"log time: {log_time - update_time}")
            self.timer["log"].append(log_time - update_time)
            state = copy.deepcopy(next_state)
        with open(f"timer-log-{episode_num}.json", "w") as file:
            json.dump(self.timer, file)
        self.timer = {
            "pre":[],
            "action":[],
            "step": [],
            "update": [],
            "log": [],
        }
        # update target q
        self.target_q_policy.load_state_dict(self.q_policy.state_dict())
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_opt_target_value)
    # fmt: on

    # fmt: off
    def log_rnd_stats(self, info):
        # int reward vs rewards
        logger.tb_logger.add_scalar("batch int rewards", self.rnd_int_rewards, global_step=self.total_steps)
        logger.tb_logger.add_scalar("batch ext rewards", self.rnd_ext_rewards, global_step=self.total_steps)
        logger.tb_logger.add_scalar("rnd loss", self.rnd_loss_item, global_step=self.total_steps)
    # fmt: on


class RNDMultiAgentAlgorithm(RNDDQNAlgorithm):
    def __init__(
        self,
        env: RefineEnvMultiAgent,
        q_policy,
        target_q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        # prefix adder 相关
        q_policy_prefix_tree,
        target_q_policy_prefix_tree,
        rnd_predictor_prefix_tree,
        rnd_target_prefix_tree,
        # 其他参数
        use_prefix_adder_power_mask=False,
        use_power_mask=False,
        **dqn_alg_kwargs,
    ):
        super().__init__(
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            rnd_predictor,
            rnd_target,
            int_reward_run_mean_std,
            **dqn_alg_kwargs,
        )
        self.q_policy_prefix_tree = q_policy_prefix_tree
        self.target_q_policy_prefix_tree = target_q_policy_prefix_tree
        self.rnd_predictor_prefix_tree = rnd_predictor_prefix_tree
        self.rnd_target_prefix_tree = rnd_target_prefix_tree

        self.use_prefix_adder_power_mask = use_prefix_adder_power_mask
        self.use_power_mask = use_power_mask

        self.policy_prefix_tree_optimizer = self.optimizer_class(
            self.q_policy_prefix_tree.parameters(), lr=self.q_net_lr
        )

        self.rnd_model_prefix_tree_optimizer = self.optimizer_class(
            self.rnd_predictor_prefix_tree.parameters(), lr=self.rnd_lr
        )

        self.rnd_prefix_tree_loss = nn.MSELoss()

    # fmt: off
    def compute_values(
        self,
        # ct
        ct_image_state_batch,
        ct_mask_batch,
        # pt
        pt_image_state_batch,
        # action
        action_ct_batch,
        action_pt_batch,
    ):
        """
        计算联合的 values
        """
        batch_size = len(ct_image_state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)

        if not self.is_adder_only:
            for i in range(batch_size):
                ct_image_state = ct_image_state_batch[i]
                ct_mask = ct_mask_batch[i]

                pt_image_state = pt_image_state_batch[i]


                if action_ct_batch is not None:
                    action_ct = action_ct_batch[i]
                    action_pt = action_pt_batch[i]
                    ct_q_values = self.q_policy(ct_image_state.unsqueeze(0).float(), state_mask=ct_mask).flatten()
                    pt_q_values = self.q_policy_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()

                    # 加号出现在这里！
                    state_action_values[i] = ct_q_values[action_ct] + pt_q_values[action_pt]
                else:
                    # target q
                    ct_q_values = self.q_policy(ct_image_state.unsqueeze(0).float(), state_mask=ct_mask).flatten()
                    pt_q_values = self.q_policy_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                    ct_action_value = ct_q_values.max().detach()
                    # 集合求和最大 但是求和最大等价于每个最大
                    state_action_values[i] = ct_action_value + pt_q_values.max().detach()
        else:
            for i in range(batch_size):
                pt_image_state = pt_image_state_batch[i]
                if action_ct_batch is not None:
                    action_pt = action_pt_batch[i]
                    pt_q_values = self.q_policy_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()

                    state_action_values[i] = pt_q_values[action_pt]
                else:
                    # target q
                    pt_q_values = self.q_policy_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()

                    state_action_values[i] = pt_q_values.max().detach()
        return state_action_values
    
    # fmt: off
    def compute_int_rewards(
            self,
            ct_image_state_batch,
            ct_mask_batch,
            pt_image_state_batch,
        ):
        batch_size = len(ct_image_state_batch)
        int_rewards = torch.zeros(batch_size, device=self.device)
        if not self.is_adder_only:
            for i in range(batch_size):
                ct_image_state = ct_image_state_batch[i]
                ct_mask = ct_mask_batch[i]
                pt_image_state = pt_image_state_batch[i]
                with torch.no_grad():
                    ct_predict_value = self.rnd_predictor(ct_image_state.unsqueeze(0).float(), is_target=self.is_target, state_mask=ct_mask).flatten()
                    pt_predict_value = self.rnd_predictor_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()

                    ct_target_value = self.rnd_target(ct_image_state.unsqueeze(0).float(), is_target=self.is_target, state_mask=ct_mask).flatten()
                    pt_target_value = self.rnd_target_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                int_rewards[i] = torch.sum((ct_predict_value - ct_target_value) ** 2) + torch.sum((pt_predict_value - pt_target_value) ** 2)
        else:
            for i in range(batch_size):
                pt_image_state = pt_image_state_batch[i]
                with torch.no_grad():
                    pt_predict_value = self.rnd_predictor_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                    pt_target_value = self.rnd_target_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                int_rewards[i] = torch.sum((pt_predict_value - pt_target_value) ** 2)
        return int_rewards

    # fmt: off
    def update_q(self):
        """
        更新联合 Q 函数
        """
        if len(self.replay_memory) < self.batch_size:
            loss = 0.0
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = MARLTransition(*zip(*transitions))

            ct_image_state_batch = batch.ct_image_state
            ct_mask_batch = batch.ct_mask
            pt_image_state_batch = batch.pt_image_state

            next_ct_image_state_batch = batch.next_ct_image_state
            next_ct_mask_batch = batch.next_ct_mask
            next_pt_image_state_batch = batch.next_pt_image_state

            action_ct_batch = batch.action_ct
            action_pt_batch = batch.action_pt
            reward_batch = torch.cat(batch.reward)

            # update reward int run mean std
            self.update_reward_int_run_mean_std(reward_batch.cpu().numpy())
            int_rewards_batch = self.compute_int_rewards(next_ct_image_state_batch, next_ct_mask_batch, next_pt_image_state_batch)
            if self.bonus_type == "noveld":
                int_rewards_last_state_batch = self.compute_int_rewards(ct_image_state_batch, ct_mask_batch, pt_image_state_batch)
                int_rewards_batch = int_rewards_batch - \
                    self.noveld_alpha * int_rewards_last_state_batch
            int_rewards_batch = int_rewards_batch / \
                torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)

            train_reward_batch = reward_batch.to(
                self.device) + self.int_reward_scale * int_rewards_batch

            # values
            state_action_values = self.compute_values(
                ct_image_state_batch,
                ct_mask_batch,
                pt_image_state_batch,
                action_ct_batch,
                action_pt_batch,
            )

            next_state_values = self.compute_values(
                next_ct_image_state_batch,
                next_ct_mask_batch,
                next_pt_image_state_batch,
                None,
                None,
            )

            target_state_action_values = next_state_values * self.gamma + train_reward_batch
            loss = self.loss_fn(state_action_values.unsqueeze(1), target_state_action_values.unsqueeze(1))
            if not self.is_adder_only:
                self.policy_optimizer.zero_grad()
            self.policy_prefix_tree_optimizer.zero_grad()
            loss.backward()
            if not self.is_adder_only:
                for param in self.q_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
            for param in self.q_policy_prefix_tree.parameters():
                param.grad.data.clamp_(-1, 1)
            if not self.is_adder_only:
                self.policy_optimizer.step()
            self.policy_prefix_tree_optimizer.step()

            info = {
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
            }
            self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
            self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())

            if self.total_steps % self.update_rnd_freq == 0:
                rnd_loss = self.update_rnd_model(
                    next_ct_image_state_batch, next_ct_mask_batch, pt_image_state_batch
                )
        return loss, info
    
    # fmt: off
    def update_rnd_model(
            self,
            ct_image_state_batch,
            ct_mask_batch,
            pt_image_state_batch,
        ):
        loss = torch.zeros(self.batch_size, device=self.device)

        if not self.is_adder_only:
            for i in range(self.batch_size):
                ct_image_state = ct_image_state_batch[i]
                ct_mask = ct_mask_batch[i]
                pt_image_state = pt_image_state_batch[i]

                predict_value = self.rnd_predictor(ct_image_state.unsqueeze(0).float(), is_target=self.is_target, state_mask=ct_mask).flatten()
                pt_predict_value = self.rnd_predictor_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                with torch.no_grad():
                    target_value = self.rnd_target(ct_image_state.unsqueeze(0).float(), is_target=self.is_target, state_mask=ct_mask).flatten()
                    pt_target_value = self.rnd_target_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                # set_trace()
                loss[i] = self.rnd_loss(predict_value, target_value) + self.rnd_prefix_tree_loss(pt_predict_value, pt_target_value)

            loss = torch.mean(loss)
            self.rnd_model_optimizer.zero_grad()
            self.rnd_model_prefix_tree_optimizer.zero_grad()
            loss.backward()
            for param in self.rnd_predictor.parameters():
                param.grad.data.clamp_(-1, 1)
            for param in self.rnd_predictor_prefix_tree.parameters():
                param.grad.data.clamp_(-1, 1)
            self.rnd_model_optimizer.step()
            self.rnd_model_prefix_tree_optimizer.step()
        else:
            for i in range(self.batch_size):
                pt_image_state = pt_image_state_batch[i]
                pt_predict_value = self.rnd_predictor_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                with torch.no_grad():
                    pt_target_value = self.rnd_target_prefix_tree(pt_image_state.unsqueeze(0).float()).flatten()
                # set_trace()
                loss[i] = self.rnd_prefix_tree_loss(pt_predict_value, pt_target_value)

            loss = torch.mean(loss)
            self.rnd_model_prefix_tree_optimizer.zero_grad()
            loss.backward()
            for param in self.rnd_predictor_prefix_tree.parameters():
                param.grad.data.clamp_(-1, 1)
            self.rnd_model_prefix_tree_optimizer.step()
        # update log
        self.rnd_loss_item = loss.item()
        return loss
    # fmt: on

    def store(  # visual
        self,
        ct_image_state,
        ct_mask,
        next_ct_image_state,
        next_ct_mask,
        pt_image_state,
        next_pt_image_state,
        action_ct,
        action_pt,
        reward,
    ):
        if not self.is_adder_only:
            self.replay_memory.push(
                ct_image_state,
                ct_mask,
                next_ct_image_state,
                next_ct_mask,
                pt_image_state,
                next_pt_image_state,
                torch.tensor([action_ct]),
                torch.tensor([action_pt]),
                torch.tensor([reward]),
            )
        else:
            self.replay_memory.push(
                ct_image_state,
                ct_mask,
                next_ct_image_state,
                next_ct_mask,
                pt_image_state,
                next_pt_image_state,
                action_ct,
                torch.tensor([action_pt]),
                torch.tensor([reward]),
            )

    # fmt: off
    def log_stats(self, loss, action_ct, action_pt, step_info_dict, q_info):
        # 方便画图的小函数
        def __tb_add_fig(data, label):
            if self.log_level > 1:
                fig = plt.figure()
                plt.cla()
                plt.imshow(data)
                plt.colorbar()
                plt.tight_layout()
                logger.tb_logger.add_figure(label, fig, self.total_steps)

        # q_info and loss
        try:
            loss = loss.item()
            q_values = np.mean(q_info['q_values'])
            target_q_values = np.mean(q_info['target_q_values'])
            positive_rewards_number = q_info['positive_rewards_number']
            logger.tb_logger.add_histogram('q_values_hist', q_info['q_values'], global_step=self.total_steps)
            logger.tb_logger.add_histogram('target_q_values_hist', q_info["target_q_values"], global_step=self.total_steps)
        except:
            loss = loss
            q_values = 0.
            target_q_values = 0.
            positive_rewards_number = 0.
        logger.tb_logger.add_scalar('q_info/train loss', loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/q_values', q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/target_q_values', target_q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar('q_info/positive_rewards_number', positive_rewards_number, global_step=self.total_steps)

        # step info
        next_state: State = step_info_dict["next_state"]
        reward = step_info_dict["reward"]
        next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]

        if not self.is_adder_only:
            logger.tb_logger.add_scalar('action_ct/value', action_ct, global_step=self.total_steps)
            logger.tb_logger.add_scalar('action_ct/column', action_ct // 4, global_step=self.total_steps)
            logger.tb_logger.add_scalar('action_ct/type', action_ct % 4, global_step=self.total_steps)
            
            pp_len = next_state.get_pp_len()
        else:
            pp_len = len(next_state.cell_map)
        logger.tb_logger.add_scalar('action_pt/value', action_pt, global_step=self.total_steps)
        logger.tb_logger.add_scalar('action_pt/type', action_pt // (pp_len ** 2), global_step=self.total_steps)
        logger.tb_logger.add_scalar('action_pt/x', (action_pt % (pp_len ** 2)) // pp_len, global_step=self.total_steps)
        logger.tb_logger.add_scalar('action_pt/y', (action_pt % (pp_len ** 2)) % pp_len, global_step=self.total_steps)
        
        logger.tb_logger.add_scalar('reward', reward, global_step=self.total_steps)

        ppa_dict = next_evaluate_worker.consult_ppa()
        
        __tb_add_fig(next_state.cell_map, "state/cellmap")
        for ppa_key in ppa_dict:
            logger.tb_logger.add_scalar(f'ppa/{ppa_key}', ppa_dict[ppa_key], global_step=self.total_steps)
        logger.tb_logger.add_scalar(f'ppa/ppa', self.env.get_ppa(ppa_dict), global_step=self.total_steps)
        if not self.is_adder_only:
            ct_decomposed = next_state.archive()
            __tb_add_fig(ct_decomposed[0], "state/ct32")
            __tb_add_fig(ct_decomposed[1], "state/ct22")
            logger.tb_logger.add_scalar('stage num', next_state.get_stage_num(), global_step=self.total_steps)

            # simulation info
            power_mask_32, power_mask_22 = next_evaluate_worker.consult_compressor_power_mask(next_state.archive())
            power_mask_total = power_mask_32 + power_mask_22
            __tb_add_fig(power_mask_32, "power mask/ct32")
            __tb_add_fig(power_mask_22, "power mask/ct22")
            __tb_add_fig(power_mask_total, "power mask/total")

            # pp routing info
            if self.env.use_routing_optimize:
                evaluate_worker_no_routing: EvaluateWorker = step_info_dict["evaluate_worker_no_routing"]
                power_mask_32_no_routing, power_mask_22_no_routing = evaluate_worker_no_routing.consult_compressor_power_mask(next_state.archive())
                power_mask_total_no_routing = power_mask_32_no_routing + power_mask_22_no_routing
                __tb_add_fig(power_mask_total_no_routing - power_mask_total, "routing_info/power incr mask")
                ppa_dict_no_routing = evaluate_worker_no_routing.consult_ppa()
                for ppa_key in ppa_dict_no_routing.keys():
                    logger.tb_logger.add_scalar(f'routing_info/{ppa_key} incr', ppa_dict_no_routing[ppa_key] - ppa_dict[ppa_key], global_step=self.total_steps)

        # found best info
        logger.tb_logger.add_scalar(f'found_best_info/found_best_ppa', self.found_best_info["found_best_ppa"], global_step=self.total_steps)
        found_best_evaluate_worker: EvaluateWorker = self.found_best_info["found_best_evaluate_worker"]
        found_best_ppa_dict = found_best_evaluate_worker.consult_ppa()
        for key in found_best_ppa_dict.keys():
            logger.tb_logger.add_scalar(f'found_best_info/found_best_{key}', found_best_ppa_dict[key], global_step=self.total_steps)

        # optimization ratio
        initial_evaluate_worker: EvaluateWorker = self.env.initial_evaluate_worker
        initial_ppa_dict = initial_evaluate_worker.consult_ppa()
        for key in initial_ppa_dict.keys():
            initial_value = initial_ppa_dict[key]
            cur_value = ppa_dict[key]
            cur_incr = 100 * (initial_value - cur_value) / initial_value
            logger.tb_logger.add_scalar(f'optimization ratio (%) current/{key}', cur_incr, global_step=self.total_steps)

            best_value = found_best_ppa_dict[key]
            best_incr = 100 * (initial_value - best_value) / initial_value
            logger.tb_logger.add_scalar(f'optimization ratio (%) found best/{key}', best_incr, global_step=self.total_steps)

        if not self.is_adder_only:
            if self.env.use_routing_optimize:
                initial_evaluate_worker_no_routing = self.env.initial_evaluate_worker_no_routing
                initial_ppa_dict_no_routing = initial_evaluate_worker_no_routing.consult_ppa()
                for key in initial_ppa_dict.keys():
                    initial_value = initial_ppa_dict_no_routing[key]
                    cur_value = ppa_dict[key]
                    cur_incr = 100 * (initial_value - cur_value) / initial_value
                    logger.tb_logger.add_scalar(f'optimization ratio (% no routing) current/{key}', cur_incr, global_step=self.total_steps)

                    best_value = found_best_ppa_dict[key]
                    best_incr = 100 * (initial_value - best_value) / initial_value
                    logger.tb_logger.add_scalar(f'optimization ratio (% no routing) found best/{key}', best_incr, global_step=self.total_steps)
    # fmt: on

    # fmt: off
    def run_episode(self, episode_num):
        # reset state
        episode_opt_target_value = {}
        for ppa_key in self.env.opt_target_label:
            episode_opt_target_value[ppa_key] = []
        state_value = 0.
        info_count = None
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
            else:
                raise NotImplementedError

        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar(
                'env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)

            self.total_steps += 1
            # environment interaction
            if not self.is_adder_only:
                action_ct, ct_policy_info = self.q_policy.select_action(
                    state,
                    self.total_steps,
                    deterministic=self.deterministic,
                    is_softmax=self.is_softmax
                )
            else:
                action_ct = None
            action_pt, pt_policy_info = self.q_policy_prefix_tree.select_action(
                state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action_ct: {action_ct}, action_pt: {action_pt}")
            step_info_dict = self.env.step(action_ct, action_pt)
            next_state: State = step_info_dict["next_state"]
            reward = step_info_dict["reward"]
            next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]
            # next_state_mask = next_state.mask_with_legality()

            # store data
            if not self.is_adder_only:
                if self.store_type == "simple":
                    self.store(
                        torch.tensor(state.archive(self.use_power_mask)),
                        torch.tensor(state.mask_with_legality()),

                        torch.tensor(next_state.archive(self.use_power_mask)),
                        torch.tensor(next_state.mask_with_legality()),

                        torch.tensor(state.archive_cell_map(self.use_prefix_adder_power_mask)),
                        torch.tensor(next_state.archive_cell_map(self.use_prefix_adder_power_mask)),

                        action_ct,
                        action_pt,
                        reward,
                    )
                elif self.store_type == "detail":
                    raise NotImplementedError
            else:
                if self.store_type == "simple":
                    self.store(
                        None,
                        None,

                        None,
                        None,

                        torch.tensor(state.archive_cell_map(self.use_prefix_adder_power_mask)),
                        torch.tensor(next_state.archive_cell_map(self.use_prefix_adder_power_mask)),

                        action_ct,
                        action_pt,
                        reward,
                    )
                elif self.store_type == "detail":
                    raise NotImplementedError

            # update initial state pool
            self.update_env_initial_state_pool(next_state, next_evaluate_worker)
            # update q policy
            loss, q_info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                if not self.is_adder_only:
                    self.target_q_policy.load_state_dict(
                        self.q_policy.state_dict()
                    )
                self.target_q_policy_prefix_tree.load_state_dict(
                        self.q_policy_prefix_tree.state_dict()
                    )
            # reset agent
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(loss, action_ct, action_pt, step_info_dict, q_info)
            self.log_rnd_stats(q_info)

            next_ppa_dict = next_evaluate_worker.consult_ppa()
            for ppa_key in self.env.opt_target_label:
                episode_opt_target_value[ppa_key].append(next_ppa_dict[ppa_key])
            avg_ppa = self.env.get_ppa(next_ppa_dict)
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

            state = copy.deepcopy(next_state)
        # update target q
        if not self.is_adder_only:
            self.target_q_policy.load_state_dict(
                self.q_policy.state_dict()
            )
        self.target_q_policy_prefix_tree.load_state_dict(
            self.q_policy_prefix_tree.state_dict()
        )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_opt_target_value)
    # fmt: on

class PowerMaskRNDDQNAlgorithm(RNDDQNAlgorithm):
    def __init__(
        self,
        env: RefineEnv,
        q_policy,
        target_q_policy,
        replay_memory: PowerMaskReplayMemory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        mask_min = 0.5,
        is_power_mask_in = False,
        decay_rate = 0.85,  # mask_min 的衰减速度
        decay_steps = 100,  # mask_min 的衰减间隔步数
        **dqn_alg_kwargs,
    ):
        super().__init__(
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            rnd_predictor,
            rnd_target,
            int_reward_run_mean_std,
            **dqn_alg_kwargs,
        )
        self.is_power_mask_in = is_power_mask_in
        if is_power_mask_in:
            self.mask_min = 0
            self.decay_rate = 1
            self.decay_steps = 1
        else:
            self.mask_min = mask_min
            self.decay_rate = decay_rate
            self.decay_steps = decay_steps
        get_logger(logger_name='ysh').info('trainer info: is_power_mask_in={}\t mask_min={}\t decay_rate={}\t decay_steps={}'.format(
            self.is_power_mask_in, self.mask_min, self.decay_rate, self.decay_steps))
    
    def compute_values(self, state_batch, action_batch, state_mask, power_mask, is_average=False):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        time_getpower = 0
        for i in range(batch_size):
            # compute image state
            state: State = state_batch[i]
            ct32, ct22 = state.archive()
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            image_state = torch.cat((ct32, ct22), dim=0)
            mask = state_mask[i]
            if action_batch is not None:
                q_values = self.q_policy(image_state.unsqueeze(0).float(), state_mask=mask).reshape(4 * state.get_pp_len())
                state_action_values[i] = q_values[action_batch[i]]
            else:
                q_values = self.target_q_policy(image_state.unsqueeze(0).float(), state_mask=mask, is_target=True)
                if i == 0:
                    is_debug = True
                else:
                    is_debug = False
                is_debug = False
                q_values_woth_power = q_values.squeeze().detach() * power_mask[i]
                if self.is_power_mask_in: # power_mask_in=True 则表示 power_mask 仅用于选择 action 而不会影响 Q 的值
                    index = torch.argmax(q_values_woth_power)
                    state_action_values[i] = q_values.squeeze()[index].detach()
                else:
                    state_action_values[i] = q_values_woth_power.max().detach()
        return state_action_values
    
    # fmt: off
    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = PowerMaskTransition(*zip(*transitions))

            next_state_batch = batch.next_state
            state_batch = batch.state
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            state_mask = torch.cat(batch.mask)
            next_state_mask = torch.cat(batch.next_state_mask)
            next_state_power_mask = torch.cat(batch.next_state_power_mask)
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
                int_rewards_batch = int_rewards_batch - \
                    self.noveld_alpha * int_rewards_last_state_batch

            int_rewards_batch = int_rewards_batch / \
                torch.tensor(
                    np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
            train_reward_batch = reward_batch.to(
                self.device) + self.int_reward_scale * int_rewards_batch
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.compute_values(
                state_batch, action_batch, state_mask, None
            )
            next_state_values = self.compute_values(
                next_state_batch, None, next_state_mask, next_state_power_mask
            )
            target_state_action_values = (
                next_state_values * self.gamma) + train_reward_batch

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
    # fmt: on

    def store(  # visual
        self, state: State, next_state: State, action, reward, mask, next_state_mask, next_state_power_mask
    ):
        self.replay_memory.push(
            state,
            action,
            next_state,
            torch.tensor([reward]),
            mask.reshape(1, -1),
            next_state_mask.reshape(1, -1),
            next_state_power_mask.reshape(1, -1)
        )
    
    # fmt: off
    def run_episode(self, episode_num):
        # reset state
        episode_opt_target_value = {}
        for ppa_key in self.env.opt_target_label:
            episode_opt_target_value[ppa_key] = []
        state_value = 0.
        info_count = None
        action_indexs = []  # 为了 tb 记录 power_mask
        for i in range(self.env.action_type_num*self.env.get_pp_len()):
            action_indexs.append(i)
        get_logger(logger_name='ysh').info('action_indexs: {}'.format(action_indexs))
        
        if self.env.random_reset_steps >= self.total_steps:
            # random reset
            env_state, sel_index = self.env.reset()
            state = copy.deepcopy(env_state)
        else:
            # reset with value or novelty
            if self.env.reset_state_policy == "random":
                env_state, sel_index = self.env.reset()
                state = copy.deepcopy(env_state)
            else:
                raise NotImplementedError

        if self.env.initial_state_pool_max_len > 0:
            if len(self.env.initial_state_pool) >= 2:
                state_mutual_distances = self.env.get_mutual_distance()
                self.log_state_mutual_distances(state_mutual_distances)
                self.log_env_pool()

        for step in range(self.len_per_episode):
            logger.tb_logger.add_scalar('env_state_pool_value', np.mean(state_value), global_step=self.total_steps)
            logger.tb_logger.add_scalar(
                'env_state_pool_sel_index', sel_index, global_step=self.total_steps)
            if self.total_steps > self.env.random_reset_steps and self.env.reset_state_policy != "random":
                logger.tb_logger.add_histogram('env_state_pool_value_distribution', state_value, global_step=self.total_steps)
                if info_count is not None:
                    logger.tb_logger.add_scalar('info_count', np.mean(info_count), global_step=self.total_steps)
                    logger.tb_logger.add_histogram('info_count_distribution', info_count, global_step=self.total_steps)

            get_logger(logger_name='ysh').info('total_steps: {}'.format(self.total_steps))
            self.total_steps += 1
            # environment interaction
            action, policy_info = self.q_policy.select_action(
                state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax
            )
            logger.log(f"total steps: {self.total_steps}, action: {action}")
            step_info_dict = self.env.step(action)
            next_state: State = step_info_dict["next_state"]
            reward = step_info_dict["reward"]
            next_evaluate_worker: EvaluateWorker = step_info_dict["evaluate_worker"]
            next_state_mask = next_state.mask_with_legality()
            
            if self.decay_rate == 1:
                decayed_mask_min = self.mask_min
            elif self.mask_min == 1:
                decayed_mask_min = 1
            else:
                decayed_mask_min = 1 - (1-self.mask_min) * pow(self.decay_rate, (self.total_steps//self.decay_steps))
            next_state_power_mask, next_state_power_coefficient = next_state.get_power_mask(decayed_mask_min) # tensor
            next_state_power_mask_numpy = next_state_power_mask.numpy()
            
            delta_power = []
            for v in next_state_power_coefficient:
                if v < -10000:
                    continue
                else:
                    delta_power.append(v)
            logger.tb_logger.add_histogram('power_mask/delta_power_distribution', np.array(delta_power), global_step=self.total_steps)
            logger.tb_logger.add_histogram('power_mask/weight_distribution_with0', next_state_power_mask_numpy, global_step=self.total_steps)
            next_state_power_mask_no0 = []
            zero_count = 0
            for v in next_state_power_mask_numpy:
                if v == decayed_mask_min:
                    zero_count += 1
                    continue
                else:
                    next_state_power_mask_no0.append(v)
            logger.tb_logger.add_histogram('power_mask/weight_distribution', np.array(next_state_power_mask_no0), global_step=self.total_steps)
            logger.tb_logger.add_scalar('power_mask/zero_count', zero_count, global_step=self.total_steps)
            action_sample_data = random.choices(action_indexs, weights=next_state_power_mask_numpy, k=100)
            logger.tb_logger.add_histogram('power_mask/action_indexs', np.array(action_sample_data), global_step=self.total_steps)
            
            get_logger(logger_name='ysh').info('next_state_power_coefficient: {}'.format(str(next_state_power_coefficient)))
            get_logger(logger_name='ysh').info('next_state_power_mask: {}'.format(str(next_state_power_mask_numpy)))
            get_logger(logger_name='ysh').info('zero_count: {}'.format(str(zero_count)))
            
            next_state_power_mask = next_state_power_mask.to(self.device)
            
            # store data
            if self.store_type == "simple":
                self.store(state, next_state, action, reward, policy_info['mask'], torch.tensor(next_state_mask), next_state_power_mask)
            elif self.store_type == "detail":
                raise NotImplementedError

            # update initial state pool
            self.update_env_initial_state_pool(next_state, next_evaluate_worker)
            # update q policy
            loss, q_info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(
                    self.q_policy.state_dict()
                )
            # reset agent
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(loss, action, step_info_dict, q_info)
            self.log_rnd_stats(q_info)

            next_ppa_dict = next_evaluate_worker.consult_ppa()
            for ppa_key in self.env.opt_target_label:
                episode_opt_target_value[ppa_key].append(next_ppa_dict[ppa_key])
            # avg_ppa = self.env.get_ppa(next_ppa_dict, self.total_steps, get_logger(logger_name='ysh'))
            avg_ppa = self.env.get_ppa(next_ppa_dict, self.total_steps)
            logger.log(f"total steps: {self.total_steps}, avg ppa: {avg_ppa}")

            state = copy.deepcopy(next_state)
        # update target q
        self.target_q_policy.load_state_dict(
            self.q_policy.state_dict()
        )
        # process and log pareto
        self.process_and_log_pareto(episode_num, episode_opt_target_value)
    # fmt: on
        
