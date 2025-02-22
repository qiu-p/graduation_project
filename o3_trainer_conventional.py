import copy
import json
import math
import multiprocessing
import multiprocessing.pool
import os
import sys
from collections import Counter
from queue import PriorityQueue
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from paretoset import paretoset
from pygmo import hypervolume
from torch.utils.tensorboard import SummaryWriter

from o0_global_const import PartialProduct
from o0_logger import logger
from o0_mul_utils import (decompose_compressor_tree, get_compressor_tree,
                          get_initial_partial_product,
                          legalize_compressor_tree, write_mul)
from o0_rtl_tasks import EvaluateWorker
from o0_state import State
from o1_environment_speedup import SpeedUpRefineEnv
from o1_environment_speedup_conventional import SpeedUpRefineEnv_Conventional
from o2_policy import BasicBlock, DeepQPolicy
from o5_utils import (MBRLMultiObjTransition, MBRLTransition,
                      MultiObjTransition, PDMultiObjTransition, RunningMeanStd,
                      Transition, set_global_seed, setup_logger)


def transition(
        _ct: np.ndarray, action_column: int, action_type: int, bit_width, encode_type,
    ) -> np.ndarray:
        """
        action is a number, action coding:
            action=0: add a 2:2 compressor
            action=1: remove a 2:2 compressor
            action=2: replace a 3:2 compressor
            action=3: replace a 2:2 compressor
        Input: cur_state, action
        Output: next_state
        """
        ct = copy.deepcopy(_ct)
        if action_type == 0:
            # add a 2:2 compressor
            ct[1][action_column] += 1
        elif action_type == 1:
            # remove a 2:2 compressor
            ct[1][action_column] -= 1
        elif action_type == 2:
            # replace a 3:2 compressor with a 2:2 compressor
            ct[1][action_column] += 1
            ct[0][action_column] -= 1
        elif action_type == 3:
            # replace a 2:2 compressor with a 3:2 compressor
            ct[1][action_column] -= 1
            ct[0][action_column] += 1
        else:
            raise NotImplementedError

        legalized_ct32, legalized_ct22 = legalize_compressor_tree(get_initial_partial_product(bit_width, encode_type), ct[0], ct[1])
        legalized_ct = np.zeros_like(ct)
        legalized_ct[0] = legalized_ct32
        legalized_ct[1] = legalized_ct22
        return legalized_ct

def mask_with_legality(state, bit_width, encode_type, max_stage_num):
    """
    有些动作会导致结果 stage 超过 max stage
    因此需要被过滤掉

    问题: 哪些动作会让stage增加? 原来的方法是遍历 有没有更好的方法
    """

    initial_pp = get_initial_partial_product(bit_width, encode_type)
    mask = np.zeros([4 * len(initial_pp)])
    remain_pp = copy.deepcopy(initial_pp)
    for column_index in range(len(remain_pp)):
        if column_index > 0:
            remain_pp[column_index] += (
                state[0][column_index - 1] + state[1][column_index - 1]
            )
        remain_pp[column_index] += -2 * state[0][column_index] - state[1][column_index]

    legal_act = []
    for column_index in range(2, len(initial_pp)):
        if remain_pp[column_index] == 2:
            legal_act.append((column_index, 0))
            if state[1][column_index] >= 1:
                legal_act.append((column_index, 3))
        if remain_pp[column_index] == 1:
            if state[0][column_index] >= 1:
                legal_act.append((column_index, 2))
            if state[1][column_index] >= 1:
                legal_act.append((column_index, 1))

    for act_col, action in legal_act:
        pp = copy.deepcopy(remain_pp)
        ct = copy.deepcopy(state)

        # change the CT structure
        if action == 0:
            ct[1][act_col] = ct[1][act_col] + 1
            pp[act_col] = pp[act_col] - 1
            if act_col + 1 < len(pp):
                pp[act_col + 1] = pp[act_col + 1] + 1
        elif action == 1:
            ct[1][act_col] = ct[1][act_col] - 1
            pp[act_col] = pp[act_col] + 1
            if act_col + 1 < len(pp):
                pp[act_col + 1] = pp[act_col + 1] - 1
        elif action == 2:
            ct[1][act_col] = ct[1][act_col] + 1
            ct[0][act_col] = ct[0][act_col] - 1
            pp[act_col] = pp[act_col] + 1
        elif action == 3:
            ct[1][act_col] = ct[1][act_col] - 1
            ct[0][act_col] = ct[0][act_col] + 1
            pp[act_col] = pp[act_col] - 1

        # legalization
        # mask 值为1 代表这个动作合法 为0代表不合法
        for i in range(act_col + 1, len(pp) + 1):
            # column number restriction
            if i == len(pp):
                mask[act_col * 4 + action] = 1
                break
            elif pp[i] == 1 or pp[i] == 2:
                mask[act_col * 4 + action] = 1
                break
            elif pp[i] == 3:
                ct[0][i] = ct[0][i] + 1
                if i + 1 < len(pp):
                    pp[i + 1] = pp[i + 1] + 1
                pp[i] = 1
            elif pp[i] == 0:
                if ct[1][i] >= 1:
                    ct[1][i] = ct[1][i] - 1
                    if i + 1 < len(pp):
                        pp[i + 1] = pp[i + 1] - 1
                    pp[i] = 1
                else:
                    ct[0][i] = ct[0][i] - 1
                    if i + 1 < len(pp):
                        pp[i + 1] = pp[i + 1] - 1
                    pp[i] = 2
    mask = mask != 0

    indices = np.where(mask)[0]

    for action in indices:
            ct = copy.deepcopy(state)
            action_type = action % 4
            action_column = action // 4
            if action_type < 4:
                next_state = transition(ct, action_column, action_type, bit_width, encode_type)
                ct32, ct22, _, __ = decompose_compressor_tree(initial_pp, next_state[0], next_state[1])
                if len(ct32) >= max_stage_num:
                    mask[int(action)] = 0
    mask = (mask != 0)

    return mask


def step(ct, action, bit_width, encode_type):
    action_type = action % 4
    action_column = action // 4
    next_ct = copy.deepcopy(ct)
    if action_type == 0:
        # add a 2:2 compressor
        next_ct[1][action_column] += 1
    elif action_type == 1:
        # remove a 2:2 compressor
        next_ct[1][action_column] -= 1
    elif action_type == 2:
        # replace a 3:2 compressor with a 2:2 compressor
        next_ct[1][action_column] += 1
        next_ct[0][action_column] -= 1
    elif action_type == 3:
        # replace a 2:2 compressor with a 3:2 compressor
        next_ct[1][action_column] -= 1
        next_ct[0][action_column] += 1
    else:
        raise NotImplementedError
    pp = get_initial_partial_product(bit_width, encode_type)
    legalized_ct = legalize_compressor_tree(pp, next_ct[0], next_ct[1])
    legalized_ct = np.asarray(legalized_ct).astype(int).tolist()
    return legalized_ct


class SimulatedAnnealing:
    def __init__(
        self,
        env: SpeedUpRefineEnv,
        q_policy: DeepQPolicy,
        ##########################
        ##begin 没有实际作用参数，只是为了和 DeepQPolicy 保持一致
        total_episodes=400,
        MAX_STAGE_NUM=4,
        # pareto
        reference_point=[2600, 1.8],
        # store type
        # SimulatedAnnealing 相关参数
        max_iter=20,  # 迭代最大步数
        search_steps=5,  # 每次搜索新状态时随机走的步数
        T0=100,  # 初始温度
        Tf=0.1,  # 结束温度
        T_dec_factor=0.9,  # 温度的下降率
        is_random=False,
        store_trajectory_freq=25,
    ):
        self.env = env
        self.q_policy = q_policy
        # hyperparameter
        self.total_episodes = total_episodes
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.is_random = is_random

        self.store_trajectory_freq = store_trajectory_freq

        # total steps
        self.total_steps = 0
        self.total_actual_steps = 0
        self.bit_width = env.bit_width
        self.int_bit_width = env.int_bit_width
        self.initial_partial_product = PartialProduct[self.bit_width][:-1]

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
        }
        # agent reset

        # pareto pointset
        self.pareto_pointset = {"area": [], "delay": [], "state": []}
        self.reference_point = reference_point

        # SimulatedAnnealing
        self.max_iter = max_iter
        self.search_steps = search_steps
        self.T0 = T0
        self.Tf = Tf
        self.T_dec_factor = T_dec_factor

        # configure figure
        plt.switch_backend("agg")

    def store(
        self,
        state,
        next_state,
        action,
        reward,
        mask,
        next_state_mask,
        # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
        # rewards_dict
    ):
        state = np.reshape(state, (1, 2, int(self.int_bit_width * 2)))
        next_state = np.reshape(next_state, (1, 2, int(self.int_bit_width * 2)))
        self.replay_memory.push(
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1, -1),
            next_state_mask.reshape(1, -1),
            # state_ct32, state_ct22, next_state_ct32, next_state_ct22,
            # rewards_dict
        )

    def end_experiments(self, episode_num):
        pass

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
        # try:
        #     loss = loss.item()
        #     q_values = np.mean(info["q_values"])
        #     target_q_values = np.mean(info["target_q_values"])
        #     positive_rewards_number = info["positive_rewards_number"]
        # except:
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

        # for i in range(len(self.found_best_info)):
        #     logger.tb_logger.add_scalar(
        #         f"best ppa {i}-th weight",
        #         self.found_best_info[i]["found_best_ppa"],
        #         global_step=self.total_steps,
        #     )
        #     logger.tb_logger.add_scalar(
        #         f"best area {i}-th weight",
        #         self.found_best_info[i]["found_best_area"],
        #         global_step=self.total_steps,
        #     )
        #     logger.tb_logger.add_scalar(
        #         f"best delay {i}-th weight",
        #         self.found_best_info[i]["found_best_delay"],
        #         global_step=self.total_steps,
        #     )
        logger.tb_logger.add_scalar(
            "best ppa",
            self.found_best_info["found_best_ppa"],
            global_step=self.total_steps,
        )
        logger.tb_logger.add_scalar(
            "best area",
            self.found_best_info["found_best_area"],
            global_step=self.total_steps,
        )
        logger.tb_logger.add_scalar(
            "best delay",
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

    def post_process_pipline(self, num_steps: int, info_dict: dict) -> None:
        """
        记录每一步的结果。由于模拟退火比较特殊，需要在内循环结束后才能知道下一个外循环输入，所以totalstep在这里+1
        """
        for step in range(num_steps + 1):
            logger.log(
                f"total steps: {self.total_steps}, action: {info_dict[step]['action']}"
            )
            self.update_env_initial_state_pool(
                info_dict[step]["state"],
                info_dict[step]["reward_dict"],
                info_dict[step]["next_state_policy_info"]["mask"],
            )
            self.log_stats(
                0,
                0,
                info_dict[step]["reward_dict"],
                None,
                info_dict[step]["action"],
                {"q_values": 0, "target_q_values": 0},
                info_dict[step]["policy_info"],
            )
            logger.log(
                f"total steps: {self.total_steps}, avg ppa: {info_dict[step]['ppa']}"
            )
            self.total_steps += 1

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
                if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
                    # push the best ppa state into the initial pool
                    avg_area = np.mean(rewards_dict["area"])
                    avg_delay = np.mean(rewards_dict["delay"])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "ppa": rewards_dict["avg_ppa"],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa",
                            "normalize_area": rewards_dict["normalize_area"],
                            "normalize_delay": rewards_dict["normalize_delay"],
                        }
                    )
            elif self.env.store_state_type == "leq":
                if self.found_best_info["found_best_ppa"] >= rewards_dict["avg_ppa"]:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict["area"])
                    avg_delay = np.mean(rewards_dict["delay"])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "ppa": rewards_dict["avg_ppa"],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa",
                        }
                    )
            elif self.env.store_state_type == "ppa_with_diversity":
                # store best ppa state
                if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
                    # push the state to the initial pool
                    avg_area = np.mean(rewards_dict["area"])
                    avg_delay = np.mean(rewards_dict["delay"])
                    self.env.initial_state_pool.append(
                        {
                            "state": copy.deepcopy(state),
                            "area": avg_area,
                            "delay": avg_delay,
                            "ppa": rewards_dict["avg_ppa"],
                            "count": 1,
                            "state_mask": state_mask,
                            "state_type": "best_ppa",
                        }
                    )
                # store better diversity state
                elif rewards_dict["avg_ppa"] < self.env.initial_state_pool[0]["ppa"]:
                    number_states = len(self.env.initial_state_pool)
                    worse_state_distances = np.zeros(number_states - 1)
                    cur_state_distances = np.zeros(number_states)
                    for i in range(number_states):
                        cur_state_dis = np.linalg.norm(
                            state - self.env.initial_state_pool[i]["state"], ord=2
                        )
                        cur_state_distances[i] = cur_state_dis
                        if i == 0:
                            continue
                        worse_state_dis = np.linalg.norm(
                            self.env.initial_state_pool[0]["state"]
                            - self.env.initial_state_pool[i]["state"],
                            ord=2,
                        )
                        worse_state_distances[i - 1] = worse_state_dis

                    worse_state_min_distance = np.min(worse_state_distances)
                    cur_state_min_distance = np.min(cur_state_distances)
                    if cur_state_min_distance > worse_state_min_distance:
                        # push the diverse state into the pool
                        avg_area = np.mean(rewards_dict["area"])
                        avg_delay = np.mean(rewards_dict["delay"])
                        self.env.initial_state_pool.append(
                            {
                                "state": copy.deepcopy(state),
                                "area": avg_area,
                                "delay": avg_delay,
                                "ppa": rewards_dict["avg_ppa"],
                                "count": 1,
                                "state_mask": state_mask,
                                "state_type": "diverse",
                            }
                        )
        if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
            self.found_best_info["found_best_ppa"] = rewards_dict["avg_ppa"]
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict["area"])
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict["delay"])

    def log_and_save_pareto_points(self, ppas_dict, episode_num):
        save_data_dict = {}
        # save ppa_csv
        save_data_dict["testing_full_ppa"] = ppas_dict
        # compute pareto points
        area_list = ppas_dict["area"]
        delay_list = ppas_dict["delay"]
        data_points = pd.DataFrame({"area": area_list, "delay": delay_list})
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
        logger.tb_logger.add_scalar(
            "testing hypervolume", hv_value, global_step=episode_num
        )

        # save pareto points and log pareto points
        fig1 = plt.figure()
        x = true_pareto_area_list
        y = true_pareto_delay_list
        f1 = plt.scatter(x, y, c="r")
        logger.tb_logger.add_figure(
            "testing pareto points", fig1, global_step=episode_num
        )

        save_data_dict["testing_pareto_points_area"] = true_pareto_area_list
        save_data_dict["testing_pareto_points_delay"] = true_pareto_delay_list

        return save_data_dict

    def store_trajectory(self, episode_num):
        trajectory_path = os.path.join(
            logger._snapshot_dir, "trajectory", f"episode_{episode_num}.json"
        )
        if not os.path.exists(os.path.dirname(trajectory_path)):
            os.makedirs(os.path.dirname(trajectory_path))
        with open(trajectory_path, "w") as file:
            json.dump(self.trajectory_list, file)
        pass

    def run_experiments(self):
        env_state, sel_index = self.env.reset()
        T = self.T0
        state = copy.deepcopy(env_state)
        ppa = self.env.last_ppa
        self.trajectory_list = []

        for iteration in range(self.max_iter):
            trajectory = {
                "info": {
                    "episode_num": 0,
                },
                "data": [],
            }
            episode_area = []
            episode_delay = []
            # 内循环，每次随机在领域中选择state
            info_dict = {}  # 用于记录每步情况，key是step
            for step in range(self.search_steps):
                # 1. 随机转移到新的邻域状态上去
                action, policy_info = self.q_policy.select_action(
                    torch.tensor(state), 0
                )
                next_state, reward, rewards_dict = self.env.step(action)
                trajectory["data"].append(
                    {
                        "action": np.asarray(action.cpu()).tolist(),
                        "state": np.asarray(state).tolist(),
                        "rewards_dict": rewards_dict,
                    }
                )
                next_ppa = rewards_dict["avg_ppa"]
                _, next_state_policy_info = self.q_policy.select_action(
                    torch.tensor(state), 0
                )
                # 2. 计算增量
                delta_T = next_ppa - ppa
                if not self.is_random:
                    # 3. 按照概率转移
                    if delta_T < 0:
                        # 接受改变
                        state = copy.deepcopy(next_state)
                        ppa = next_ppa
                    else:
                        if np.random.uniform(0, 1) < np.exp(-delta_T / T):
                            # 以一个小概率接受改变
                            state = copy.deepcopy(next_state)
                            ppa = next_ppa
                        else:
                            # 不改变
                            self.env.cur_state = copy.deepcopy(state)
                else:
                    # 接受改变
                    state = copy.deepcopy(next_state)
                    ppa = next_ppa

                # 记录相关内容
                info_dict[step] = {
                    "ppa": ppa,
                    "action": action,
                    "state": copy.deepcopy(state),
                    "reward_dict": rewards_dict,
                    "policy_info": policy_info,
                    "next_state_policy_info": next_state_policy_info,
                }
                episode_area.extend(rewards_dict["area"])
                episode_delay.extend(rewards_dict["delay"])
                # end 内循环

            self.trajectory_list.append(trajectory)
            if iteration % self.store_trajectory_freq == 0:
                self.store_trajectory(iteration)

            # 4. 找到最好的状态，更新T，后处理
            if not self.is_random:
                best_ppa_found_step = min(
                    info_dict, key=lambda step: info_dict[step]["ppa"]
                )
            else:
                # 随机的，就找最后一个
                best_ppa_found_step = step
            best_ppa_found = info_dict[best_ppa_found_step]["ppa"]
            best_ppa_found_state = info_dict[best_ppa_found_step]["state"]

            # 重设环境状态
            self.env.cur_state = copy.deepcopy(best_ppa_found_state)

            # 更新状态
            state = best_ppa_found_state
            ppa = best_ppa_found

            # 不丢弃被截断的步骤，计入所有搜索的步骤
            self.post_process_pipline(len(info_dict) - 1, info_dict)
            # 降低 T
            iteration += 1
            T *= self.T_dec_factor
            # end 外循环
            if iteration % 10 == 0:
                self.end_experiments(iteration)
        # 结束实验
        self.store_trajectory(iteration)
        self.end_experiments(iteration)


class SimulatedAnnealing_v1:
    def __init__(
        self,
        bit_width=8,
        encode_type="and",
        init_ct_type="wallace",
        MAX_STAGE_NUM=6,
        area_scale=438.5,
        delay_scale=0.7499,
        ppa_scale=100,
        weight=[4, 1],
        reference_point=[2600, 1.8],
        max_iter=20,  # 迭代最大步数
        search_steps=5,  # 每次搜索新状态时随机走的步数
        T0=100,  # 初始温度
        Tf=0.1,  # 结束温度
        T_dec_factor=0.9,  # 温度的下降率
        store_trajectory_freq=25,
        end_exp_freq=25,
        build_path="build",
        target_delay=[50,250,400,650],
        n_processing=4,
    ):
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.init_ct_type = init_ct_type
        self.target_delay = target_delay
        self.MAX_STAGE_NUM = MAX_STAGE_NUM

        self.area_scale = area_scale
        self.delay_scale = delay_scale
        self.ppa_scale = ppa_scale
        self.weight = weight

        self.n_processing = n_processing
        self.end_exp_freq = end_exp_freq

        self.store_trajectory_freq = store_trajectory_freq

        # total steps
        self.total_steps = 0
        self.total_actual_steps = 0
        self.initial_partial_product = get_initial_partial_product(
            bit_width, encode_type
        )

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
        }

        # pareto pointset
        self.pareto_pointset = {"area": [], "delay": [], "state": []}
        self.reference_point = reference_point

        # SimulatedAnnealing
        self.max_iter = max_iter
        self.search_steps = search_steps
        self.T0 = T0
        self.Tf = Tf
        self.T_dec_factor = T_dec_factor
        
        self.initial_cwd_path = os.getcwd()
        self.build_path = os.path.join(self.initial_cwd_path, build_path)
        if not os.path.exists(self.build_path):
            os.mkdir(self.build_path)

        # configure figure
        plt.switch_backend("agg")

    # fmt: on
    @staticmethod
    def get_ppa(build_path_base, state, action, bit_width, encode_type, worker_id, target_delay_list, area_scale, delay_scale, ppa_scale, weight):
        cur_state = copy.deepcopy(state)
        next_state = step(cur_state, action, bit_width, encode_type)
        build_path = os.path.join(build_path_base, f"worker_{worker_id}")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(bit_width, encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, next_state[0], next_state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay_list, build_path, False, False, False, False, False, False, 1)
        worker.evaluate()

        ppa_dict = worker.consult_ppa()
        ppa_list = worker.consult_ppa_list()

        ppa = ppa_scale * (weight[0] * ppa_dict["area"] / area_scale + weight[1] * ppa_dict["delay"] / delay_scale)

        return {
            "state": next_state,
            "ppa": ppa,
            "action": action,
            "ppa_dict": ppa_dict,
            "ppa_list": ppa_list,
        }

    def run_experiments(self):
        self.trajectory_list = []
        
        T = self.T0
        state = get_compressor_tree(
            self.initial_partial_product, self.bit_width, self.init_ct_type
        )
        self.found_best_info["found_best_state"] = state
        ppa = 500
        info = {
            "state": state,
            "ppa": ppa,
            "action": -1,
            "ppa_dict": {
                "area": 0,
                "delay": 0,
            },
            "ppa_list": [],
        }

        for iteration in range(self.max_iter):
            trajectory = {
                "info": {
                    "episode_num": iteration,
                },
                "data": [],
            }
            action_mask = mask_with_legality(state, self.bit_width, self.encode_type, self.MAX_STAGE_NUM)
            legal_action_list = np.where(action_mask)[0]
            if len(legal_action_list) < self.search_steps:
                selected_action_list = legal_action_list
            else:
                selected_action_list = np.random.choice(legal_action_list, self.search_steps, replace=False)
            param_list = [
                (
                    self.build_path,
                    state,
                    selected_action_list[i],
                    self.bit_width,
                    self.encode_type,
                    i,
                    self.target_delay,
                    self.area_scale,
                    self.delay_scale,
                    self.ppa_scale,
                    self.weight,
                ) for i in range(len(selected_action_list))
            ]
            with multiprocessing.Pool(self.n_processing) as pool:
                result = pool.starmap_async(self.get_ppa, param_list)
                pool.close()
                pool.join()
            result = result.get()
            sorted_indices = sorted(list(range(len(result))), key=lambda x: result[x]["ppa"])
            if result[sorted_indices[0]]["ppa"] < ppa:
                # 如果最小的 ppa 小于当前的 ppa
                ppa = result[sorted_indices[0]]["ppa"]
                state = result[sorted_indices[0]]["state"]
                info = result[sorted_indices[0]]
            else:
                # 计算转移概率
                candidate_state = [copy.deepcopy(state)]
                candidate_ppa = [ppa]
                for item in result:
                    candidate_state.append(item["state"])
                    candidate_ppa.append(item["ppa"])
                candidate_ppa = np.asarray(candidate_ppa)
                delta_ppa = candidate_ppa - ppa
                p = np.exp( - delta_ppa / T)
                p = p / np.sum(p)
                selected_index = np.random.choice(len(candidate_state), p=p)
                state = candidate_state[selected_index]
                if selected_index == 0:
                    ppa = ppa
                else:
                    info = result[selected_index - 1]
                    ppa = result[selected_index - 1]["ppa"]
            # 完成一轮
            for item in result:
                item["state"] = np.asarray(item["state"]).astype(int).tolist()
            trajectory["data"] = result
            self.trajectory_list.append(trajectory)
            
            if ppa < self.found_best_info["found_best_ppa"]:
                self.found_best_info["found_best_ppa"] = ppa
                self.found_best_info["found_best_state"] = state
                self.found_best_info["found_best_area"] = info["ppa_dict"]["area"]
                self.found_best_info["found_best_delay"] = info["ppa_dict"]["delay"]

            logger.tb_logger.add_histogram("candidate_ppa", np.asarray([item["ppa"] for item in result]), global_step=iteration)
            logger.tb_logger.add_scalar("selected_action_len", (len(selected_action_list)), global_step=iteration)
            logger.tb_logger.add_scalar("legal_action_len", (len(legal_action_list)), global_step=iteration)
            logger.tb_logger.add_scalar("action", info["action"], global_step=iteration)
            logger.tb_logger.add_scalar("ppa", ppa, global_step=iteration)
            logger.tb_logger.add_scalar("area", info["ppa_dict"]["area"], global_step=iteration)
            logger.tb_logger.add_scalar("delay", info["ppa_dict"]["delay"], global_step=iteration)

            logger.tb_logger.add_scalar("found_best_ppa", self.found_best_info["found_best_ppa"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_area", self.found_best_info["found_best_area"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_delay", self.found_best_info["found_best_delay"], global_step=iteration)

            logger.tb_logger.add_scalar("T", T, global_step=iteration)

            if iteration > 0 and iteration % self.end_exp_freq == 0:
               self.end_experiments(iteration)
            
            T = T * self.T_dec_factor

        self.end_experiments(iteration)
    
    def store_trajectory(self, episode_num):
        trajectory_path = os.path.join(
            logger._snapshot_dir, "trajectory", f"episode_{episode_num}.json"
        )
        if not os.path.exists(os.path.dirname(trajectory_path)):
            os.makedirs(os.path.dirname(trajectory_path))
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # 将 ndarray 转换为列表
            else:
                raise TypeError(f"Type {type(obj)} not serializable")
        with open(trajectory_path, "w") as file:
            json.dump(self.trajectory_list, file, default=convert_numpy)
        pass
    
    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        save_data_dict["found_best_info"] = self.found_best_info
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        best_state = copy.deepcopy(self.found_best_info["found_best_state"])
        ppas_dict = self.get_ppa_full_delay_cons(best_state)
        save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(episode_num, save_data_dict)
        self.store_trajectory(episode_num)

    def get_ppa_full_delay_cons(self, test_state):
        # generate target delay
        target_delay=[]
        input_width = self.bit_width
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
        n_processing = 12
        # config_abc_sta

        build_path = os.path.join(self.build_path, f"ppa_full_delay")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, test_state[0], test_state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, self.bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay, build_path, False, False, False, False, False, False, n_processing)
        worker.evaluate()

        return worker.consult_ppa_list()
    
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


class AntColonyOptimization:
    def __init__(
        self,
        bit_width=8,
        encode_type="and",
        init_ct_type="wallace",
        area_scale=438.5,
        delay_scale=0.7499,
        ppa_scale=100,
        weight=[4, 1],
        reference_point=[2600, 1.8],
        max_iter=20,
        end_exp_freq=25,
        build_path="build",
        target_delay=[50,250,400,650],
        n_processing=4,
        # ACO
        initial_pheromone=1.0,
        rho=0.05,
        Q=50,
        max_compressor_num=10,
        alpha=1,
        beta=0.1,
    ):
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.init_ct_type = init_ct_type
        self.target_delay = target_delay

        self.area_scale = area_scale
        self.delay_scale = delay_scale
        self.ppa_scale = ppa_scale
        self.weight = weight

        self.n_processing = n_processing
        self.end_exp_freq = end_exp_freq


        # total steps
        self.total_steps = 0
        self.total_actual_steps = 0
        self.initial_partial_product = get_initial_partial_product(
            bit_width, encode_type
        )

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
        }

        # pareto pointset
        self.pareto_pointset = {"area": [], "delay": [], "state": []}
        self.reference_point = reference_point

        self.max_iter = max_iter

        # ACO
        self.initial_pheromone = initial_pheromone
        self.rho = rho
        self.Q = Q
        self.max_compressor_num = max_compressor_num
        self.alpha = alpha
        self.beta = beta
        
        self.initial_cwd_path = os.getcwd()
        self.build_path = os.path.join(self.initial_cwd_path, build_path)
        if not os.path.exists(self.build_path):
            os.mkdir(self.build_path)

        # configure figure
        plt.switch_backend("agg")

    # fmt: on
    @staticmethod
    def get_ppa(build_path_base, state, action, bit_width, encode_type, worker_id, target_delay_list, area_scale, delay_scale, ppa_scale, weight):
        cur_state = copy.deepcopy(state)
        next_state = step(cur_state, action, bit_width, encode_type)
        build_path = os.path.join(build_path_base, f"worker_{worker_id}")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(bit_width, encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, next_state[0], next_state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay_list, build_path, False, False, False, False, False, False, 1)
        worker.evaluate()

        ppa_dict = worker.consult_ppa()
        ppa_list = worker.consult_ppa_list()

        ppa = ppa_scale * (weight[0] * ppa_dict["area"] / area_scale + weight[1] * ppa_dict["delay"] / delay_scale)

        return {
            "state": next_state,
            "ppa": ppa,
            "action": action,
            "ppa_dict": ppa_dict,
            "ppa_list": ppa_list,
        }

    def run_experiments(self):
        self.trajectory_list = []
        # 构造信息素矩阵
        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        pheromone = np.full([2 * len(pp), self.max_compressor_num], self.initial_pheromone)
        heuristic = np.full([2 * len(pp), self.max_compressor_num], 1/500)
        for iteration in range(self.max_iter):
            trajectory = {
                "info": {
                    "episode_num": iteration,
                },
                "data": [],
            }
            # step1 ConstructAntSolutions
            ct = np.zeros([2 * len(pp)])
            for column_index in range(2 * len(pp)):
               tau = pheromone[column_index]
               eta = heuristic[column_index]
               p = tau ** self.alpha * eta ** self.beta
               p = p / np.sum(p)
               sampled_value = np.random.choice(list(range(self.max_compressor_num)), p=p)
               ct[column_index] = sampled_value
            ct = ct.reshape([2, -1])
            ct32, ct22 = legalize_compressor_tree(pp, ct[0], ct[1])
            legalized_ct = np.asarray([ct32, ct22])

            # step2 仿真
            build_path = os.path.join(self.build_path, f"build")
            if not os.path.exists(build_path):
                os.makedirs(build_path)
            rtl_path = os.path.join(build_path, "MUL.v")
            env = SpeedUpRefineEnv_Conventional()
            dec_32, dec_22, _, __ = decompose_compressor_tree(pp, legalized_ct[0], legalized_ct[1])
            env.write_mul(rtl_path, self.bit_width, np.asarray([dec_32, dec_22]))
            worker = EvaluateWorker(rtl_path, ["ppa"], self.target_delay, build_path, False, False, False, False, False, False, self.n_processing)
            worker.evaluate()
            ppa_dict = worker.consult_ppa()
            ppa = self.ppa_scale * (
                self.weight[0] * ppa_dict["area"] / self.area_scale
                + self.weight[1] * ppa_dict["delay"] / self.delay_scale
            )

            trajectory["data"].append(
                    {
                        "state": np.asarray(legalized_ct).tolist(),
                        "rewards_dict": ppa_dict,
                        "rewards_list": worker.consult_ppa_list(),
                        "ppa": ppa,
                    }
                )
            self.trajectory_list.append(trajectory)
            # step2 UpdatePheromones
            legalized_ct = legalized_ct.flatten().astype(int)
            pheromone = (1 - self.rho) * pheromone # 首先是信息素会衰减
            for column_index in range(2 * len(pp)):
                # 然后是信息素会更新
                pheromone[column_index][legalized_ct[column_index]] += self.Q / ppa
            for column_index in range(2 * len(pp)):
                # 最后是更新一下启发式信息
                heuristic[column_index][legalized_ct[column_index]] = 1 / ppa

            # 完成一轮
            if ppa < self.found_best_info["found_best_ppa"]:
                self.found_best_info["found_best_ppa"] = ppa
                self.found_best_info["found_best_state"] = legalized_ct.reshape([2, -1])
                self.found_best_info["found_best_area"] = ppa_dict["area"]
                self.found_best_info["found_best_delay"] = ppa_dict["delay"]
            
            logger.tb_logger.add_histogram("pheromone", pheromone.flatten(), global_step=iteration)
            fig = plt.figure()
            ax = fig.add_subplot(111)  # 添加子图
            im = ax.imshow(pheromone, aspect='auto')  # 确保绘制在子图上
            fig.colorbar(im)  # 添加颜色条
            logger.tb_logger.add_figure("pheromone", fig, global_step=iteration)

            logger.tb_logger.add_scalar("ppa", ppa, global_step=iteration)
            logger.tb_logger.add_scalar("area", ppa_dict["area"], global_step=iteration)
            logger.tb_logger.add_scalar("delay", ppa_dict["delay"], global_step=iteration)

            logger.tb_logger.add_scalar("found_best_ppa", self.found_best_info["found_best_ppa"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_area", self.found_best_info["found_best_area"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_delay", self.found_best_info["found_best_delay"], global_step=iteration)

            if iteration > 0 and iteration % self.end_exp_freq == 0:
               self.end_experiments(iteration)
        self.end_experiments(iteration)

    def store_trajectory(self, episode_num):
        trajectory_path = os.path.join(
            logger._snapshot_dir, "trajectory", f"episode_{episode_num}.json"
        )
        if not os.path.exists(os.path.dirname(trajectory_path)):
            os.makedirs(os.path.dirname(trajectory_path))
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # 将 ndarray 转换为列表
            else:
                raise TypeError(f"Type {type(obj)} not serializable")
        with open(trajectory_path, "w") as file:
            json.dump(self.trajectory_list, file, default=convert_numpy)
        pass

    def end_experiments(self, episode_num):
        # save datasets 
        save_data_dict = {}
        # replay memory
        save_data_dict["found_best_info"] = self.found_best_info
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        best_state = copy.deepcopy(self.found_best_info["found_best_state"])
        ppas_dict = self.get_ppa_full_delay_cons(best_state)
        save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(episode_num, save_data_dict)
        self.store_trajectory(episode_num)

    def get_ppa_full_delay_cons(self, test_state):
        # generate target delay
        target_delay=[]
        input_width = self.bit_width
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
        n_processing = 12
        # config_abc_sta

        build_path = os.path.join(self.build_path, f"ppa_full_delay")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, test_state[0], test_state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, self.bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay, build_path, False, False, False, False, False, False, n_processing)
        worker.evaluate()

        return worker.consult_ppa_list()
    
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
        try:
            hv = hypervolume(combine_array)
            hv_value = hv.compute(self.reference_point)
        except:
            hv_value = 0.0
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


class ConventionalBase:
    def __init__(
        self,
        bit_width=8,
        encode_type="and",
        init_ct_type="wallace",
        area_scale=438.5,
        delay_scale=0.7499,
        ppa_scale=100,
        weight=[4, 1],
        reference_point=[2600, 1.8],
        max_iter=20,
        end_exp_freq=25,
        build_path="build",
        target_delay=[50,250,400,650],
        n_processing=4,
    ):
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.init_ct_type = init_ct_type
        self.target_delay = target_delay

        self.area_scale = area_scale
        self.delay_scale = delay_scale
        self.ppa_scale = ppa_scale
        self.weight = weight

        self.n_processing = n_processing
        self.end_exp_freq = end_exp_freq


        # total steps
        self.total_steps = 0
        self.total_actual_steps = 0
        self.initial_partial_product = get_initial_partial_product(
            bit_width, encode_type
        )

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
        }

        self.trajectory_list = []

        # pareto pointset
        self.pareto_pointset = {"area": [], "delay": [], "state": []}
        self.reference_point = reference_point

        self.max_iter = max_iter

        self.initial_cwd_path = os.getcwd()
        self.build_path = os.path.join(self.initial_cwd_path, build_path)
        if not os.path.exists(self.build_path):
            os.mkdir(self.build_path)

        # configure figure
        plt.switch_backend("agg")

    def run_experiments(self):
        raise NotImplementedError

    def store_trajectory(self, episode_num):
        trajectory_path = os.path.join(
            logger._snapshot_dir, "trajectory", f"episode_{episode_num}.json"
        )
        if not os.path.exists(os.path.dirname(trajectory_path)):
            os.makedirs(os.path.dirname(trajectory_path))
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # 将 ndarray 转换为列表
            else:
                raise TypeError(f"Type {type(obj)} not serializable")
        with open(trajectory_path, "w") as file:
            json.dump(self.trajectory_list, file, default=convert_numpy)
        pass

    def end_experiments(self, episode_num):
            # save datasets 
            save_data_dict = {}
            # replay memory
            save_data_dict["found_best_info"] = self.found_best_info
            # test to get full pareto points
            # input: found_best_info state
            # output: testing pareto points and hypervolume
            best_state = copy.deepcopy(self.found_best_info["found_best_state"])
            ppas_dict = self.get_ppa_full_delay_cons(best_state)
            save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
            save_data_dict["testing_pareto_data"] = save_pareto_data_dict
            logger.save_npy(episode_num, save_data_dict)
            self.store_trajectory(episode_num)

    def get_ppa_full_delay_cons(self, test_state):
        # generate target delay
        target_delay=[]
        input_width = self.bit_width
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
        n_processing = 12
        # config_abc_sta

        build_path = os.path.join(self.build_path, f"ppa_full_delay")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, test_state[0], test_state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, self.bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay, build_path, False, False, False, False, False, False, n_processing)
        worker.evaluate()

        return worker.consult_ppa_list()
    
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
        try:
            hv = hypervolume(combine_array)
            hv_value = hv.compute(self.reference_point)
        except:
            hv_value = 0.0
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


import heapq
import random


class LimitedMaxHeap:
    """
    最大堆
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.heap = []  # 使用最大堆存储元素 (-priority, item)

    def push(self, data):
        priority, item = data
        priority = -priority
        if len(self.heap) < self.capacity:
            # 队列未满，直接插入
            heapq.heappush(self.heap, (priority, item))
        else:
            # 队列已满，检查新元素的优先级是否高于最低优先级
            lowest_priority = self.heap[0][0]
            if priority > lowest_priority:
                # 新元素优先级更高，替换最低优先级的元素
                heapq.heapreplace(self.heap, (priority, item))
            else:
                # 新元素优先级不够高，忽略
                pass
    
    def sample(self):
        """
        从队列中随机抽取一个元素（等概率）。
        :return: item
        """
        if not self.heap:
            return None
        priority, item = random.choice(self.heap)
        return item


    def pop(self):
        # 弹出优先级最低的元素
        if self.queue:
            return heapq.heappop(self.queue)
        else:
            return None
class PriorityRandom(ConventionalBase):
    def __init__(
            self,
            pool_size=20,
            sample_action_num=4,
            sample_action_level=2,
            max_stage_num=5,
            sample_path=None,
            **policy_kwargs,
        ):
        super().__init__(**policy_kwargs)
        self.pool_size = pool_size
        self.priority_pool = LimitedMaxHeap(capacity=pool_size)
        self.pool = []
        self.sample_action_num = []
        self.sample_action_level = []

        self.sample_action_num = sample_action_num
        self.sample_action_level = sample_action_level
        self.max_stage_num = max_stage_num
        self.sample_path = sample_path

        self.__load_pool_from_json()
    
    def tree_sample_from_state(self, state, level):
        if level == 0:
            return []
        else:
            action_mask = mask_with_legality(state, self.bit_width, self.encode_type, self.max_stage_num)
            legal_action_list = np.where(action_mask)[0]
            sampled_action_list = np.random.choice(legal_action_list, size=self.sample_action_num, replace=False)

            sampled_state_list = []
            for action in sampled_action_list:
                next_state = step(state, action, self.bit_width, self.encode_type)
                sampled_state_list.append(next_state)
            sampled_next_level_list = []
            for next_state in sampled_state_list:
                next_level = self.tree_sample_from_state(next_state, level - 1)
                sampled_next_level_list = sampled_next_level_list + next_level

            sampled_state_list = sampled_state_list + sampled_next_level_list
            return sampled_state_list

    def __load_pool_from_json(self):
        for file_path in self.sample_path:
            project_dir = os.path.dirname(__file__)
            refined_file_path = os.path.join(project_dir, file_path)
            with open(refined_file_path, "r") as file:
                data = json.load(file)
            for trajectory in data:
                for item in trajectory["data"]:
                    state = item["state"]
                    ppa = item["rewards_dict"]["avg_ppa"]
                    self.priority_pool.push((ppa, state))
    
    def end_experiments(self, episode_num):
        sample_path = os.path.join(
            logger._snapshot_dir, "sample", f"episode_{episode_num}.json"
        )
        if not os.path.exists(os.path.dirname(sample_path)):
            os.makedirs(os.path.dirname(sample_path))
        with open(sample_path, "w") as file:
            json.dump(self.pool, file)
    
    @staticmethod
    def get_ppa(build_path_base, state, bit_width, encode_type, target_delay_list, area_scale, delay_scale, ppa_scale, weight, worker_id):
        build_path = os.path.join(build_path_base, f"worker_{worker_id}")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(bit_width, encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, state[0], state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay_list, build_path, False, False, False, False, False, False, 1)
        worker.evaluate()

        ppa_dict = worker.consult_ppa()
        ppa = ppa_scale * (weight[0] * ppa_dict["area"] / area_scale + weight[1] * ppa_dict["delay"] / delay_scale)

        return (ppa, np.asarray(state).astype(int).tolist())

    def run_experiments(self):
        for iteration in range(self.max_iter):
            state = self.priority_pool.sample()
            sampled_state_list = self.tree_sample_from_state(state, self.sample_action_level)
            params_list = [
                (
                    self.build_path,
                    sampled_state_list[index],
                    self.bit_width,
                    self.encode_type,
                    self.target_delay,
                    self.area_scale,
                    self.delay_scale,
                    self.ppa_scale,
                    self.weight,
                    index,
                ) for index in range(len(sampled_state_list))
            ]
            with multiprocessing.Pool(self.n_processing) as pool:
                result = pool.starmap_async(self.get_ppa, params_list)
                pool.close()
                pool.join()
            result = result.get()

            # 更新 pool
            for item in result:
                self.priority_pool.push(item)
                self.pool.append(item)
            
            sampled_ppa_array = np.asarray([item[0] for item in result])
            pool_ppa_array = np.asarray([- item[0] for item in self.priority_pool.heap])
            logger.tb_logger.add_scalar("sampled_num", len(self.pool), global_step=iteration)
            logger.tb_logger.add_histogram("sampled_ppa", sampled_ppa_array, global_step=iteration)
            logger.tb_logger.add_histogram("pool_ppa", pool_ppa_array, global_step=iteration)
            if iteration > 0 and iteration % self.end_exp_freq == 0:
                self.end_experiments(iteration)
        self.end_experiments(iteration)


class BayesianOptimization(ConventionalBase):
    def __init__(
            self,
            kappa=2.0,
            kernel_constant_value=1.0,
            kernel_constant_bounds = [1e-2, 1e2],
            kernel_length_scale = 1.0,
            n_restarts_optimizer = 10,
            length_scale_bounds = [1e-2, 1e2],
            alpha = 1e-6,
            beta = 2.0,
            max_stage_num=6,
            n_restarts=10,
            batch_size=64,
            normalized_bound=3,
            trajectory_path="None",
            **policy_kwargs,
        ):
        super().__init__(**policy_kwargs)
        self.kappa = kappa
        self.beta = beta
        self.max_stage_num = max_stage_num
        self.n_restarts = n_restarts
        self.batch_size = batch_size
        self.normalized_bound = normalized_bound

        self.pool = []

        kernel = ConstantKernel(
            kernel_constant_value,
            kernel_constant_bounds
        ) * RBF(
            length_scale=kernel_length_scale,
            length_scale_bounds=length_scale_bounds)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)

        self.bounds = np.asarray([[-normalized_bound, normalized_bound] for _ in range(2 * len(self.initial_partial_product))])

        if trajectory_path is not None and trajectory_path != "None" and trajectory_path != "none":
            self.trajectory_path = trajectory_path
            self.__load_pool_from_json()


    def __load_pool_from_json(self):
        if type(self.trajectory_path) == list:
            for file_path in self.trajectory_path:
                project_dir = os.path.dirname(__file__)
                refined_file_path = os.path.join(project_dir, file_path)
                with open(refined_file_path, "r") as file:
                    data = json.load(file)
                for trajectory in data:
                    for item in trajectory["data"]:
                        state = item["state"]
                        ppa = item["rewards_dict"]["avg_ppa"]
                        self.pool.append((ppa, state))
        else:
            project_dir = os.path.dirname(__file__)
            refined_file_path = os.path.join(project_dir, self.trajectory_path)
            with open(refined_file_path, "r") as file:
                data = json.load(file)
            for trajectory in data:
                for item in trajectory["data"]:
                    state = item["state"]
                    try:
                        ppa = item["rewards_dict"]["avg_ppa"]
                    except:
                        ppa = item["ppa"]
                    self.pool.append((ppa, state))


    def ucb(self, X):
        """UCB 采集函数
        :param X: 输入点（二维数组）
        :param gp: 拟合好的高斯过程模型
        :param kappa: 权衡探索和利用的超参数
        :return: UCB 值
        """
        X = X.reshape(-1, self.gp.X_train_.shape[1])  # 确保输入维度一致
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu - self.kappa * sigma


    def propose_location(self):
        """
        Propose the next evaluation point by optimizing the acquisition function (UCB).
        """
        dim = self.bounds.shape[0]
        best_ucb = float('inf')
        best_x = None

        for _ in range(self.n_restarts):
            x_start = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(dim,))
            res = minimize(lambda x: self.ucb(x), x_start, bounds=self.bounds, method='L-BFGS-B')

            if res.fun < best_ucb:
                best_ucb = res.fun
                best_x = res.x

        mu, sigma = self.gp.predict(best_x.reshape(1, -1), return_std=True)
        return best_x, mu, sigma

    def get_ppa(self, state):
        ct32, ct22, _, __ =  decompose_compressor_tree(self.initial_partial_product, state[0], state[1])
        build_path = os.path.join(self.build_path, "build")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, self.bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], self.target_delay, build_path, False, False, False, False, False, False, self.n_processing)
        worker.evaluate()
        ppa_dict = worker.consult_ppa()
        ppa = self.ppa_scale * (self.weight[0] * ppa_dict["area"] / self.area_scale + self.weight[1] * ppa_dict["delay"] / self.delay_scale)

        return ppa, ppa_dict

    def run_experiments(self):
        state = get_compressor_tree(self.initial_partial_product, self.bit_width, self.init_ct_type)

        for iteration in range(self.max_iter):
            if len(self.pool) < self.batch_size:
                # pool 中的还不够 batch_size: 随机游走
                action_mask = mask_with_legality(state, self.bit_width, self.encode_type, self.max_stage_num)
                legal_actions = np.where(action_mask)[0]
                action = np.random.choice(legal_actions)
                next_state = step(state, action, self.bit_width, self.encode_type)
                use_model_flag = False
            else:
                # 随机采样 batch_size 个点
                # batch = random.sample(self.pool, self.batch_size)
                # y_train = np.asarray([item[0] / (self.ppa_scale * (self.weight[0] + self.weight[1])) for item in batch])
                # x_train = np.asarray([np.asarray(item[1]).astype(float).flatten() / self.max_compressor_num for item in batch]) # x 做一下归一化

                overall_ppa = np.asarray([item[0] for item in self.pool])
                p = np.max(overall_ppa) - overall_ppa
                p = p + 1e-2
                p = p / np.sum(p)
                batch_indices = np.random.choice(list(range(len(self.pool))), self.batch_size, replace=False, p=p).astype(int)

                y_train = np.asarray([self.pool[index][0] / (self.ppa_scale * (self.weight[0] + self.weight[1])) for index in batch_indices])

                x_train = np.asarray([np.asarray(self.pool[index][1]).astype(float).flatten() for index in batch_indices])
                column_width = len(self.initial_partial_product)

                ct_32_mean = np.mean(x_train[:, :column_width], axis=0)
                ct_32_var = np.var(x_train[:, :column_width], axis=0)

                ct_22_mean = np.mean(x_train[:, column_width:], axis=0)
                ct_22_var = np.var(x_train[:, column_width: ], axis=0)

                x_train[:, :column_width] = (x_train[:, :column_width] - ct_32_mean) / (ct_32_var + 1e-5)
                x_train[:, column_width:] = (x_train[:, column_width:] - ct_22_mean) / (ct_22_var + 1e-5)

                x_train = np.clip(x_train, - self.normalized_bound, self.normalized_bound)
                self.gp.fit(x_train, y_train)

                next_state, mu, sigma = self.propose_location()
                # next_state = (np.asarray(next_state) * self.max_compressor_num).astype(int).reshape([2, -1])

                next_state = np.asarray(next_state).flatten()
                next_state[:column_width] = next_state[:column_width] * ct_32_var + ct_32_mean
                next_state[column_width:] = next_state[column_width:] * ct_22_var + ct_22_mean

                next_state = next_state.astype(int).reshape([2, -1])

                next_state = legalize_compressor_tree(self.initial_partial_product, next_state[0], next_state[1])
                use_model_flag = True

            # 评估 next_state
            ppa, ppa_dict = self.get_ppa(next_state)
            self.pool.append((ppa, next_state))

            # 结束一轮循环
            if self.found_best_info["found_best_ppa"] > ppa:
                self.found_best_info["found_best_ppa"] = ppa
                self.found_best_info["found_best_state"] = state
                self.found_best_info["found_best_area"] = ppa_dict["area"]
                self.found_best_info["found_best_delay"] = ppa_dict["delay"]
            
            logger.tb_logger.add_scalar("ppa", ppa, global_step=iteration)
            logger.tb_logger.add_scalar("area", ppa_dict["area"], global_step=iteration)
            logger.tb_logger.add_scalar("delay", ppa_dict["delay"], global_step=iteration)

            logger.tb_logger.add_scalar("found_best_ppa", self.found_best_info["found_best_ppa"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_area", self.found_best_info["found_best_area"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_delay", self.found_best_info["found_best_delay"], global_step=iteration)

            if use_model_flag:
                logger.tb_logger.add_scalar("mean - true", mu - ppa / (self.ppa_scale * (self.weight[0] + self.weight[1])), global_step=iteration)
                logger.tb_logger.add_scalar("sigma", sigma, global_step=iteration)

            state = next_state

            if iteration > 0 and iteration % self.end_exp_freq == 0:
                self.end_experiments(iteration)
        self.end_experiments(iteration)

    def end_experiments(self, episode_num):
            # save datasets 
            save_data_dict = {}
            # replay memory
            save_data_dict["found_best_info"] = self.found_best_info
            # test to get full pareto points
            # input: found_best_info state
            # output: testing pareto points and hypervolume
            best_state = copy.deepcopy(self.found_best_info["found_best_state"])
            ppas_dict = self.get_ppa_full_delay_cons(best_state)
            save_pareto_data_dict = self.log_and_save_pareto_points(ppas_dict, episode_num)
            save_data_dict["testing_pareto_data"] = save_pareto_data_dict
            logger.save_npy(episode_num, save_data_dict)


class SimulatedAnnealing_v2(SimulatedAnnealing_v1):
    def __init__(
            self,
            step_scale=0.1,
            time_out = 20,
            **policy_kwargs,
        ):
        super().__init__(**policy_kwargs)
        self.step_scale = step_scale
        self.time_out = time_out

    @staticmethod
    def get_ppa(build_path_base, state, bit_width, encode_type, worker_id, target_delay_list, area_scale, delay_scale, ppa_scale, weight):
        build_path = os.path.join(build_path_base, f"worker_{worker_id}")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        rtl_path = os.path.join(build_path, "MUL.v")

        pp = get_initial_partial_product(bit_width, encode_type)
        ct32, ct22, _, __ =  decompose_compressor_tree(pp, state[0], state[1])
        ct = np.asarray([ct32, ct22])
        env = SpeedUpRefineEnv_Conventional()
        env.write_mul(rtl_path, bit_width, ct)
        worker = EvaluateWorker(rtl_path, ["ppa"], target_delay_list, build_path, False, False, False, False, False, False, 1)
        worker.evaluate()

        ppa_dict = worker.consult_ppa()
        ppa_list = worker.consult_ppa_list()

        ppa = ppa_scale * (weight[0] * ppa_dict["area"] / area_scale + weight[1] * ppa_dict["delay"] / delay_scale)

        return {
            "state": state,
            "ppa": ppa,
            "ppa_dict": ppa_dict,
            "ppa_list": ppa_list,
        }
    
    def run_experiments(self):
        self.trajectory_list = []
        
        T = self.T0
        state = get_compressor_tree(
            self.initial_partial_product, self.bit_width, self.init_ct_type
        )
        self.found_best_info["found_best_state"] = state
        ppa = 500
        info = {
            "state": state,
            "ppa": ppa,
            "action": -1,
            "ppa_dict": {
                "area": 0,
                "delay": 0,
            },
            "ppa_list": [],
        }

        for iteration in range(self.max_iter):
            trajectory = {
                "info": {
                    "episode_num": iteration,
                },
                "data": [],
            }
            cur_state = np.asarray(state).flatten()
            next_state_list = []
            counter = 0
            try_counter = 0
            while counter < 1 or (counter < self.search_steps and try_counter < self.time_out):
                try_counter += 1
                delta_state = np.random.normal(0, self.step_scale, cur_state.size)
                next_state = (np.round(cur_state.astype(float) + delta_state)).astype(int)
                if not (next_state == cur_state.astype(int)).all():
                    next_state = next_state.reshape([2, -1])
                    next_state = legalize_compressor_tree(self.initial_partial_product, next_state[0], next_state[1])
                    next_state_list.append(next_state)
                    counter += 1
            param_list = [
                (
                    self.build_path,
                    next_state_list[i],
                    self.bit_width,
                    self.encode_type,
                    i,
                    self.target_delay,
                    self.area_scale,
                    self.delay_scale,
                    self.ppa_scale,
                    self.weight,
                ) for i in range(len(next_state_list))
            ]
            with multiprocessing.Pool(self.n_processing) as pool:
                result = pool.starmap_async(self.get_ppa, param_list)
                pool.close()
                pool.join()
            result = result.get()
            sorted_indices = sorted(list(range(len(result))), key=lambda x: result[x]["ppa"])
            if result[sorted_indices[0]]["ppa"] < ppa:
                # 如果最小的 ppa 小于当前的 ppa
                ppa = result[sorted_indices[0]]["ppa"]
                state = result[sorted_indices[0]]["state"]
                info = result[sorted_indices[0]]
            else:
                # 计算转移概率
                candidate_state = [copy.deepcopy(state)]
                candidate_ppa = [ppa]
                for item in result:
                    candidate_state.append(item["state"])
                    candidate_ppa.append(item["ppa"])
                candidate_ppa = np.asarray(candidate_ppa)
                delta_ppa = candidate_ppa - ppa
                p = np.exp( - delta_ppa / T)
                p = p / np.sum(p)
                selected_index = np.random.choice(len(candidate_state), p=p)
                state = candidate_state[selected_index]
                if selected_index == 0:
                    ppa = ppa
                else:
                    info = result[selected_index - 1]
                    ppa = result[selected_index - 1]["ppa"]
            # 完成一轮
            for item in result:
                item["state"] = np.asarray(item["state"]).astype(int).tolist()
            trajectory["data"] = result
            self.trajectory_list.append(trajectory)
            
            if ppa < self.found_best_info["found_best_ppa"]:
                self.found_best_info["found_best_ppa"] = ppa
                self.found_best_info["found_best_state"] = state
                self.found_best_info["found_best_area"] = info["ppa_dict"]["area"]
                self.found_best_info["found_best_delay"] = info["ppa_dict"]["delay"]

            logger.tb_logger.add_scalar("try_counter", try_counter, global_step=iteration)
            logger.tb_logger.add_histogram("candidate_ppa", np.asarray([item["ppa"] for item in result]), global_step=iteration)
            logger.tb_logger.add_scalar("state_len", (len(next_state_list)), global_step=iteration)
            logger.tb_logger.add_scalar("ppa", ppa, global_step=iteration)
            logger.tb_logger.add_scalar("area", info["ppa_dict"]["area"], global_step=iteration)
            logger.tb_logger.add_scalar("delay", info["ppa_dict"]["delay"], global_step=iteration)

            logger.tb_logger.add_scalar("found_best_ppa", self.found_best_info["found_best_ppa"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_area", self.found_best_info["found_best_area"], global_step=iteration)
            logger.tb_logger.add_scalar("found_best_delay", self.found_best_info["found_best_delay"], global_step=iteration)

            logger.tb_logger.add_scalar("T", T, global_step=iteration)

            if iteration > 0 and iteration % self.end_exp_freq == 0:
               self.end_experiments(iteration)
            
            T = T * self.T_dec_factor

        self.end_experiments(iteration)
    


if __name__ == "__main__":
    sa = SimulatedAnnealing_v1()
    sa.run_experiments()
