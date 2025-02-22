import copy
import random
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque, Counter
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
from pygmo import hypervolume
import math
import sys
import curses

import sys
sys.path.append("..")
from o1_environment_adder import RefineEnv, State
from o0_logger import logger
from o2_policy_adder import DeepQPolicy

Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "next_state",
        "reward",
        "mask",
        "next_state_mask",
        # "rewards_dict",
    ),
)


class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        optimizer_class="Adam",
        q_net_lr=1e-4,
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
        action_num=4,
        # pareto
        reference_point=[2600, 1.8],
        # multiobj
        multiobj_type="pure_max",  # [pure_max, weight_max]
        # store type
        store_type="simple",  # simple or detail
        # end_exp_log
        end_exp_freq=25,
        initial_adder_type=0,
    ):
        self.initial_adder_type = initial_adder_type
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
        self.action_num = action_num
        self.multiobj_type = multiobj_type
        self.end_exp_freq = end_exp_freq
        # optimizer
        # TODO: lr, lrdecay, gamma
        # TODO: double q
        self.q_net_lr = q_net_lr
        if isinstance(optimizer_class, str):
            optimizer_class = eval("optim." + optimizer_class)
            self.optimizer_class = optimizer_class
        self.policy_optimizer = optimizer_class(
            self.q_policy.parameters(), lr=self.q_net_lr
        )

        # loss function
        self.loss_fn = nn.SmoothL1Loss()

        # total steps
        self.total_steps = 0
        self.int_bit_width = env.bit_width

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
        }
        # agent reset
        self.agent_reset_freq = agent_reset_freq
        self.agent_reset_type = agent_reset_type

        self.deterministic = deterministic
        self.is_softmax = is_softmax

        # # pareto pointset
        self.pareto_pointset = {"area": [], "delay": [], "state": []}
        self.reference_point = reference_point
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
    ):
        self.replay_memory.push(
            # torch.tensor(state),
            state,
            action,
            # torch.tensor(next_state),
            next_state,
            torch.tensor([reward]),
            mask.reshape(1, -1),
            next_state_mask.reshape(1, -1),
        )

    def store_detail(
        self,
        state,
        next_state,
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
            torch.tensor(state),
            action,
            torch.tensor(next_state),
            torch.tensor([reward]),
            mask.reshape(1, -1),
            next_state_mask.reshape(1, -1),
            state_ct32,
            state_ct22,
            next_state_ct32,
            next_state_ct22,
            rewards_dict,
        )

    # parallel version
    def compute_values(self, state_batch, action_batch, state_mask, is_average=False):
        batch_size = len(state_batch)
        state_action_values = torch.zeros(batch_size, device=self.device)
        states = []
        for i in range(batch_size):
            state = torch.tensor(state_batch[i].cell_map).unsqueeze(0)
            states.append(state)
        states = torch.stack(states)
        if action_batch is not None:
            q_values = self.q_policy(
                state_batch,
                tensor_x=states,
                state_mask=state_mask
            )
            for i in range(batch_size):
                state_action_values[i] = q_values[i, action_batch[i]]
        else:
            q_values = self.target_q_policy(
                state_batch,
                tensor_x=states,
                state_mask=state_mask
            )
            for i in range(batch_size):
                state_action_values[i] = q_values[i:i+1].max(1)[0].detach()
        return state_action_values

    # serial version
    # def compute_values(self, state_batch, action_batch, state_mask, is_average=False):
    #     batch_size = len(state_batch)
    #     state_action_values = torch.zeros(batch_size, device=self.device)
    #     for i in range(batch_size):
    #         # compute image state
    #         state = state_batch[i]
    #         # compute image state
    #         if action_batch is not None:
    #             # reshape 有问题************
    #             q_values = self.q_policy(state, state_mask=state_mask[i]).reshape(
    #                 (int(self.int_bit_width**2)) * 2
    #             )
    #             state_action_values[i] = q_values[action_batch[i]]
    #         else:
    #             q_values = self.target_q_policy(
    #                 state, is_target=True, state_mask=state_mask[i]
    #             )
    #             if self.is_double_q:
    #                 current_q_values = self.q_policy(
    #                     state, state_mask=state_mask[i]
    #                 ).reshape((int(self.int_bit_width**2)) * 2)
    #                 index = torch.argmax(current_q_values)
    #                 state_action_values[i] = q_values.squeeze()[index].detach()
    #             else:
    #                 state_action_values[i] = q_values.max(1)[0].detach()
    #     return state_action_values

    def update_q(self):
        if len(self.replay_memory) < self.batch_size:
            loss = 0.0
            info = {}
        else:
            transitions = self.replay_memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            # next_state_batch = torch.cat(batch.next_state)
            # state_batch = torch.cat(batch.state)
            next_state_batch = batch.next_state
            state_batch = batch.state
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
            target_state_action_values = (
                next_state_values * self.gamma
            ) + reward_batch.to(self.device)

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
        print(f"ppa dict: {ppas_dict}")
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
    ):
        try:
            loss = loss.item()
            q_values = np.mean(info["q_values"])
            target_q_values = np.mean(info["target_q_values"])
            positive_rewards_number = info["positive_rewards_number"]
        except Exception:
            loss = loss
            q_values = 0.0
            target_q_values = 0.0
            positive_rewards_number = 0.0

        logger.tb_logger.add_scalar("train loss", loss, global_step=self.total_steps)
        logger.tb_logger.add_scalar("reward", reward, global_step=self.total_steps)
        logger.tb_logger.add_scalar(
            "avg ppa", rewards_dict["avg_ppa"], global_step=self.total_steps
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
        logger.tb_logger.add_scalar(
            "positive_rewards_number",
            positive_rewards_number,
            global_step=self.total_steps,
        )

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
        if self.env.initial_state_pool_max_len > 0:
            logger.tb_logger.add_scalar(
                "env pool length",
                len(self.env.initial_state_pool),
                global_step=self.total_steps,
            )
        # log q values info
        logger.tb_logger.add_scalar("q_values", q_values, global_step=self.total_steps)
        logger.tb_logger.add_scalar(
            "target_q_values", target_q_values, global_step=self.total_steps
        )

    def get_env_pool_log(self):
        avg_area = {}
        avg_delay = {}
        for i in range(len(self.env.initial_state_pool)):
            avg_area[f"{i}-th state area"] = self.env.initial_state_pool[i]["area"]
            avg_delay[f"{i}-th state delay"] = self.env.initial_state_pool[i]["delay"]
        return avg_area, avg_delay

    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.env.initial_state_pool_max_len > 0:
            if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
                self.env.initial_state_pool.append(
                    copy.deepcopy(state)
                )
        if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
            self.found_best_info["found_best_ppa"] = rewards_dict["avg_ppa"]
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict["area"])
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict["delay"])

    def reset_agent(self):
        if self.total_steps % self.agent_reset_freq == 0:
            self.q_policy.partially_reset(reset_type=self.agent_reset_type)
            self.target_q_policy.partially_reset(reset_type=self.agent_reset_type)

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
        data_points = pd.DataFrame({"area": area_list, "delay": delay_list})
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
        logger.tb_logger.add_scalar("hypervolume", hv_value, global_step=episode_num)
        logger.log(f"episode {episode_num}, hypervolume: {hv_value}")

        # 3. log pareto points
        fig1 = plt.figure()
        x = new_pareto_area_list
        y = new_pareto_delay_list
        plt.scatter(x, y, c="r")
        logger.tb_logger.add_figure("pareto points", fig1, global_step=episode_num)

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
        combine_array = np.array(combine_array)
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
        plt.scatter(x, y, c="r")
        logger.tb_logger.add_figure(
            "testing pareto points", fig1, global_step=episode_num
        )

        save_data_dict["testing_pareto_points_area"] = true_pareto_area_list
        save_data_dict["testing_pareto_points_delay"] = true_pareto_delay_list

        return save_data_dict

    def run_episode(self, episode_num):
        episode_area = []
        episode_delay = []
        # init state
        env_state = self.env.reset(self.initial_adder_type, is_from_pool=self.env.is_from_pool)
        state = env_state.copy()
        for step in range(self.len_per_episode):
            self.total_steps += 1
            # environment interaction
            state.update_available_choice()
            action, policy_info = self.q_policy.select_action(
                state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax,
            )

            next_state, reward, rewards_dict = self.env.step(action)
            _, next_state_policy_info = self.q_policy.select_action(
                next_state,
                self.total_steps,
                deterministic=self.deterministic,
                is_softmax=self.is_softmax,
            )

            # store data
            if self.store_type == "simple":
                self.store(
                    state,
                    next_state,
                    action,
                    reward,
                    policy_info["mask"],
                    next_state_policy_info["mask"],
                )
            elif self.store_type == "detail":
                self.store_detail(
                    state,
                    next_state,
                    action,
                    reward,
                    policy_info["mask"],
                    next_state_policy_info["mask"],
                    rewards_dict,
                )

            # update initial state pool
            self.update_env_initial_state_pool(
                next_state, rewards_dict, next_state_policy_info["mask"]
            )
            # update q policy
            loss, info = self.update_q()

            # update target q (TODO: SOFT UPDATE)
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_policy.load_state_dict(self.q_policy.state_dict())
            # state = copy.deepcopy(next_state)
            state = next_state.copy()
            # reset agent
            if self.agent_reset_freq > 0:
                self.reset_agent()
            # log datasets
            self.log_stats(
                loss, reward, rewards_dict, next_state, action, info, policy_info
            )
            avg_ppa = rewards_dict["avg_ppa"]
            # episode_area.extend(rewards_dict["area"])
            episode_area.append(rewards_dict["area"])
            # episode_delay.extend(rewards_dict["delay"])
            episode_delay.append(rewards_dict["delay"])
            logger.log(
                f"total steps: {self.total_steps}, avg ppa: {avg_ppa}, action = {action}, delay = {rewards_dict['delay']}, area = {rewards_dict['area']}"
            )
        # update target q
        self.target_q_policy.load_state_dict(self.q_policy.state_dict())
        # process and log pareto
        # self.process_and_log_pareto(episode_num, episode_area, episode_delay)

class DQNGAAlgorithm(DQNAlgorithm):
    def __init__(
        self,
        env: RefineEnv,
        q_policy: DeepQPolicy,
        target_q_policy: DeepQPolicy,
        replay_memory: ReplayMemory,
        start_episodes=40,
        model_env_iterative_epi_num=20,
        **dqn_alg_kwargs
    ):
        super().__init__(
            env,
            q_policy,
            target_q_policy,
            replay_memory,
            **dqn_alg_kwargs
        )
        self.start_episodes = start_episodes
        self.model_env_iterative_epi_num = model_env_iterative_epi_num

    def run_experiments(self):
        search_type = 1
        for episode_num in range(self.total_episodes):
            if episode_num < self.start_episodes:
                # 1. warm start DQN
                self.run_episode(episode_num)
                if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                    self.end_experiments(episode_num)
            else:
                # 2. iterate between learning and EA
                if (episode_num+1) % self.model_env_iterative_epi_num == 0:
                    search_type = -1 * search_type
                if search_type == 1:
                    self.EA_search(episode_num)
                if search_type == -1:
                    self.run_episode(episode_num)
                if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                    self.end_experiments(episode_num + self.start_episodes)
        self.end_experiments(episode_num)

    def evaluate_state(self, state, step, episode_num):
        # simulation
        areas = []
        delays = []
        powers = []
        for td in self.env.target_delay:
            state.output_verilog(self.env.build_path)
            state.run_yosys(self.env.build_path, td)
            delay, area, power = state.run_openroad(self.env.build_path)
            areas.append(area)
            delays.append(delay)
            powers.append(power)
        
        area = np.mean(areas)
        delay = np.mean(delays)
        power = np.mean(powers)
        # reward
        rewards_dict = {"delay": delay, "area": area, "power": power}
        ppa = self.env.weight_area * (area / self.env.area_scale) + self.env.weight_delay * (
            delay / self.env.delay_scale
        )
        ppa = self.env.ppa_scale * ppa

        rewards_dict["avg_ppa"] = ppa
        rewards_dict["reward"] = 0

        if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
            self.env.initial_state_pool.append(
                copy.deepcopy(state)
            )
            logger.log(f"total steps: {episode_num*self.len_per_episode+step}, found better avg ppa: {ppa}")
            logger.tb_logger.add_scalar(f'found better avg ppa', ppa, global_step=episode_num*self.len_per_episode+step)

        if self.found_best_info["found_best_ppa"] > rewards_dict["avg_ppa"]:
            self.found_best_info["found_best_ppa"] = rewards_dict["avg_ppa"]
            self.found_best_info["found_best_state"] = copy.deepcopy(state)
            self.found_best_info["found_best_area"] = np.mean(rewards_dict["area"])
            self.found_best_info["found_best_delay"] = np.mean(rewards_dict["delay"])

    def EA_search(self, episode_num):
        for step in range(self.len_per_episode):
            self.total_steps += 1
            pool_length = len(self.env.initial_state_pool)
            sel_indexes = np.random.choice(pool_length, 2)
            state1 = self.env.initial_state_pool[sel_indexes[0]]
            state2 = self.env.initial_state_pool[sel_indexes[1]]
            
            crossover_states = []
            # column crossover
            cc_state1, cc_state2 = self.env.column_crossover(
                copy.deepcopy(state1), copy.deepcopy(state2)
            )
            # block crossover
            bc_state1, bc_state2 = self.env.block_crossover(
                copy.deepcopy(state1), copy.deepcopy(state2)
            )
            if cc_state1 is not None:
                crossover_states.append(cc_state1)
            if cc_state2 is not None:
                crossover_states.append(cc_state2)
            if bc_state1 is not None:
                crossover_states.append(bc_state1)
            if bc_state2 is not None:
                crossover_states.append(bc_state2)
            
            if len(crossover_states) > 0:
                for crossover_state in crossover_states:
                    self.evaluate_state(crossover_state, step, episode_num)

class MCTSNode(object):
    def __init__(self):
        self.parent = None
        self.children = []

        self.visit_times = 0
        self.quality_value = 0.0
        self.best_reward = -sys.maxsize

        self.state = None

    def set_state(self, state: State):
        self.state = state

    def get_state(self) -> State:
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def update_best_reward(self, n):
        self.best_reward = max(self.best_reward, n)

    def get_best_reward(self):
        return self.best_reward

    def is_all_expand(self):
        return len(self.children) == self.state.available_choice

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, best: {}, state: {}".format(
            hash(self),
            self.quality_value,
            self.visit_times,
            self.best_reward,
            self.state,
        )


class MCTSAlgorithm:
    def __init__(
        self,
        env: RefineEnv,
        initial_adder_type=0,
        total_episodes: int = 1,
        len_per_episode: int = 1000,
        end_exp_freq: int = 25,
        build_path: str = "build",
        weight_area=1,
        weight_delay=1,
        ppa_scale=1,
        area_scale=1,
        delay_scale=1,
    ):
        self.initial_adder_type = initial_adder_type
        self.env = env
        self.len_per_episode = len_per_episode
        self.total_episodes = total_episodes
        self.end_exp_freq = end_exp_freq

        self.build_path = build_path
        self.weight_area = weight_area
        self.weight_delay = weight_delay
        self.ppa_scale = ppa_scale
        self.delay_scale = delay_scale
        self.area_scale = area_scale

        self.int_bit_width = env.bit_width

        # total steps
        self.total_steps = 0

        # best ppa found
        self.found_best_info = {
            "found_best_ppa": 1e5,
            "found_best_state": None,
            "found_best_area": 1e5,
            "found_best_delay": 1e5,
        }

        # pareto pointset
        self.pareto_pointset = {"area": [], "delay": [], "state": []}
        # configure figure
        plt.switch_backend("agg")

    def end_experiments(self, episode_num):
        # save datasets
        save_data_dict = {}
        # best state best design
        save_data_dict["found_best_info"] = self.found_best_info
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]

        logger.save_npy(self.total_steps, save_data_dict)

    def run_experiments(self, stdscr=None):
        for episode_num in range(self.total_episodes):
            self.run_episode(episode_num, stdscr=stdscr)
            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(episode_num)
        self.end_experiments(episode_num)

    def log_action_stats(self, action):
        pass

    def log_best_info(self, step: int) -> None:
        logger.tb_logger.add_scalar(
            "best ppa",
            self.found_best_info["found_best_ppa"],
            global_step=step,
        )
        logger.tb_logger.add_scalar(
            "best area",
            self.found_best_info["found_best_area"],
            global_step=step,
        )
        logger.tb_logger.add_scalar(
            "best delay",
            self.found_best_info["found_best_delay"],
            global_step=step,
        )

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
        data_points = pd.DataFrame({"area": area_list, "delay": delay_list})
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
        logger.tb_logger.add_scalar("hypervolume", hv_value, global_step=episode_num)
        logger.log(f"episode {episode_num}, hypervolume: {hv_value}")

        # 3. log pareto points
        fig1 = plt.figure()
        x = new_pareto_area_list
        y = new_pareto_delay_list
        plt.scatter(x, y, c="r")
        logger.tb_logger.add_figure("pareto points", fig1, global_step=episode_num)

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
        plt.scatter(x, y, c="r")
        logger.tb_logger.add_figure(
            "testing pareto points", fig1, global_step=episode_num
        )

        save_data_dict["testing_pareto_points_area"] = true_pareto_area_list
        save_data_dict["testing_pareto_points_delay"] = true_pareto_delay_list

        return save_data_dict

    def tree_policy(self, node: MCTSNode) -> MCTSNode:
        eps = 0.8
        while not node.get_state().is_terminal():
            if node.is_all_expand() or (
                random.random() > eps and len(node.get_children()) >= 1
            ):
                print("\rIS ALL EXPAND")
                node = self.best_child(node, True)
            else:
                node = self.expand(node)
                break
        return node

    def default_policy(self, node: MCTSNode) -> float:
        current_state = node.get_state()
        best_state_reward = current_state.compute_reward(
            self.weight_area,
            self.weight_delay,
            self.ppa_scale,
            self.area_scale,
            self.delay_scale,
        )
        reward_dict = {
            "avg_ppa": -best_state_reward,
            "area": current_state.area,
            "delay": current_state.delay,
            "reward": best_state_reward,
        }
        self.total_steps += 1
        self.update_env_initial_state_pool(current_state, reward_dict, None)
        logger.tb_logger.add_scalar("reward", best_state_reward, self.total_steps)

        step = 0
        while not current_state.is_terminal() and (
            (step < 16 and current_state.initial_adder_type == 0)
            or (step < 16 and current_state.initial_adder_type != 0)
        ):
            current_state = current_state.get_next_state_with_random_choice(
                self.build_path
            )
            if current_state is None:
                break
            print("\r-------- step = {} ---------".format(step))
            step += 1
            best_state_reward = max(
                best_state_reward,
                current_state.compute_reward(
                    self.weight_area,
                    self.weight_delay,
                    self.ppa_scale,
                    self.area_scale,
                    self.delay_scale,
                ),
            )
            reward_dict = {
                "avg_ppa": -best_state_reward,
                "area": current_state.area,
                "delay": current_state.delay,
                "reward": best_state_reward,
            }
            self.update_env_initial_state_pool(current_state, reward_dict, None)
            self.total_steps += 1
            logger.tb_logger.add_scalar("reward", best_state_reward, self.total_steps)

        print("\rdefault policy finished")
        return best_state_reward

    def expand(self, node: MCTSNode) -> MCTSNode:
        tried_sub_node_states = [
            sub_node.get_state().action for sub_node in node.get_children()
        ]
        new_state = node.get_state().get_next_state_with_random_choice(self.build_path)
        while new_state.action in tried_sub_node_states:
            new_state = node.get_state().get_next_state_with_random_choice(
                self.build_path
            )

        sub_node = MCTSNode()
        sub_node.set_state(new_state)
        node.add_child(sub_node)
        return sub_node

    def best_child(self, node: MCTSNode, is_exploration: bool) -> MCTSNode:
        best_score = -sys.maxsize
        best_sub_node = None
        for sub_node in node.get_children():
            if is_exploration:
                C = 1 / math.sqrt(2.0)
            else:
                C = 0.0

            if node.get_visit_times() >= 1e-2 and sub_node.get_visit_times() >= 1e-2:
                left = (
                    sub_node.get_best_reward() * 0.99
                    + sub_node.get_quality_value() / sub_node.get_visit_times() * 0.01
                )
                right = math.log(node.get_visit_times()) / sub_node.get_visit_times()
                right = C * 10 * math.sqrt(right)
                print("\rleft = {}, right = {}".format(left, right))
                score = left + right
            else:
                score = 1e9

            if score > best_score:
                best_sub_node = sub_node
                best_score = score

        return best_sub_node

    def backup(self, node: MCTSNode, reward: float) -> MCTSNode:
        while node is not None:
            node.visit_times_add_one()
            node.quality_value_add_n(reward)
            node.update_best_reward(reward)

            if node.parent is not None:
                node = node.parent
            else:
                break

        assert node is not None
        assert node.parent is None
        return node

    def run_episode(self, episode_num, stdscr=None):
        episode_area = []
        episode_delay = []
        # init state
        env_state = self.env.reset(self.initial_adder_type)
        state = env_state.copy()
        node = MCTSNode()
        node.set_state(state)

        for step in range(self.len_per_episode):
            if stdscr is not None:
                stdscr.clear()
            print(f"\r================= start search step {step} =================")
            node = self.tree_policy(node)
            reward = self.default_policy(node)
            node = self.backup(node, reward)

            if stdscr is not None:
                stdscr.refresh()

            self.log_best_info(step)
            if step % 50 == 0 and step > 1:
                self.end_experiments(step)
