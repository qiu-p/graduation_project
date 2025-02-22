"""
    DQN training algorithm drawed by the paper 
    "RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning"
"""
import copy
import math
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from paretoset import paretoset
from torch.multiprocessing import Pool, set_start_method
import torch.multiprocessing as mp
from pygmo import hypervolume

from o0_logger import logger
from o5_utils import Transition, MBRLTransition, MultiObjTransition, SharedMultiObjTransition
from o0_global_const import PartialProduct
# from o1_environment import RefineEnv, SerialRefineEnv
from o1_environment_speedup import SerialSpeedUpRefineEnv as SerialRefineEnv

from ipdb import set_trace

"""
    function for scalar q learning
"""
#### serial ####
def compute_values(
    state_batch, action_batch, state_mask, env, q_policy, target_q_policy, task_index,
    device, initial_partial_product, MAX_STAGE_NUM, int_bit_width
):
    batch_size = len(state_batch)
    state_action_values = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        # compute image state
        ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < MAX_STAGE_NUM-1:
            zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        state = torch.cat((ct32, ct22), dim=0)
        # compute image state
        if action_batch is not None:
            q_values = q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i])[task_index]
            q_values = q_values.reshape((int(int_bit_width*2))*4)           
            # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
            state_action_values[i] = q_values[action_batch[i]]
        else:
            q_values = target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])[task_index]
            state_action_values[i] = q_values.max(1)[0].detach()
    return state_action_values

#### parallel ####
# def compute_values(
#     state_batch, action_batch, state_mask, env, q_policy, target_q_policy, task_index,
#     device, initial_partial_product, MAX_STAGE_NUM, int_bit_width
# ):
#     batch_size = len(state_batch)
#     state_action_values = torch.zeros(batch_size, device=device)
#     states = []
#     for i in range(batch_size):
#         # compute image state
#         ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
#         ct32 = torch.tensor(np.array([ct32]))
#         ct22 = torch.tensor(np.array([ct22]))
#         if stage_num < MAX_STAGE_NUM-1:
#             zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
#             ct32 = torch.cat((ct32, zeros), dim=1)
#             ct22 = torch.cat((ct22, zeros), dim=1)
#         state = torch.cat((ct32, ct22), dim=0)
#         states.append(state.unsqueeze(0))
#     states = torch.cat(states)
#     # compute image state
#     if action_batch is not None:
#         q_values = q_policy(states.float(), state_mask=state_mask)[task_index]
#         q_values = q_values.reshape(-1, (int(int_bit_width*2))*4)           
#         # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
#         for i in range(batch_size):
#             state_action_values[i] = q_values[i, action_batch[i]]
#     else:
#         q_values = target_q_policy(states.float(), is_target=True, state_mask=state_mask)[task_index]
#         for i in range(batch_size):
#             state_action_values[i] = q_values[i:i+1].max(1)[0].detach()
#     return state_action_values

def decode_transition(transitions):
    batch = {
        "state": [],
        "action": [],
        "next_state": [],
        "reward": [],
        "mask": [],
        "next_state_mask": [],
        "area_reward": [],
        "delay_reward": []
    }
    for transition in transitions:
        batch["state"].append(transition.state)
        batch["action"].append(transition.action)
        batch["next_state"].append(transition.next_state)
        batch["reward"].append(transition.reward)
        batch["mask"].append(transition.mask)
        batch["next_state_mask"].append(transition.next_state_mask)
        batch["area_reward"].append(transition.area_reward)
        batch["delay_reward"].append(transition.delay_reward)
    return batch

#### serial ####
def compute_int_rewards(state_batch, state_mask, rnd_predictor, rnd_target, env, initial_partial_product, MAX_STAGE_NUM, int_bit_width):
    batch_size = len(state_batch)
    int_rewards = torch.zeros(batch_size, device=rnd_predictor.device)
    for i in range(batch_size):
        ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < MAX_STAGE_NUM-1:
            zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        state = torch.cat((ct32, ct22), dim=0)
        
        with torch.no_grad():
            predict_value = rnd_predictor(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i]).reshape((int(int_bit_width*2))*4)
            target_value = rnd_target(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i]).reshape((int(int_bit_width*2))*4)
        # set_trace()
        int_rewards[i] = torch.sum(
            (predict_value - target_value)**2
        )
    return int_rewards

#### parallel ####
# def compute_int_rewards(state_batch, state_mask, rnd_predictor, rnd_target, env, initial_partial_product, MAX_STAGE_NUM, int_bit_width):
#     batch_size = len(state_batch)
#     int_rewards = torch.zeros(batch_size, device=rnd_predictor.device)
#     states = []
#     for i in range(batch_size):
#         ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
#         ct32 = torch.tensor(np.array([ct32]))
#         ct22 = torch.tensor(np.array([ct22]))
#         if stage_num < MAX_STAGE_NUM-1:
#             zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
#             ct32 = torch.cat((ct32, zeros), dim=1)
#             ct22 = torch.cat((ct22, zeros), dim=1)
#         state = torch.cat((ct32, ct22), dim=0)
#         states.append(state.unsqueeze(0))
#     states = torch.cat(states)
#     with torch.no_grad():
#         predict_value = rnd_predictor(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (int(int_bit_width*2))*4)
#         target_value = rnd_target(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (int(int_bit_width*2))*4)
#     # set_trace()
#     int_rewards = torch.sum(
#         (predict_value - target_value)**2, dim=1
#     )
#     return int_rewards

def compute_q_loss(replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std, loss_fn, task_weight_vectors, task_index, **q_kwargs):
    if len(replay_memory) < q_kwargs["batch_size"]:
        loss = 0.
        info = {
            "is_update": False
        }
        return loss, info
    else:
        transitions = replay_memory.sample(q_kwargs["batch_size"])
        batch = decode_transition(transitions)
        next_state_batch = torch.tensor(np.concatenate(batch["next_state"]))
        state_batch = torch.tensor(np.concatenate(batch["state"]))
        action_batch = torch.tensor(np.concatenate(batch["action"]))
        reward_batch = torch.tensor(np.concatenate(batch["reward"]))
        state_mask = torch.tensor(np.concatenate(batch["mask"]))
        next_state_mask = torch.tensor(np.concatenate(batch["next_state_mask"]))
        area_reward_batch = torch.tensor(np.concatenate(batch["area_reward"]))
        delay_reward_batch = torch.tensor(np.concatenate(batch["delay_reward"]))
        # TODO: add rnd model update reward int run mean std
        # self.update_reward_int_run_mean_std(
        #     reward_batch.cpu().numpy()
        # )
        # compute reward int 
        int_rewards_batch = compute_int_rewards(
            next_state_batch, next_state_mask, rnd_predictor, rnd_target, env,
            q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
        )
        int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(int_reward_run_mean_std.var), device=rnd_predictor.device)
        # train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
        q_policy = q_policy.to(q_kwargs["device"])
        target_q_policy = target_q_policy.to(q_kwargs["device"])

        if q_kwargs["loss_type"] == "mix":
            num_task = len(task_weight_vectors)
            losses = torch.zeros(num_task, device=q_kwargs["device"])
            q_values = []
            target_q_values = []
            for i, task_weight in enumerate(task_weight_vectors):
                train_reward_batch = task_weight[0] * area_reward_batch + task_weight[1] * delay_reward_batch
                train_reward_batch = train_reward_batch.to(q_kwargs["device"])

                state_action_values = compute_values(
                    state_batch, action_batch, state_mask,
                    env, q_policy, target_q_policy, i,
                    q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
                )
                next_state_values = compute_values(
                    next_state_batch, None, next_state_mask,
                    env, q_policy, target_q_policy, i,
                    q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
                )
                target_state_action_values = (next_state_values * q_kwargs["gamma"]) + train_reward_batch

                loss = loss_fn(
                    state_action_values.unsqueeze(1), 
                    target_state_action_values.unsqueeze(1)
                )
                losses[i] = loss
                q_values.append(state_action_values.detach().cpu().numpy())
                target_q_values.append(target_state_action_values.detach().cpu().numpy())
            
            mean_loss = torch.mean(losses)
            info = {
                "loss": mean_loss.item(),
                "q_values": q_values,
                "target_q_values": target_q_values,
                "is_update": True
            }
        elif q_kwargs["loss_type"] == "separate":
            train_reward_batch = task_weight_vectors[task_index][0] * area_reward_batch + task_weight_vectors[task_index][1] * delay_reward_batch
            train_reward_batch = train_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"]

            state_action_values = compute_values(
                state_batch, action_batch, state_mask,
                env, q_policy, target_q_policy, task_index,
                q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
            )
            next_state_values = compute_values(
                next_state_batch, None, next_state_mask,
                env, q_policy, target_q_policy, task_index,
                q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
            )
            target_state_action_values = (next_state_values * q_kwargs["gamma"]) + train_reward_batch

            mean_loss = loss_fn(
                state_action_values.unsqueeze(1), 
                target_state_action_values.unsqueeze(1)
            )
            info = {
                "loss": mean_loss.item(),
                "q_values": state_action_values.detach().cpu().numpy(),
                "target_q_values": target_state_action_values.detach().cpu().numpy(),
                "is_update": True,
                "int_rewards": int_rewards_batch.cpu().numpy()
            }

    return mean_loss, info

def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()

"""
    function for vector condition q learning
"""
def compute_values_vector_conditionq(
    state_batch, action_batch, state_mask, env, q_policy, target_q_policy, task_index,
    device, initial_partial_product, MAX_STAGE_NUM, int_bit_width, weight_condition, delay_condition
):
    batch_size = len(state_batch)
    state_action_values = torch.zeros(batch_size, device=device)
    state_action_area_values = torch.zeros(batch_size, device=device)
    state_action_delay_values = torch.zeros(batch_size, device=device)
    weight_condition = weight_condition.float().unsqueeze(0)
    for i in range(batch_size):
        # compute image state
        ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < MAX_STAGE_NUM-1:
            zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        
        state = torch.cat((ct32, ct22), dim=0)
        cur_delay_condition = delay_condition[i:i+1].float().unsqueeze(0)
        # compute image state
        if action_batch is not None:
            q_area_list, q_delay_list, q_values_list = q_policy(state.unsqueeze(0).float(), weight_condition, cur_delay_condition, state_mask=state_mask[i])
            q_area = q_area_list[task_index].reshape((int(int_bit_width*2))*4)
            q_delay = q_delay_list[task_index].reshape((int(int_bit_width*2))*4)
            q_values = q_values_list[task_index].reshape((int(int_bit_width*2))*4)
            # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
            state_action_values[i] = q_values[action_batch[i]]
            state_action_area_values[i] = q_area[action_batch[i]]
            state_action_delay_values[i] = q_delay[action_batch[i]]
        else:
            q_area_list, q_delay_list, q_values_list = target_q_policy(state.unsqueeze(0).float(), weight_condition, cur_delay_condition, is_target=True, state_mask=state_mask[i])
            state_action_values[i] = q_values_list[task_index].max(1)[0].detach()
            cur_q_values = q_values_list[task_index].reshape((int(int_bit_width*2))*4)
            index = torch.argmax(cur_q_values)
            state_action_area_values[i] = q_area_list[task_index].squeeze()[index].detach()
            state_action_delay_values[i] = q_delay_list[task_index].squeeze()[index].detach()
    return state_action_values, state_action_area_values, state_action_delay_values

def decode_transition_vector_conditionq(transitions):
    batch = {
        "state": [],
        "action": [],
        "next_state": [],
        "reward": [],
        "mask": [],
        "next_state_mask": [],
        "area_reward": [],
        "delay_reward": [],
        "weight_vector": [],
        "delay_condition": []
    }
    for transition in transitions:
        batch["state"].append(transition.state)
        batch["action"].append(transition.action)
        batch["next_state"].append(transition.next_state)
        batch["reward"].append(transition.reward)
        batch["mask"].append(transition.mask)
        batch["next_state_mask"].append(transition.next_state_mask)
        batch["area_reward"].append(transition.area_reward)
        batch["delay_reward"].append(transition.delay_reward)
        batch["weight_vector"].append(transition.weight_vector)
        batch["delay_condition"].append(transition.delay_condition)
    return batch

def compute_q_loss_vector_conditionq(replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std, loss_fn, task_weight_vectors, task_index, **q_kwargs):
    if len(replay_memory) < q_kwargs["batch_size"]:
        loss = 0.
        info = {
            "is_update": False
        }
        return loss, info
    else:
        transitions = replay_memory.sample(q_kwargs["batch_size"])
        batch = decode_transition_vector_conditionq(transitions)
        next_state_batch = torch.tensor(np.concatenate(batch["next_state"]))
        state_batch = torch.tensor(np.concatenate(batch["state"]))
        action_batch = torch.tensor(np.concatenate(batch["action"]))
        reward_batch = torch.tensor(np.concatenate(batch["reward"]))
        state_mask = torch.tensor(np.concatenate(batch["mask"]))
        next_state_mask = torch.tensor(np.concatenate(batch["next_state_mask"]))
        area_reward_batch = torch.tensor(np.concatenate(batch["area_reward"]))
        delay_reward_batch = torch.tensor(np.concatenate(batch["delay_reward"]))
        delay_condition_batch = torch.tensor(np.concatenate(batch["delay_condition"]))

        weight_condition = torch.tensor(np.array(task_weight_vectors[task_index]))

        # compute reward int 
        int_rewards_batch = compute_int_rewards(
            next_state_batch, next_state_mask, rnd_predictor, rnd_target, env,
            q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
        )
        int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(int_reward_run_mean_std.var), device=rnd_predictor.device)
        
        q_policy = q_policy.to(q_kwargs["device"])
        target_q_policy = target_q_policy.to(q_kwargs["device"])

        # train reward
        train_reward_batch = task_weight_vectors[task_index][0] * area_reward_batch + task_weight_vectors[task_index][1] * delay_reward_batch
        train_reward_batch = train_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"]
        # area reward
        train_area_reward_batch = area_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"] / task_weight_vectors[task_index][0]
        # delay reward 
        train_delay_reward_batch = delay_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"] / task_weight_vectors[task_index][1]

        # comments by zhihai: 
        # 

        state_action_values, state_action_area_values, state_action_delay_values = compute_values_vector_conditionq(
            state_batch, action_batch, state_mask,
            env, q_policy, target_q_policy, task_index,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"],
            weight_condition, delay_condition_batch
        )
        next_state_values, next_state_area_values, next_state_delay_values = compute_values_vector_conditionq(
            next_state_batch, None, next_state_mask,
            env, q_policy, target_q_policy, task_index,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"],
            weight_condition, delay_condition_batch
        )

        target_state_action_values = (next_state_values * q_kwargs["gamma"]) + train_reward_batch
        target_state_action_area_values = (next_state_area_values * q_kwargs["gamma"]) + train_area_reward_batch
        target_state_action_delay_values = (next_state_delay_values * q_kwargs["gamma"]) + train_delay_reward_batch
        
        area_loss = loss_fn(
            state_action_area_values.unsqueeze(1), 
            target_state_action_area_values.unsqueeze(1)
        )
        delay_loss = loss_fn(
            state_action_delay_values.unsqueeze(1), 
            target_state_action_delay_values.unsqueeze(1)   
        )
        mean_loss = area_loss + delay_loss
        info = {
            "loss": mean_loss.item(),
            "q_values": state_action_values.detach().cpu().numpy(),
            "target_q_values": target_state_action_values.detach().cpu().numpy(),
            "is_update": True,
            "int_rewards": int_rewards_batch.cpu().numpy()
        }

    return mean_loss, info

"""
    parallel of function for vector condiiton q learning
"""
def compute_int_rewards_parallel(state_batch, state_mask, rnd_predictor, rnd_target, env, initial_partial_product, MAX_STAGE_NUM, int_bit_width):
    batch_size = len(state_batch)
    int_rewards = torch.zeros(batch_size, device=rnd_predictor.device)
    states = []
    for i in range(batch_size):
        ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < MAX_STAGE_NUM-1:
            zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        state = torch.cat((ct32, ct22), dim=0)
        states.append(state.unsqueeze(0))
    states = torch.cat(states)
    with torch.no_grad():
        predict_value = rnd_predictor(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (int(int_bit_width*2))*4)
        target_value = rnd_target(states.float(), is_target=True, state_mask=state_mask).reshape(-1, (int(int_bit_width*2))*4)
    
    int_rewards = torch.sum(
        (predict_value - target_value)**2, dim=1
    )
    return int_rewards

def compute_values_vector_conditionq_parallel(
    state_batch, action_batch, state_mask, env, q_policy, target_q_policy, task_index,
    device, initial_partial_product, MAX_STAGE_NUM, int_bit_width, weight_condition, delay_condition
):
    batch_size = len(state_batch)
    state_action_values = torch.zeros(batch_size, device=device)
    state_action_area_values = torch.zeros(batch_size, device=device)
    state_action_delay_values = torch.zeros(batch_size, device=device)
    weight_condition = weight_condition.float().unsqueeze(0)
    weight_conditions = []
    delay_conditions = []
    states = []
    for i in range(batch_size):
        # compute image state
        ct32, ct22, pp, stage_num = env.decompose_compressor_tree(initial_partial_product, state_batch[i].cpu().numpy())
        ct32 = torch.tensor(np.array([ct32]))
        ct22 = torch.tensor(np.array([ct22]))
        if stage_num < MAX_STAGE_NUM-1:
            zeros = torch.zeros(1, MAX_STAGE_NUM-1-stage_num, int(int_bit_width*2))
            ct32 = torch.cat((ct32, zeros), dim=1)
            ct22 = torch.cat((ct22, zeros), dim=1)
        state = torch.cat((ct32, ct22), dim=0)
        states.append(state.unsqueeze(0))
        weight_conditions.append(weight_condition)
        delay_conditions.append(delay_condition[i:i+1].unsqueeze(0))
    states = torch.cat(states)
    weight_conditions = torch.cat(weight_conditions)
    delay_conditions = torch.cat(delay_conditions)
        
    # compute image state
    if action_batch is not None:
        q_area_list, q_delay_list, q_values_list = q_policy(states.float(), weight_conditions.float(), delay_conditions.float(), state_mask=state_mask)
        q_area = q_area_list[task_index].reshape(-1, (int(int_bit_width*2))*4)
        q_delay = q_delay_list[task_index].reshape(-1, (int(int_bit_width*2))*4)
        q_values = q_values_list[task_index].reshape(-1, (int(int_bit_width*2))*4)
        for i in range(batch_size):
            state_action_values[i] = q_values[i, action_batch[i]]
            state_action_area_values[i] = q_area[i, action_batch[i]]
            state_action_delay_values[i] = q_delay[i, action_batch[i]]
    else:
        q_area_list, q_delay_list, q_values_list = target_q_policy(states.float(), weight_conditions.float(), delay_conditions.float(), is_target=True, state_mask=state_mask)
        for i in range(batch_size):
            state_action_values[i] = q_values_list[task_index][i:i+1].max(1)[0].detach()
        cur_q_values = q_values_list[task_index].reshape(-1, (int(int_bit_width*2))*4)
        for i in range(batch_size):
            index = torch.argmax(cur_q_values[i])
            state_action_area_values[i] = q_area_list[task_index][i:i+1].squeeze()[index].detach()
            state_action_delay_values[i] = q_delay_list[task_index][i:i+1].squeeze()[index].detach()
    return state_action_values, state_action_area_values, state_action_delay_values

def compute_q_loss_vector_conditionq_parallel(replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std, loss_fn, task_weight_vectors, task_index, **q_kwargs):
    if len(replay_memory) < q_kwargs["batch_size"]:
        loss = 0.
        info = {
            "is_update": False
        }
        return loss, info
    else:
        transitions = replay_memory.sample(q_kwargs["batch_size"])
        batch = decode_transition_vector_conditionq(transitions)
        next_state_batch = torch.tensor(np.concatenate(batch["next_state"]))
        state_batch = torch.tensor(np.concatenate(batch["state"]))
        action_batch = torch.tensor(np.concatenate(batch["action"]))
        reward_batch = torch.tensor(np.concatenate(batch["reward"]))
        state_mask = torch.tensor(np.concatenate(batch["mask"]))
        next_state_mask = torch.tensor(np.concatenate(batch["next_state_mask"]))
        area_reward_batch = torch.tensor(np.concatenate(batch["area_reward"]))
        delay_reward_batch = torch.tensor(np.concatenate(batch["delay_reward"]))
        delay_condition_batch = torch.tensor(np.concatenate(batch["delay_condition"]))

        weight_condition = torch.tensor(np.array(task_weight_vectors[task_index]))

        # compute reward int 
        int_rewards_batch = compute_int_rewards_parallel(
            next_state_batch, next_state_mask, rnd_predictor, rnd_target, env,
            q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
        )
        int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(int_reward_run_mean_std.var), device=rnd_predictor.device)
        
        q_policy = q_policy.to(q_kwargs["device"])
        target_q_policy = target_q_policy.to(q_kwargs["device"])

        # train reward
        train_reward_batch = task_weight_vectors[task_index][0] * area_reward_batch + task_weight_vectors[task_index][1] * delay_reward_batch
        train_reward_batch = train_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"]
        # area reward
        train_area_reward_batch = area_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"] / task_weight_vectors[task_index][0]
        # delay reward 
        train_delay_reward_batch = delay_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"] / task_weight_vectors[task_index][1]

        # comments by zhihai: 
        # 

        state_action_values, state_action_area_values, state_action_delay_values = compute_values_vector_conditionq_parallel(
            state_batch, action_batch, state_mask,
            env, q_policy, target_q_policy, task_index,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"],
            weight_condition, delay_condition_batch
        )
        next_state_values, next_state_area_values, next_state_delay_values = compute_values_vector_conditionq_parallel(
            next_state_batch, None, next_state_mask,
            env, q_policy, target_q_policy, task_index,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"],
            weight_condition, delay_condition_batch
        )

        target_state_action_values = (next_state_values * q_kwargs["gamma"]) + train_reward_batch
        target_state_action_area_values = (next_state_area_values * q_kwargs["gamma"]) + train_area_reward_batch
        target_state_action_delay_values = (next_state_delay_values * q_kwargs["gamma"]) + train_delay_reward_batch
        
        area_loss = loss_fn(
            state_action_area_values.unsqueeze(1), 
            target_state_action_area_values.unsqueeze(1)
        )
        delay_loss = loss_fn(
            state_action_delay_values.unsqueeze(1), 
            target_state_action_delay_values.unsqueeze(1)   
        )
        mean_loss = area_loss + delay_loss
        info = {
            "loss": mean_loss.item(),
            "q_values": state_action_values.detach().cpu().numpy(),
            "target_q_values": target_state_action_values.detach().cpu().numpy(),
            "is_update": True,
            "int_rewards": int_rewards_batch.cpu().numpy(),
            "area_rewards": area_reward_batch.cpu().numpy(),
            "delay_rewards": delay_reward_batch.cpu().numpy(),
            "target_q_area_values": target_state_action_area_values.detach().cpu().numpy(),
            "target_q_delay_values": target_state_action_delay_values.detach().cpu().numpy()
        }

    return mean_loss, info

def compute_q_loss_vector_conditionq_parallel_weight_loss(replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std, loss_fn, task_weight_vectors, task_index, **q_kwargs):
    if len(replay_memory) < q_kwargs["batch_size"]:
        loss = 0.
        info = {
            "is_update": False
        }
        return loss, info
    else:
        transitions = replay_memory.sample(q_kwargs["batch_size"])
        batch = decode_transition_vector_conditionq(transitions)
        next_state_batch = torch.tensor(np.concatenate(batch["next_state"]))
        state_batch = torch.tensor(np.concatenate(batch["state"]))
        action_batch = torch.tensor(np.concatenate(batch["action"]))
        reward_batch = torch.tensor(np.concatenate(batch["reward"]))
        state_mask = torch.tensor(np.concatenate(batch["mask"]))
        next_state_mask = torch.tensor(np.concatenate(batch["next_state_mask"]))
        area_reward_batch = torch.tensor(np.concatenate(batch["area_reward"]))
        delay_reward_batch = torch.tensor(np.concatenate(batch["delay_reward"]))
        delay_condition_batch = torch.tensor(np.concatenate(batch["delay_condition"]))

        weight_condition = torch.tensor(np.array(task_weight_vectors[task_index]))

        # compute reward int 
        int_rewards_batch = compute_int_rewards_parallel(
            next_state_batch, next_state_mask, rnd_predictor, rnd_target, env,
            q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
        )
        int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(int_reward_run_mean_std.var), device=rnd_predictor.device)
        
        q_policy = q_policy.to(q_kwargs["device"])
        target_q_policy = target_q_policy.to(q_kwargs["device"])

        # train reward
        train_reward_batch = task_weight_vectors[task_index][0] * area_reward_batch + task_weight_vectors[task_index][1] * delay_reward_batch
        train_reward_batch = train_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"]
        # area reward
        train_area_reward_batch = area_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"] / task_weight_vectors[task_index][0]
        # delay reward 
        train_delay_reward_batch = delay_reward_batch.to(q_kwargs["device"]) + int_rewards_batch * q_kwargs["int_reward_scale"] / task_weight_vectors[task_index][1]

        # comments by zhihai: 
        # 

        state_action_values, state_action_area_values, state_action_delay_values = compute_values_vector_conditionq_parallel(
            state_batch, action_batch, state_mask,
            env, q_policy, target_q_policy, task_index,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"],
            weight_condition, delay_condition_batch
        )
        next_state_values, next_state_area_values, next_state_delay_values = compute_values_vector_conditionq_parallel(
            next_state_batch, None, next_state_mask,
            env, q_policy, target_q_policy, task_index,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"],
            weight_condition, delay_condition_batch
        )

        target_state_action_values = (next_state_values * q_kwargs["gamma"]) + train_reward_batch
        target_state_action_area_values = (next_state_area_values * q_kwargs["gamma"]) + train_area_reward_batch
        target_state_action_delay_values = (next_state_delay_values * q_kwargs["gamma"]) + train_delay_reward_batch
        
        area_loss = loss_fn(
            state_action_area_values.unsqueeze(1), 
            target_state_action_area_values.unsqueeze(1)
        )
        delay_loss = loss_fn(
            state_action_delay_values.unsqueeze(1), 
            target_state_action_delay_values.unsqueeze(1)   
        )
        mean_loss = task_weight_vectors[task_index][0] * area_loss + task_weight_vectors[task_index][1] * delay_loss
        info = {
            "loss": mean_loss.item(),
            "q_values": state_action_values.detach().cpu().numpy(),
            "target_q_values": target_state_action_values.detach().cpu().numpy(),
            "is_update": True,
            "int_rewards": int_rewards_batch.cpu().numpy(),
            "area_rewards": area_reward_batch.cpu().numpy(),
            "delay_rewards": delay_reward_batch.cpu().numpy(),
            "target_q_area_values": target_state_action_area_values.detach().cpu().numpy(),
            "target_q_delay_values": target_state_action_delay_values.detach().cpu().numpy()
        }

    return mean_loss, info

class AsyncMultiTaskDQNAlgorithm():
    def __init__(
        self,
        q_policy,
        replay_memory,
        rnd_predictor,
        rnd_target,
        int_reward_run_mean_std,
        seed,
        # meta-agent
        meta_agent,
        # rnd kwargs 
        rnd_lr=3e-4,
        update_rnd_freq=10,
        int_reward_scale=1,
        rnd_reset_freq=20,
        # evaluate kwargs
        evaluate_freq=5,
        evaluate_num=5,
        # multi task kwargs
        task_weight_vectors=[[4,1],[3,2],[2,3],[1,4]],
        target_delay=[
            [50,200,500,1200], [50,200,500,1200],
            [50,200,500,1200], [50,200,500,1200]
        ],
        # meta_agent_kwargs
        meta_agent_optimizer_class='Adam',
        meta_agent_lr=1e-3,
        meta_action_num=3, # 3 or 2
        # dqn kwargs
        optimizer_class='Adam',
        q_net_lr=1e-4,
        batch_size=64,
        gamma=0.8,
        len_per_episode=25,
        total_episodes=400,
        target_update_freq=10,
        MAX_STAGE_NUM=4,
        device='cpu',
        action_num=4,
        loss_type="mix",
        # buffer sharing
        is_buffer_sharing=True,
        is_buffer_sharing_ablation=True,
        # bit width
        bit_width="16_bits",
        int_bit_width=16,
        str_bit_width=16,
        # pareto
        reference_point=[2600,1.8],
        # adaptive multi-task
        adaptive_multi_task_type="heuristics",
        meta_agent_type="learning",
        start_adaptive_episodes=50,
        update_tasks_freq=8,
        weight_bias=[0.75,0.5,0.25,0.],
        delay_weight=[500,400,400,150],
        delay_bias=[1000,600,200,50],
        delay_cons=[[1500,1200],[1100,800],[700,200],[100,50]],
        weight_cons=[[5,3.75],[3.75,2.5],[2.5,1.25],[1.25,0]],
        is_target_delay_change=True,
        is_weight_change=True,
        # vector condition q
        is_vector_condition_q=False,
        max_target_delay=[1500,1000,500,100],
        is_parallel=False,
        is_weight_loss=False,
        load_state_pool_path=None,
        # gomil
        gomil_area=1936,
        gomil_delay=1.35,
        load_gomil=True,
        # end experiments
        end_exp_freq=25,
        # env kwargs
        env_kwargs={}
    ):
        # 1. 处理好启动并行线程需要的共享变量
        # 2. run experiments，并行启动多进程，多线程需要能调用gpu，多线程传入的参数不一样，其他执行的程序是一样的，要上锁
        # 3. 按照一个episode 一个episode来，每次并行启动一个episode？ 统一更新RND model? 统一分配weight vectors？

        self.q_policy = q_policy
        self.replay_memory = replay_memory
        self.rnd_predictor = rnd_predictor
        self.rnd_target = rnd_target
        self.int_reward_run_mean_std = int_reward_run_mean_std
        self.seed = seed
        self.meta_agent = meta_agent.to(device[0])
        # meta_agent_kwargs
        self.meta_agent_optimizer_class = meta_agent_optimizer_class
        self.meta_agent_lr = meta_agent_lr
        # get meta_agent optimizer
        if isinstance(meta_agent_optimizer_class, str):
            meta_agent_optimizer_class = eval('optim.' + meta_agent_optimizer_class)
        self.meta_agent_optimizer = meta_agent_optimizer_class(
            self.meta_agent.parameters(),
            lr=self.meta_agent_lr
        )
        self.meta_agent_type = meta_agent_type
        # rnd optimizer
        self.rnd_model_optimizer = meta_agent_optimizer_class(
            self.rnd_predictor.parameters(),
            lr=rnd_lr
        )
        self.meta_action_num = meta_action_num
        # kwargs
        self.rnd_lr = rnd_lr
        self.update_rnd_freq = update_rnd_freq
        self.rnd_reset_freq = rnd_reset_freq
        self.int_reward_scale = int_reward_scale
        self.evaluate_freq = evaluate_freq
        self.evaluate_num = evaluate_num
        self.task_weight_vectors = task_weight_vectors
        self.target_delay = target_delay
        self.total_steps = [mp.Manager().Value("i", 0) for _ in range(len(task_weight_vectors))]
        self.optimizer_class = optimizer_class
        self.q_net_lr = q_net_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.len_per_episode = len_per_episode
        self.total_episodes = total_episodes
        self.target_update_freq = target_update_freq
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.device = device
        self.action_num = action_num
        self.loss_type = loss_type
        self.is_buffer_sharing = is_buffer_sharing
        self.is_buffer_sharing_ablation = is_buffer_sharing_ablation
        self.start_adaptive_episodes = start_adaptive_episodes
        self.adaptive_multi_task_type = adaptive_multi_task_type
        self.weight_bias = weight_bias
        self.delay_weight = delay_weight
        self.delay_bias = delay_bias
        self.is_target_delay_change = is_target_delay_change
        self.is_weight_change = is_weight_change
        self.update_tasks_freq = update_tasks_freq
        self.delay_cons = delay_cons
        self.weight_cons = weight_cons
        # vector condition q
        self.is_vector_condition_q = is_vector_condition_q
        self.max_target_delay = max_target_delay
        self.is_parallel = is_parallel
        self.is_weight_loss = is_weight_loss

        self.bit_width = bit_width
        self.int_bit_width = int_bit_width
        self.initial_partial_product = PartialProduct[self.bit_width][:-1]

        # gomil
        self.gomil_area = gomil_area
        self.gomil_delay = gomil_delay
        self.load_gomil = load_gomil
        self.end_exp_freq = end_exp_freq

        # load pool
        self.load_state_pool_path = load_state_pool_path

        self.env_kwargs = env_kwargs

        self.loss_fn = nn.SmoothL1Loss()
        self.rnd_loss = nn.MSELoss()    

        # # pareto pointset
        self.pareto_pointset = {
            "area": [],
            "delay": []
        }
        self.reference_point = reference_point
        logger.log(f"reference_point: {self.reference_point}")
        self.current_hv_value = None

        ## meta-agent datasets
        self.meta_agent_datasets = {
            "state": None,
            "action": None,
            "hypervolume": None
        }

    @staticmethod
    def run_episode_vector_conditionq_v3(task_index, task_weight_vectors, shared_q_policy, replay_memory, env, total_steps, loss_fn, rnd_predictor, rnd_target, int_reward_run_mean_std, lock, **kwargs):
        log_info = {
            "reward": [],
            "loss": [],
            "q_values": [],
            "target_q_values": [],
            "avg_ppa": [],
            "task_index": task_index,
            "eps_threshold": [],
            "area": [],
            "delay": [],
            "int_rewards": []
        }
        # 1. get optimizer
        if isinstance(kwargs["optimizer_class"], str):
            optimizer_class = eval('optim.'+kwargs["optimizer_class"])
        q_policy_optimizer = optimizer_class(
            shared_q_policy.parameters(),
            lr=kwargs["q_net_lr"]
        )
        # 2. copy q policy 
        q_policy = copy.deepcopy(shared_q_policy)
        q_policy.device = kwargs["device"]
        env.task_index = task_index
        q_policy.task_index = task_index
        target_q_policy = copy.deepcopy(q_policy)
        q_policy.to(kwargs["device"])
        target_q_policy.to(kwargs["device"])
        rnd_predictor.to(kwargs["device"])
        rnd_predictor.device = kwargs["device"]
        rnd_target.to(kwargs["device"])
        rnd_target.device = kwargs["device"]
        q_kwargs = {
            "batch_size": kwargs["batch_size"],
            "device": kwargs["device"],
            "initial_partial_product": kwargs["initial_partial_product"],
            "MAX_STAGE_NUM": kwargs["MAX_STAGE_NUM"],
            "int_bit_width": kwargs["int_bit_width"],
            "gamma": kwargs["gamma"],
            "loss_type": kwargs["loss_type"],
            "int_reward_scale": kwargs["int_reward_scale"]
        }
        # 3. run an episode
        # init state 
        env_state, sel_index = env.reset()
        state = copy.deepcopy(env_state)
        
        # newly added preference condition
        weight_condition = np.array(task_weight_vectors[task_index])
        delay_condition = env.target_delay[0] / env.max_target_delay

        for step in range(kwargs["len_per_episode"]):
            print(f"task index: {task_index}, steps: {step}")
            # logger.log(f"task index: {task_index}, steps: {step}")
            # environment interaction
            total_steps.value += 1
            action, policy_info = q_policy.select_action(
                torch.tensor(state), total_steps.value, task_index,
                weight_condition, delay_condition
            )
            next_state, reward, rewards_dict = env.step(action)
            _, next_state_policy_info = q_policy.select_action(
                torch.tensor(next_state), total_steps.value, task_index,
                weight_condition, delay_condition
            )
            # store data to replay buffer
            store_state = np.reshape(state, (1,2,int(kwargs["int_bit_width"]*2)))
            store_next_state = np.reshape(next_state, (1,2,int(kwargs["int_bit_width"]*2)))
            # shared replay memory via area reward and delay reward
            replay_memory.push(
                store_state,
                action.cpu().numpy(),
                store_next_state,
                np.array([reward]),
                policy_info["mask"].reshape(1,-1).cpu().numpy(),
                next_state_policy_info["mask"].reshape(1,-1).cpu().numpy(),
                np.array([rewards_dict["area_reward"]]),
                np.array([rewards_dict["delay_reward"]]),
                weight_condition,
                np.array([delay_condition])
            )
            # update initial state pool
            # TODO: environment 得补充下这个函数
            # TODO: 环境里面的found best area delay 是取平均值的，得修改下；
            env.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            
            # Sync local model with shared model
            q_policy.load_state_dict(shared_q_policy.state_dict())
            
            # update q policy
            # TODO: add args/kwargs 调整kwargs
            # comments: compute_q_loss 计算 q 损失相应修改下；
            q_loss, loss_info = compute_q_loss_vector_conditionq_parallel_weight_loss(
                replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std,
                loss_fn, task_weight_vectors, task_index, **q_kwargs)

            if loss_info["is_update"]:
                # update shared model
                q_loss.backward()
                for param in q_policy.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                # The critical section begins
                lock.acquire()
                shared_q_policy.zero_grad()
                copy_gradients(shared_q_policy, q_policy)
                q_policy_optimizer.step()
                lock.release()
                # The critical section ends
                q_policy.zero_grad()
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(q_loss.item())
                log_info["q_values"].append(np.mean(loss_info["q_values"]))
                log_info["target_q_values"].append(np.mean(loss_info["target_q_values"]))
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(np.mean(loss_info["int_rewards"]))
            else:
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(0)
                log_info["q_values"].append(0)
                log_info["target_q_values"].append(0)
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(0)

            log_info["area"].append(np.mean(rewards_dict["area"]))
            log_info["delay"].append(np.mean(rewards_dict["delay"]))
                        
            state = copy.deepcopy(next_state)

            # update target q (TODO: SOFT UPDATE)
            if total_steps.value % kwargs["target_update_freq"] == 0:
                target_q_policy.load_state_dict(
                    q_policy.state_dict()
                )
        
        found_best_ppa = env.found_best_info["found_best_ppa"].value
        logger.log(f'run episode task index {task_index} found best ppa: {found_best_ppa}')
        return log_info

    @staticmethod
    def run_episode_vector_conditionq_v2(task_index, task_weight_vectors, shared_q_policy, replay_memory, env, total_steps, loss_fn, rnd_predictor, rnd_target, int_reward_run_mean_std, lock, **kwargs):
        log_info = {
            "reward": [],
            "loss": [],
            "q_values": [],
            "target_q_values": [],
            "avg_ppa": [],
            "task_index": task_index,
            "eps_threshold": [],
            "area": [],
            "delay": [],
            "int_rewards": [],
            "area_rewards": [],
            "delay_rewards": [],
            "target_q_area_values": [],
            "target_q_delay_values": []
        }
        # 1. get optimizer
        if isinstance(kwargs["optimizer_class"], str):
            optimizer_class = eval('optim.'+kwargs["optimizer_class"])
        q_policy_optimizer = optimizer_class(
            shared_q_policy.parameters(),
            lr=kwargs["q_net_lr"]
        )
        # 2. copy q policy 
        q_policy = copy.deepcopy(shared_q_policy)
        q_policy.device = kwargs["device"]
        env.task_index = task_index
        q_policy.task_index = task_index
        target_q_policy = copy.deepcopy(q_policy)
        q_policy.to(kwargs["device"])
        target_q_policy.to(kwargs["device"])
        rnd_predictor.to(kwargs["device"])
        rnd_predictor.device = kwargs["device"]
        rnd_target.to(kwargs["device"])
        rnd_target.device = kwargs["device"]
        q_kwargs = {
            "batch_size": kwargs["batch_size"],
            "device": kwargs["device"],
            "initial_partial_product": kwargs["initial_partial_product"],
            "MAX_STAGE_NUM": kwargs["MAX_STAGE_NUM"],
            "int_bit_width": kwargs["int_bit_width"],
            "gamma": kwargs["gamma"],
            "loss_type": kwargs["loss_type"],
            "int_reward_scale": kwargs["int_reward_scale"]
        }
        # 3. run an episode
        # init state 
        env_state, sel_index = env.reset()
        state = copy.deepcopy(env_state)
        
        # newly added preference condition
        weight_condition = np.array(task_weight_vectors[task_index])
        delay_condition = env.target_delay[0] / env.max_target_delay

        for step in range(kwargs["len_per_episode"]):
            print(f"task index: {task_index}, steps: {step}")
            # logger.log(f"task index: {task_index}, steps: {step}")
            # environment interaction
            total_steps.value += 1
            action, policy_info = q_policy.select_action(
                torch.tensor(state), total_steps.value, task_index,
                weight_condition, delay_condition
            )
            next_state, reward, rewards_dict = env.step(action)
            _, next_state_policy_info = q_policy.select_action(
                torch.tensor(next_state), total_steps.value, task_index,
                weight_condition, delay_condition
            )
            # store data to replay buffer
            store_state = np.reshape(state, (1,2,int(kwargs["int_bit_width"]*2)))
            store_next_state = np.reshape(next_state, (1,2,int(kwargs["int_bit_width"]*2)))
            # shared replay memory via area reward and delay reward
            replay_memory.push(
                store_state,
                action.cpu().numpy(),
                store_next_state,
                np.array([reward]),
                policy_info["mask"].reshape(1,-1).cpu().numpy(),
                next_state_policy_info["mask"].reshape(1,-1).cpu().numpy(),
                np.array([rewards_dict["area_reward"]]),
                np.array([rewards_dict["delay_reward"]]),
                weight_condition,
                np.array([delay_condition])
            )
            # update initial state pool
            # TODO: environment 得补充下这个函数
            # TODO: 环境里面的found best area delay 是取平均值的，得修改下；
            env.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            
            # Sync local model with shared model
            q_policy.load_state_dict(shared_q_policy.state_dict())
            
            # update q policy
            # TODO: add args/kwargs 调整kwargs
            # comments: compute_q_loss 计算 q 损失相应修改下；
            if kwargs["is_parallel"]:
                q_loss, loss_info = compute_q_loss_vector_conditionq_parallel(
                    replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std,
                    loss_fn, task_weight_vectors, task_index, **q_kwargs)
            else:
                q_loss, loss_info = compute_q_loss_vector_conditionq(
                    replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std,
                    loss_fn, task_weight_vectors, task_index, **q_kwargs)
            if loss_info["is_update"]:
                # update shared model
                q_loss.backward()
                for param in q_policy.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                # The critical section begins
                lock.acquire()
                shared_q_policy.zero_grad()
                copy_gradients(shared_q_policy, q_policy)
                q_policy_optimizer.step()
                lock.release()
                # The critical section ends
                q_policy.zero_grad()
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(q_loss.item())
                log_info["q_values"].append(np.mean(loss_info["q_values"]))
                log_info["target_q_values"].append(np.mean(loss_info["target_q_values"]))
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(np.mean(loss_info["int_rewards"]))
                
                log_info["area_rewards"].append(np.mean(loss_info["area_rewards"]))
                log_info["delay_rewards"].append(np.mean(loss_info["delay_rewards"]))
                log_info["target_q_area_values"].append(np.mean(loss_info["target_q_area_values"]))
                log_info["target_q_delay_values"].append(np.mean(loss_info["target_q_delay_values"]))
                
            else:
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(0)
                log_info["q_values"].append(0)
                log_info["target_q_values"].append(0)
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(0)
                log_info["area_rewards"].append(0)
                log_info["delay_rewards"].append(0)
                log_info["target_q_area_values"].append(0)
                log_info["target_q_delay_values"].append(0)
                

            log_info["area"].append(np.mean(rewards_dict["area"]))
            log_info["delay"].append(np.mean(rewards_dict["delay"]))
                        
            state = copy.deepcopy(next_state)

            # update target q (TODO: SOFT UPDATE)
            if total_steps.value % kwargs["target_update_freq"] == 0:
                target_q_policy.load_state_dict(
                    q_policy.state_dict()
                )
        
        found_best_ppa = env.found_best_info["found_best_ppa"].value
        logger.log(f'run episode task index {task_index} found best ppa: {found_best_ppa}')
        return log_info

    @staticmethod
    def run_episode_vector_conditionq(task_index, task_weight_vectors, shared_q_policy, replay_memory, env, total_steps, loss_fn, rnd_predictor, rnd_target, int_reward_run_mean_std, lock, **kwargs):
        log_info = {
            "reward": [],
            "loss": [],
            "q_values": [],
            "target_q_values": [],
            "avg_ppa": [],
            "task_index": task_index,
            "eps_threshold": [],
            "area": [],
            "delay": [],
            "int_rewards": []
        }
        # 1. get optimizer
        if isinstance(kwargs["optimizer_class"], str):
            optimizer_class = eval('optim.'+kwargs["optimizer_class"])
        q_policy_optimizer = optimizer_class(
            shared_q_policy.parameters(),
            lr=kwargs["q_net_lr"]
        )
        # 2. copy q policy 
        q_policy = copy.deepcopy(shared_q_policy)
        q_policy.device = kwargs["device"]
        env.task_index = task_index
        q_policy.task_index = task_index
        target_q_policy = copy.deepcopy(q_policy)
        q_policy.to(kwargs["device"])
        target_q_policy.to(kwargs["device"])
        rnd_predictor.to(kwargs["device"])
        rnd_predictor.device = kwargs["device"]
        rnd_target.to(kwargs["device"])
        rnd_target.device = kwargs["device"]
        q_kwargs = {
            "batch_size": kwargs["batch_size"],
            "device": kwargs["device"],
            "initial_partial_product": kwargs["initial_partial_product"],
            "MAX_STAGE_NUM": kwargs["MAX_STAGE_NUM"],
            "int_bit_width": kwargs["int_bit_width"],
            "gamma": kwargs["gamma"],
            "loss_type": kwargs["loss_type"],
            "int_reward_scale": kwargs["int_reward_scale"]
        }
        # 3. run an episode
        # init state 
        env_state, sel_index = env.reset()
        state = copy.deepcopy(env_state)
        
        # newly added preference condition
        weight_condition = np.array(task_weight_vectors[task_index])
        delay_condition = env.target_delay[0] / env.max_target_delay

        for step in range(kwargs["len_per_episode"]):
            print(f"task index: {task_index}, steps: {step}")
            # logger.log(f"task index: {task_index}, steps: {step}")
            # environment interaction
            total_steps.value += 1
            action, policy_info = q_policy.select_action(
                torch.tensor(state), total_steps.value, task_index,
                weight_condition, delay_condition
            )
            next_state, reward, rewards_dict = env.step(action)
            _, next_state_policy_info = q_policy.select_action(
                torch.tensor(next_state), total_steps.value, task_index,
                weight_condition, delay_condition
            )
            # store data to replay buffer
            store_state = np.reshape(state, (1,2,int(kwargs["int_bit_width"]*2)))
            store_next_state = np.reshape(next_state, (1,2,int(kwargs["int_bit_width"]*2)))
            # shared replay memory via area reward and delay reward
            replay_memory.push(
                store_state,
                action.cpu().numpy(),
                store_next_state,
                np.array([reward]),
                policy_info["mask"].reshape(1,-1).cpu().numpy(),
                next_state_policy_info["mask"].reshape(1,-1).cpu().numpy(),
                np.array([rewards_dict["area_reward"]]),
                np.array([rewards_dict["delay_reward"]]),
                weight_condition,
                np.array([delay_condition])
            )
            # update initial state pool
            # TODO: environment 得补充下这个函数
            # TODO: 环境里面的found best area delay 是取平均值的，得修改下；
            env.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            
            # Sync local model with shared model
            q_policy.load_state_dict(shared_q_policy.state_dict())
            
            # update q policy
            # TODO: add args/kwargs 调整kwargs
            # comments: compute_q_loss 计算 q 损失相应修改下；
            q_loss, loss_info = compute_q_loss_vector_conditionq(
                replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std,
                loss_fn, task_weight_vectors, task_index, **q_kwargs)
            if loss_info["is_update"]:
                # update shared model
                q_loss.backward()
                for param in q_policy.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                # The critical section begins
                lock.acquire()
                shared_q_policy.zero_grad()
                copy_gradients(shared_q_policy, q_policy)
                q_policy_optimizer.step()
                lock.release()
                # The critical section ends
                q_policy.zero_grad()
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(q_loss.item())
                log_info["q_values"].append(np.mean(loss_info["q_values"]))
                log_info["target_q_values"].append(np.mean(loss_info["target_q_values"]))
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(np.mean(loss_info["int_rewards"]))
            else:
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(0)
                log_info["q_values"].append(0)
                log_info["target_q_values"].append(0)
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(0)

            log_info["area"].append(np.mean(rewards_dict["area"]))
            log_info["delay"].append(np.mean(rewards_dict["delay"]))
                        
            state = copy.deepcopy(next_state)

            # update target q (TODO: SOFT UPDATE)
            if total_steps.value % kwargs["target_update_freq"] == 0:
                target_q_policy.load_state_dict(
                    q_policy.state_dict()
                )
        
        found_best_ppa = env.found_best_info["found_best_ppa"].value
        logger.log(f'run episode task index {task_index} found best ppa: {found_best_ppa}')
        return log_info

    @staticmethod
    def run_episode(task_index, task_weight_vectors, shared_q_policy, replay_memory, env, total_steps, loss_fn, rnd_predictor, rnd_target, int_reward_run_mean_std, lock, **kwargs):
        log_info = {
            "reward": [],
            "loss": [],
            "q_values": [],
            "target_q_values": [],
            "avg_ppa": [],
            "task_index": task_index,
            "eps_threshold": [],
            "area": [],
            "delay": [],
            "int_rewards": []
        }
        # 1. get optimizer
        if isinstance(kwargs["optimizer_class"], str):
            optimizer_class = eval('optim.'+kwargs["optimizer_class"])
        q_policy_optimizer = optimizer_class(
            shared_q_policy.parameters(),
            lr=kwargs["q_net_lr"]
        )
        # 2. copy q policy 
        q_policy = copy.deepcopy(shared_q_policy)
        q_policy.device = kwargs["device"]
        env.task_index = task_index
        q_policy.task_index = task_index
        target_q_policy = copy.deepcopy(q_policy)
        q_policy.to(kwargs["device"])
        target_q_policy.to(kwargs["device"])
        rnd_predictor.to(kwargs["device"])
        rnd_predictor.device = kwargs["device"]
        rnd_target.to(kwargs["device"])
        rnd_target.device = kwargs["device"]
        q_kwargs = {
            "batch_size": kwargs["batch_size"],
            "device": kwargs["device"],
            "initial_partial_product": kwargs["initial_partial_product"],
            "MAX_STAGE_NUM": kwargs["MAX_STAGE_NUM"],
            "int_bit_width": kwargs["int_bit_width"],
            "gamma": kwargs["gamma"],
            "loss_type": kwargs["loss_type"],
            "int_reward_scale": kwargs["int_reward_scale"]
        }
        # 3. run an episode
        # init state 
        env_state, sel_index = env.reset()
        state = copy.deepcopy(env_state)

        for step in range(kwargs["len_per_episode"]):
            print(f"task index: {task_index}, steps: {step}")
            # logger.log(f"task index: {task_index}, steps: {step}")
            # environment interaction
            total_steps.value += 1
            action, policy_info = q_policy.select_action(
                torch.tensor(state), total_steps.value, task_index
            )
            next_state, reward, rewards_dict = env.step(action)
            _, next_state_policy_info = q_policy.select_action(
                torch.tensor(next_state), total_steps.value, task_index
            )
            # store data to replay buffer
            store_state = np.reshape(state, (1,2,int(kwargs["int_bit_width"]*2)))
            store_next_state = np.reshape(next_state, (1,2,int(kwargs["int_bit_width"]*2)))
            # shared replay memory via area reward and delay reward
            replay_memory.push(
                store_state,
                action.cpu().numpy(),
                store_next_state,
                np.array([reward]),
                policy_info["mask"].reshape(1,-1).cpu().numpy(),
                next_state_policy_info["mask"].reshape(1,-1).cpu().numpy(),
                np.array([rewards_dict["area_reward"]]),
                np.array([rewards_dict["delay_reward"]])
            )
            # update initial state pool
            # TODO: environment 得补充下这个函数
            # TODO: 环境里面的found best area delay 是取平均值的，得修改下；
            env.update_env_initial_state_pool(next_state, rewards_dict, next_state_policy_info['mask'])
            
            # Sync local model with shared model
            q_policy.load_state_dict(shared_q_policy.state_dict())
            
            # update q policy
            # TODO: add args/kwargs 调整kwargs
            q_loss, loss_info = compute_q_loss(
                replay_memory, env, q_policy, target_q_policy, rnd_predictor, rnd_target, int_reward_run_mean_std,
                loss_fn, task_weight_vectors, task_index, **q_kwargs)
            if loss_info["is_update"]:
                # update shared model
                q_loss.backward()
                for param in q_policy.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                # The critical section begins
                lock.acquire()
                shared_q_policy.zero_grad()
                copy_gradients(shared_q_policy, q_policy)
                q_policy_optimizer.step()
                lock.release()
                # The critical section ends
                q_policy.zero_grad()
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(q_loss.item())
                log_info["q_values"].append(np.mean(loss_info["q_values"]))
                log_info["target_q_values"].append(np.mean(loss_info["target_q_values"]))
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(np.mean(loss_info["int_rewards"]))
            else:
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(0)
                log_info["q_values"].append(0)
                log_info["target_q_values"].append(0)
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
                log_info["int_rewards"].append(0)

            log_info["area"].append(np.mean(rewards_dict["area"]))
            log_info["delay"].append(np.mean(rewards_dict["delay"]))
                        
            state = copy.deepcopy(next_state)

            # update target q (TODO: SOFT UPDATE)
            if total_steps.value % kwargs["target_update_freq"] == 0:
                target_q_policy.load_state_dict(
                    q_policy.state_dict()
                )
        
        found_best_ppa = env.found_best_info["found_best_ppa"].value
        logger.log(f'run episode task index {task_index} found best ppa: {found_best_ppa}')
        return log_info
    
    def _get_initial_envs(self, n_processing):
        envs = []
        for i in range(n_processing):
            env = SerialRefineEnv(
                self.seed, self.q_policy,
                weight_area=self.task_weight_vectors[i][0],
                weight_delay=self.task_weight_vectors[i][1],
                task_index=i, target_delay=self.target_delay[i],
                max_target_delay=self.max_target_delay[i],
                load_state_pool_path=self.load_state_pool_path,
                pool_index=i,
                gomil_area=self.gomil_area,
                gomil_delay=self.gomil_delay,
                load_gomil=self.load_gomil,
                **self.env_kwargs
            )
            envs.append(env)
        return envs

    def _get_episode_kwargs(self, task_index):
        run_episode_kwargs = {
            "optimizer_class": self.optimizer_class,
            "batch_size": self.batch_size,
            "q_net_lr": self.q_net_lr,
            "gamma": self.gamma,
            "len_per_episode": self.len_per_episode,
            "target_update_freq": self.target_update_freq,
            "device": self.device[task_index],
            "int_bit_width": self.int_bit_width,
            "initial_partial_product": self.initial_partial_product,
            "MAX_STAGE_NUM": self.MAX_STAGE_NUM,
            "loss_type": self.loss_type,
            "int_reward_scale": self.int_reward_scale
        }
        return run_episode_kwargs

    def _get_area_delay(self, log_infos):
        area_list = []
        delay_list = []
        for i in range(len(log_infos)):
            area_list.extend(log_infos[i]["area"])
            delay_list.extend(log_infos[i]["delay"])
        return area_list, delay_list

    def _combine(self):
        combine_array = []
        for i in range(len(self.pareto_pointset["area"])):
            point = [self.pareto_pointset["area"][i], self.pareto_pointset["delay"][i]]
            combine_array.append(point)
        return np.array(combine_array)

    def process_and_log_pareto(self, episode_num, log_infos):
        # 1. compute pareto pointset
        area_list, delay_list = self._get_area_delay(log_infos)
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
    
        return hv_value
            
    def log_global_information(self, episode_num, log_infos, envs):
        # log q value stats
        for i in range(len(log_infos)):
            task_index = log_infos[i]["task_index"]
            logger.tb_logger.add_scalar(f'task index {task_index} reward', np.mean(log_infos[i]["reward"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} avg ppa', np.mean(log_infos[i]["avg_ppa"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} loss', np.mean(log_infos[i]["loss"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} q_values', np.mean(log_infos[i]["q_values"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} target_q_values', np.mean(log_infos[i]["target_q_values"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} eps_threshold', np.mean(log_infos[i]["eps_threshold"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} int_rewards', np.mean(log_infos[i]["int_rewards"]), global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} total_steps', self.total_steps[task_index].value, global_step=episode_num)
            if "area_rewards" in log_infos[i].keys():
                logger.tb_logger.add_scalar(f'task index {task_index} area_rewards', np.mean(log_infos[i]["area_rewards"]), global_step=episode_num)
                logger.tb_logger.add_scalar(f'task index {task_index} delay_rewards', np.mean(log_infos[i]["delay_rewards"]), global_step=episode_num)
                logger.tb_logger.add_scalar(f'task index {task_index} target_q_area_values', np.mean(log_infos[i]["target_q_area_values"]), global_step=episode_num)
                logger.tb_logger.add_scalar(f'task index {task_index} target_q_delay_values', np.mean(log_infos[i]["target_q_delay_values"]), global_step=episode_num)
            # log env stats
            env = envs[task_index]
            logger.tb_logger.add_scalar(f'task index {task_index} found best ppa', env.found_best_info["found_best_ppa"].value, global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} found best area', env.found_best_info["found_best_area"].value, global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} found best delay', env.found_best_info["found_best_delay"].value, global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} env state pool len', len(env.initial_state_pool), global_step=episode_num)
            
        for i, env in enumerate(envs):
            found_best_ppa = env.found_best_info['found_best_ppa'].value
            logger.log(f'global task index {i} found best ppa: {found_best_ppa}')
            logger.log(f'global task index {i} env state pool len: {len(env.initial_state_pool)}')

        # pareto process
        hv_value = self.process_and_log_pareto(episode_num, log_infos)

        return hv_value

    def _log_modified_weight(self, envs, episode_num):
        for task_index in range(len(self.task_weight_vectors)):
            logger.tb_logger.add_scalar(f'task index {task_index} weight area', self.task_weight_vectors[task_index][0], global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} weight delay', self.task_weight_vectors[task_index][1], global_step=episode_num)
            
            logger.tb_logger.add_scalar(f'task index {task_index} env weight area', envs[task_index].weight_area, global_step=episode_num)
            logger.tb_logger.add_scalar(f'task index {task_index} env weight delay', envs[task_index].weight_delay, global_step=episode_num)
            
            logger.tb_logger.add_scalar(f'task index {task_index} env target delay', envs[task_index].target_delay[0], global_step=episode_num)

    def _train_meta_agent(self, episode_num):
        device = self.device[0]
        # process data
        state = torch.tensor(self.meta_agent_datasets["state"]).unsqueeze(0).float().to(device)
        action = self.meta_agent_datasets["action"]
        reward = self.current_hv_value - self.meta_agent_datasets["hypervolume"]
        # compute loss 
        probs, logits = self.meta_agent(state)
        dist = Categorical(probs.squeeze())
        logp = dist.log_prob(action)
        reinforce_loss = -1. * reward * logp
        # update meta agent
        self.meta_agent_optimizer.zero_grad()
        reinforce_loss.backward()
        for param in self.meta_agent.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_agent_optimizer.step()
        # log reward and loss
        logger.tb_logger.add_scalar('hypervolume reward', reward, global_step=episode_num)
        logger.tb_logger.add_scalar('reinforce loss', reinforce_loss.item(), global_step=episode_num)

    def execute_meta_agent(self, envs, episode_num):
        if self.meta_agent_type == "learning":
            # 1. train the meta agent
            if episode_num > self.start_adaptive_episodes:
                self._train_meta_agent(episode_num)
            # 2. sample action from the agent
            meta_state = self.get_meta_state(envs)
            device = self.device[0]
            with torch.no_grad():
                state = torch.tensor(meta_state).unsqueeze(0).float().to(device)
                probs, logits = self.meta_agent(state)
                dist = Categorical(probs.squeeze())
                action = dist.sample()
            ## store datasets
            self.meta_agent_datasets["state"] = meta_state
            self.meta_agent_datasets["action"] = action
            self.meta_agent_datasets["hypervolume"] = self.current_hv_value
            logger.tb_logger.add_histogram('meta agent action probs', probs, global_step=episode_num)
        
        elif self.meta_agent_type == "random":
            action = random.sample([i for i in range(4 * self.meta_action_num)],1)[0]

        # 3. execute the action 
        ## action_code: 0 加档，1减档，2不动
        action_region = int(action / self.meta_action_num)
        action_type = int(action % self.meta_action_num)
        envs = self._execute_action(action_region, action_type, envs)
        # log action
        logger.tb_logger.add_scalar('action region', action_region, global_step=episode_num)
        logger.tb_logger.add_scalar('action type', action_type, global_step=episode_num)
        
        return envs
    
    def get_meta_state(self, envs):
        meta_state = []
        # delay heuristics
        current_pareto_delay = self.pareto_pointset["delay"]
        num_tasks = len(envs)
        # 1. 取max min delay
        max_delay = np.max(current_pareto_delay)
        min_delay = np.min(current_pareto_delay)
        delay_interval = (max_delay - min_delay) / num_tasks
        # 2. 计算四个delay区间内的pareto点数, delay 从高到低排，从而target delay 从高到低排
        num_pareto = [0 for _ in range(num_tasks)]
        for delay in current_pareto_delay:
            for i in range(num_tasks):
                if delay <= (max_delay - i * delay_interval) and delay > (max_delay - (i+1) * delay_interval):
                    num_pareto[i] += 1
        for num in num_pareto:
            meta_state.append(num)
        
        # area heuristics
        current_pareto_area = self.pareto_pointset["area"]
        num_tasks = len(envs)
        # 1. 取max min delay
        max_area = np.max(current_pareto_area)
        min_area = np.min(current_pareto_area)
        area_interval = (max_area - min_area) / num_tasks
        # 2. 计算四个delay区间内的pareto点数, delay 从高到低排，从而target delay 从高到低排
        num_pareto = [0 for _ in range(num_tasks)]
        for area in current_pareto_area:
            for i in range(num_tasks):
                if area >= (min_area + i * area_interval) and area <= (min_area + (i+1) * area_interval):
                    num_pareto[i] += 1
        for num in num_pareto:
            meta_state.append(num)
        return np.array(meta_state)

    def _execute_action(self, action_region, action_type, envs):
        # action type 0 1 2: 加档 减档 不动
        # 加档：target delay +50 weight +0.05
        # 减档：target delay -50 weight -0.05
        # four region ranges
            # s- area比例：0.8；0.6；0.4；0.2
            #   - 1-0.75：（0,0.75）(1,1) y = 0.25x+0.75; 5-3.75
            #   - 0.75-0.5：（0,0.5）（1,0.75）y = 0.25x+0.5; 3.75-2.5
            #   - 0.5-0.25: (0,0.25) (1,0.5) y=0.25x+0.25; 2.5-1.25
            #   - 0.25-0 y = 0.25x; 1.25-0
            # - target delay
            #   - 1500: 1500-1150
            #   - 800：1150-700
            #   - 200：700-200 
            #   - 50：200-50 
        if action_type == 0:
            # 加档
            # change target delay
            if (self.target_delay[action_region][0] + 50) <= self.delay_cons[action_region][0]:
                self.target_delay[action_region][0] += 50
            else:
                self.target_delay[action_region][0] = self.delay_cons[action_region][0]
            envs[action_region].target_delay = self.target_delay[action_region]
            # change weight
            if (self.task_weight_vectors[action_region][0] + 0.05) <= self.weight_cons[action_region][0]:
                self.task_weight_vectors[action_region][0] += 0.05
                self.task_weight_vectors[action_region][1] -= 0.05
            else:
                self.task_weight_vectors[action_region][0] = self.weight_cons[action_region][0]
                self.task_weight_vectors[action_region][1] = 5 - self.weight_cons[action_region][0]
            envs[action_region].weight_area = self.task_weight_vectors[action_region][0]
            envs[action_region].weight_delay = self.task_weight_vectors[action_region][1]
        
        elif action_type == 1:
            # 减档
            # change target delay
            if (self.target_delay[action_region][0] - 50) >= self.delay_cons[action_region][1]:
                self.target_delay[action_region][0] -= 50
            else:
                self.target_delay[action_region][0] = self.delay_cons[action_region][1]
            envs[action_region].target_delay = self.target_delay[action_region]
            # change weight
            if (self.task_weight_vectors[action_region][0] - 0.05) >= self.weight_cons[action_region][1]:
                self.task_weight_vectors[action_region][0] -= 0.05
                self.task_weight_vectors[action_region][1] += 0.05
            else:
                self.task_weight_vectors[action_region][0] = self.weight_cons[action_region][1]
                self.task_weight_vectors[action_region][1] = 5 - self.weight_cons[action_region][1]
            envs[action_region].weight_area = self.task_weight_vectors[action_region][0]
            envs[action_region].weight_delay = self.task_weight_vectors[action_region][1]
        
        return envs
    def _modify_task_assignment(self, envs, episode_num):
        # random strategy
        if self.adaptive_multi_task_type == "random":
            random_samples = [np.random.rand() for _ in range(len(envs))] # random sample from [0,1)
            # weight bias; delay weight; delay bias
            for i in range(len(random_samples)):
                random_sample = random_samples[i]
                if self.is_weight_change:
                    weight_area = (0.25 * random_sample + self.weight_bias[i]) * 5
                    weight_delay = 5 - weight_area
                    self.task_weight_vectors[i] = [weight_area, weight_delay]
                    envs[i].weight_area = weight_area
                    envs[i].weight_delay = weight_delay
                if self.is_target_delay_change:
                    target_delay = [int(self.delay_weight[i] * random_sample + self.delay_bias[i])]
                    envs[i].target_delay = target_delay
            
        # heuristics strategy
        elif self.adaptive_multi_task_type == "heuristics":
            # delay heuristics
            current_pareto_delay = self.pareto_pointset["delay"]
            num_tasks = len(envs)
            # 1. 取max min delay
            max_delay = np.max(current_pareto_delay)
            min_delay = np.min(current_pareto_delay)
            delay_interval = (max_delay - min_delay) / num_tasks
            # 2. 计算四个delay区间内的pareto点数, delay 从高到低排，从而target delay 从高到低排
            num_pareto = [0 for _ in range(num_tasks)]
            for delay in current_pareto_delay:
                for i in range(num_tasks):
                    if delay <= (max_delay - i * delay_interval) and delay > (max_delay - (i+1) * delay_interval):
                            num_pareto[i] += 1
            # 3. 找到点最少的，对delay相应调整,target delay 变化区间为100；weight area delay 呢以0.5 为区间？
            worse_task_index = np.argmin(num_pareto)
            for task_index in range(len(num_pareto)):
                logger.tb_logger.add_scalar(f'task index {task_index} num pareto point delay', num_pareto[task_index], global_step=episode_num)
            logger.tb_logger.add_scalar('worse task index delay', worse_task_index, global_step=episode_num)
            is_delay_change = 1
            if worse_task_index == 0:
                # change target delay 得是delta的变换
                if (self.target_delay[0][0] - 100) >= 1200:
                    self.target_delay[0][0] -= 100
                else:
                    self.target_delay[0][0] = 1200
                    is_delay_change = 0
                envs[0].target_delay = self.target_delay[0]
                # change weight
                if (self.task_weight_vectors[0][0] - 0.1) >= 3.5:
                    self.task_weight_vectors[0][0] -= 0.1
                    self.task_weight_vectors[0][1] += 0.1
                else:
                    self.task_weight_vectors[0][0] = 3.5
                    self.task_weight_vectors[0][1] = 1.5
                envs[0].weight_area = self.task_weight_vectors[0][0]
                envs[0].weight_delay = self.task_weight_vectors[0][1]
            elif worse_task_index == 1:
                # change target delay 
                if (self.target_delay[1][0] + 100) <= 1100:
                    self.target_delay[1][0] += 100
                else:
                    self.target_delay[1][0] = 1100
                    is_delay_change = 0
                envs[1].target_delay = self.target_delay[1]
                # change weight
                if (self.task_weight_vectors[1][0] + 0.1) <= 3.4:
                    self.task_weight_vectors[1][0] += 0.1
                    self.task_weight_vectors[1][1] -= 0.1
                else:
                    self.task_weight_vectors[1][0] = 3.4
                    self.task_weight_vectors[1][1] = 1.6
                envs[1].weight_area = self.task_weight_vectors[1][0]
                envs[1].weight_delay = self.task_weight_vectors[1][1]
            elif worse_task_index == 2:
                # change target delay 
                if (self.target_delay[2][0] + 100) <= 700:
                    self.target_delay[2][0] += 100
                else:
                    self.target_delay[2][0] = 700
                    is_delay_change = 0
                envs[2].target_delay = self.target_delay[2]
                # change weight
                if (self.task_weight_vectors[2][0] + 0.1) <= 3:
                    self.task_weight_vectors[2][0] += 0.1
                    self.task_weight_vectors[2][1] -= 0.1
                else:
                    self.task_weight_vectors[2][0] = 3
                    self.task_weight_vectors[2][1] = 2
                envs[2].weight_area = self.task_weight_vectors[2][0]
                envs[2].weight_delay = self.task_weight_vectors[2][1]
            elif worse_task_index == 3:
                # change target delay 
                if (self.target_delay[3][0] + 30) <= 170:
                    self.target_delay[3][0] += 30
                else:
                    self.target_delay[3][0] = 170
                    is_delay_change = 0
                envs[3].target_delay = self.target_delay[3]
                # change weight
                if (self.task_weight_vectors[3][0] + 0.1) <= 2:
                    self.task_weight_vectors[3][0] += 0.1
                    self.task_weight_vectors[3][1] -= 0.1
                else:
                    self.task_weight_vectors[3][0] = 2
                    self.task_weight_vectors[3][1] = 3
                envs[3].weight_area = self.task_weight_vectors[3][0]
                envs[3].weight_delay = self.task_weight_vectors[3][1]
            if is_delay_change == 0:
                # area heuristics
                current_pareto_area = self.pareto_pointset["area"]
                num_tasks = len(envs)
                # 1. 取max min delay
                max_area = np.max(current_pareto_area)
                min_area = np.min(current_pareto_area)
                area_interval = (max_area - min_area) / num_tasks
                # 2. 计算四个delay区间内的pareto点数, delay 从高到低排，从而target delay 从高到低排
                num_pareto = [0 for _ in range(num_tasks)]
                for area in current_pareto_area:
                    for i in range(num_tasks):
                        if area >= (min_area + i * area_interval) and area <= (min_area + (i+1) * area_interval):
                            num_pareto[i] += 1
                # 3. 找到点最少的，对delay相应调整,target delay 变化区间为100；weight area delay 呢以0.5 为区间？
                worse_task_index = np.argmin(num_pareto)
                for task_index in range(len(num_pareto)):
                    logger.tb_logger.add_scalar(f'task index {task_index} num pareto point area', num_pareto[task_index], global_step=episode_num)
                logger.tb_logger.add_scalar('worse task index area', worse_task_index, global_step=episode_num)
                    
                if worse_task_index == 0:
                    # change target delay 得是delta的变换
                    if (self.target_delay[0][0] - 100) >= 1200:
                        self.target_delay[0][0] -= 100
                    else:
                        self.target_delay[0][0] = 1200
                    envs[0].target_delay = self.target_delay[0]
                    # change weight
                    if (self.task_weight_vectors[0][0] - 0.1) >= 3.5:
                        self.task_weight_vectors[0][0] -= 0.1
                        self.task_weight_vectors[0][1] += 0.1
                    else:
                        self.task_weight_vectors[0][0] = 3.5
                        self.task_weight_vectors[0][1] = 1.5
                    envs[0].weight_area = self.task_weight_vectors[0][0]
                    envs[0].weight_delay = self.task_weight_vectors[0][1]
                elif worse_task_index == 1:
                    # change target delay 
                    if (self.target_delay[1][0] + 100) <= 1100:
                        self.target_delay[1][0] += 100
                    else:
                        self.target_delay[1][0] = 1100
                    envs[1].target_delay = self.target_delay[1]
                    # change weight
                    if (self.task_weight_vectors[1][0] + 0.1) <= 3.4:
                        self.task_weight_vectors[1][0] += 0.1
                        self.task_weight_vectors[1][1] -= 0.1
                    else:
                        self.task_weight_vectors[1][0] = 3.4
                        self.task_weight_vectors[1][1] = 1.6
                    envs[1].weight_area = self.task_weight_vectors[1][0]
                    envs[1].weight_delay = self.task_weight_vectors[1][1]
                elif worse_task_index == 2:
                    # change target delay 
                    if (self.target_delay[2][0] + 100) <= 700:
                        self.target_delay[2][0] += 100
                    else:
                        self.target_delay[2][0] = 700
                    envs[2].target_delay = self.target_delay[2]
                    # change weight
                    if (self.task_weight_vectors[2][0] + 0.1) <= 3:
                        self.task_weight_vectors[2][0] += 0.1
                        self.task_weight_vectors[2][1] -= 0.1
                    else:
                        self.task_weight_vectors[2][0] = 3
                        self.task_weight_vectors[2][1] = 2
                    envs[2].weight_area = self.task_weight_vectors[2][0]
                    envs[2].weight_delay = self.task_weight_vectors[2][1]
                elif worse_task_index == 3:
                    # change target delay 
                    if (self.target_delay[3][0] + 30) <= 170:
                        self.target_delay[3][0] += 30
                    else:
                        self.target_delay[3][0] = 170
                    envs[3].target_delay = self.target_delay[3]
                    # change weight
                    if (self.task_weight_vectors[3][0] + 0.1) <= 2:
                        self.task_weight_vectors[3][0] += 0.1
                        self.task_weight_vectors[3][1] -= 0.1
                    else:
                        self.task_weight_vectors[3][0] = 2
                        self.task_weight_vectors[3][1] = 3
                    envs[3].weight_area = self.task_weight_vectors[3][0]
                    envs[3].weight_delay = self.task_weight_vectors[3][1]
        # learning-based
        elif self.adaptive_multi_task_type == "learning":
            envs = self.execute_meta_agent(envs, episode_num)
        return envs

    def update_reward_int_run_mean_std(self, rewards):
        mean, std, count = np.mean(rewards), np.std(rewards), len(rewards)
        self.int_reward_run_mean_std.update_from_moments(
            mean, std**2, count
        )

    def update_rnd_model(self, episode_num, envs):
        logger.log(f"replay memory length: {len(self.replay_memory)}")
        if self.is_buffer_sharing_ablation:
            transitions = self.replay_memory.sample(self.batch_size)
        else:
            if len(self.replay_memory[0]) < self.batch_size:
                loss = 0
                rnd_info = {
                        "rnd_loss": 0,
                        "int_reward_mean": 0,
                        "int_reward_std": 0
                    }
                return loss, rnd_info
            else:
                transitions = self.replay_memory[0].sample(self.batch_size)
        batch = decode_transition(transitions)
        state_batch = torch.tensor(np.concatenate(batch["next_state"]))
        state_mask = torch.tensor(np.concatenate(batch["next_state_mask"]))
        # update int reward run mean std
        reward_batch = torch.tensor(np.concatenate(batch["reward"]))
        self.update_reward_int_run_mean_std(
            reward_batch.cpu().numpy()
        )
        # to gpu
        self.rnd_predictor.to(self.device[0])
        self.rnd_predictor.device = self.device[0]
        self.rnd_target.to(self.device[0])
        self.rnd_target.device = self.device[0]

        loss = torch.zeros(self.batch_size, device=self.device[0])
        for i in range(self.batch_size):
            ct32, ct22, pp, stage_num = envs[0].decompose_compressor_tree(self.initial_partial_product, state_batch[i].cpu().numpy())
            ct32 = torch.tensor(np.array([ct32]))
            ct22 = torch.tensor(np.array([ct22]))
            if stage_num < self.MAX_STAGE_NUM-1:
                zeros = torch.zeros(1, self.MAX_STAGE_NUM-1-stage_num, int(self.int_bit_width*2))
                ct32 = torch.cat((ct32, zeros), dim=1)
                ct22 = torch.cat((ct22, zeros), dim=1)
            state = torch.cat((ct32, ct22), dim=0)
            
            predict_value = self.rnd_predictor(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
            with torch.no_grad():
                target_value = self.rnd_target(state.unsqueeze(0).float(), state_mask=state_mask[i]).reshape((int(self.int_bit_width*2))*4)
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
        rnd_info = {
            "rnd_loss": loss.item(),
            "int_reward_mean": self.int_reward_run_mean_std.mean,
            "int_reward_std": self.int_reward_run_mean_std.var
        }
        # log rnd_info 
        logger.tb_logger.add_scalar('rnd loss', rnd_info["rnd_loss"], global_step=episode_num)
        logger.tb_logger.add_scalar('rnd int reward mean std mean', np.mean(rnd_info["int_reward_mean"]), global_step=episode_num)
        logger.tb_logger.add_scalar('rnd int reward mean std std', np.mean(rnd_info["int_reward_std"]), global_step=episode_num)
    
        # to cpu
        self.rnd_predictor.to("cpu")
        self.rnd_predictor.device = "cpu"
        self.rnd_target.to("cpu")
        self.rnd_target.device = "cpu"
        if self.rnd_reset_freq > 0 and episode_num % self.rnd_reset_freq == 0:
            self.rnd_target.partially_reset()
        return loss, rnd_info
    
    def run_experiments(self):
        # 共享：shared model replay memory 
        # 分配：
            # task weight vector
            # env 初始化
            # q policy --> task index
            # optimizer
            # 
        # Create a multiprocessing manager
        manager = mp.Manager()
        # Create a shared lock using the manager
        shared_lock = manager.Lock()
        # multiprocessing init
        n_processing = len(self.task_weight_vectors)
        # TODO: we can add task weight vector assignment by a meta-agent
        # 1. initialize the shared args
        # TODO: 处理环境的时候，如果wallace area delay 调整，env pool 里面的ppa 也需要相应更新一下，用normalize area 和 normalize delay更新
        envs = self._get_initial_envs(n_processing)
        
        log_infos = []
        logger.log("starting experiments")
        if self.is_buffer_sharing:
            # share buffer
            shared_replay_memory = self.replay_memory
        else:
            # no share buffer
            shared_replay_memory = [copy.deepcopy(self.replay_memory) for _ in range(n_processing)]
        for episode_num in range(self.total_episodes):
            log_infos = []
            # lock = mp.Lock()
            # 1. change task weight vectors at the beginning of each episode
            # run episode 传入的参数，和环境的env 的 weight
            # 2. 更新环境的weight 和 target delay
            if episode_num >= self.start_adaptive_episodes:
                if (episode_num - self.start_adaptive_episodes) % self.update_tasks_freq == 0:
                    envs = self._modify_task_assignment(envs, episode_num)
                    self._log_modified_weight(envs, episode_num)

            logger.log(f"starting episode {episode_num}")
            # 1. run an episode for each task agent
            with Pool(n_processing) as pool:
                def collect_info(info):
                    log_infos.append(info)
                def error_info(error):
                    raise error
                for i, task_weight in enumerate(self.task_weight_vectors):
                    run_episode_kwargs = self._get_episode_kwargs(i)
                    if self.is_parallel:
                        run_episode_kwargs["is_parallel"] = True
                    else:
                        run_episode_kwargs["is_parallel"] = False
                    if self.is_buffer_sharing:
                        if self.is_vector_condition_q:
                            if self.is_parallel:
                                if self.is_weight_loss:
                                    if self.is_buffer_sharing_ablation:
                                        pool.apply_async(
                                            func=AsyncMultiTaskDQNAlgorithm.run_episode_vector_conditionq_v3,
                                            args=(i, self.task_weight_vectors, self.q_policy, self.replay_memory, envs[i], self.total_steps[i], self.loss_fn, self.rnd_predictor, self.rnd_target, self.int_reward_run_mean_std, shared_lock),
                                            kwds=run_episode_kwargs,
                                            callback=collect_info,
                                            error_callback=error_info
                                        )
                                    else:
                                        pool.apply_async(
                                            func=AsyncMultiTaskDQNAlgorithm.run_episode_vector_conditionq_v3,
                                            args=(i, self.task_weight_vectors, self.q_policy, self.replay_memory[i], envs[i], self.total_steps[i], self.loss_fn, self.rnd_predictor, self.rnd_target, self.int_reward_run_mean_std, shared_lock),
                                            kwds=run_episode_kwargs,
                                            callback=collect_info,
                                            error_callback=error_info
                                        )
                                else:
                                    pool.apply_async(
                                        func=AsyncMultiTaskDQNAlgorithm.run_episode_vector_conditionq_v2,
                                        args=(i, self.task_weight_vectors, self.q_policy, self.replay_memory, envs[i], self.total_steps[i], self.loss_fn, self.rnd_predictor, self.rnd_target, self.int_reward_run_mean_std, shared_lock),
                                        kwds=run_episode_kwargs,
                                        callback=collect_info,
                                        error_callback=error_info
                                    )
                            else:
                                pool.apply_async(
                                    func=AsyncMultiTaskDQNAlgorithm.run_episode_vector_conditionq,
                                    args=(i, self.task_weight_vectors, self.q_policy, self.replay_memory, envs[i], self.total_steps[i], self.loss_fn, self.rnd_predictor, self.rnd_target, self.int_reward_run_mean_std, shared_lock),
                                    kwds=run_episode_kwargs,
                                    callback=collect_info,
                                    error_callback=error_info
                                )
                        else:    
                            pool.apply_async(
                                func=AsyncMultiTaskDQNAlgorithm.run_episode,
                                args=(i, self.task_weight_vectors, self.q_policy, self.replay_memory, envs[i], self.total_steps[i], self.loss_fn, self.rnd_predictor, self.rnd_target, self.int_reward_run_mean_std, shared_lock),
                                kwds=run_episode_kwargs,
                                callback=collect_info,
                                error_callback=error_info
                            )
                    else:
                        pool.apply_async(
                            func=AsyncMultiTaskDQNAlgorithm.run_episode,
                            args=(i, self.task_weight_vectors, self.q_policy, shared_replay_memory[i], envs[i], self.total_steps[i], self.loss_fn, self.rnd_predictor, self.rnd_target, self.int_reward_run_mean_std, shared_lock),
                            kwds=run_episode_kwargs,
                            callback=collect_info,
                            error_callback=error_info
                        )
                pool.close()
                pool.join()
            logger.log(f"ending episode {episode_num}")
            # log data for each episode
            print(log_infos)
            rnd_loss, rnd_info = self.update_rnd_model(episode_num, envs)
            hv_value = self.log_global_information(episode_num, log_infos, envs)
            self.current_hv_value = hv_value

            if episode_num > 0 and (episode_num % self.end_exp_freq == 0):
                self.end_experiments(envs, episode_num)
        self.end_experiments(envs, episode_num)

    def end_experiments(self, envs, episode_num):
        # save datasets 
        save_data_dict = {}
        ppas_list = []
        # save env stats
        for i, env in enumerate(envs):
            # env initial state pool 
            save_data_dict[f"{i}-th env_initial_state_pool"] = list(env.initial_state_pool)
            # best state best design
            save_data_dict[f"{i}-th found_best_info"] = {
                "found_best_ppa": env.found_best_info["found_best_ppa"].value,
                "found_best_area": env.found_best_info["found_best_area"].value,
                "found_best_delay": env.found_best_info["found_best_delay"].value
            }

            best_state = copy.deepcopy(list(env.initial_state_pool)[-1]["state"])
            ppas_dict = env.get_ppa_full_delay_cons(best_state)
            ppas_list.append(ppas_dict)
        # pareto point set
        save_data_dict["pareto_area_points"] = self.pareto_pointset["area"]
        save_data_dict["pareto_delay_points"] = self.pareto_pointset["delay"]
        
        # test to get full pareto points
        # input: found_best_info state
        # output: testing pareto points and hypervolume
        merge_ppas_dict = self._merge_ppa(ppas_list)
        save_pareto_data_dict = self.log_and_save_pareto_points(merge_ppas_dict, episode_num)
        save_data_dict["testing_pareto_data"] = save_pareto_data_dict
        logger.save_npy(self.total_steps[0].value, save_data_dict)

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

# meta agent
class AsyncMultiTaskDQNAlgorithmV2(AsyncMultiTaskDQNAlgorithm):
    def _train_meta_agent(self, episode_num):
        device = self.device[0]
        # process data
        state = torch.tensor(self.meta_agent_datasets["state"]).unsqueeze(0).float().to(device)
        actions = self.meta_agent_datasets["action"]
        reward = self.current_hv_value - self.meta_agent_datasets["hypervolume"]
        # compute loss 
        logits = self.meta_agent(state)
        logp = torch.zeros(len(actions))
        for i in range(len(actions)):
            dist = Categorical(logits=logits[i*2:(i+1)*2])
            logp[i] = dist.log_prob(actions[i])
        reinforce_loss = -1. * reward * torch.sum(logp)
        # update meta agent
        self.meta_agent_optimizer.zero_grad()
        reinforce_loss.backward()
        for param in self.meta_agent.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_agent_optimizer.step()
        # log reward and loss
        logger.tb_logger.add_scalar('hypervolume reward', reward, global_step=episode_num)
        logger.tb_logger.add_scalar('reinforce loss', reinforce_loss.item(), global_step=episode_num)

    def execute_meta_agent(self, envs, episode_num):
        if self.meta_agent_type == "learning":
            # 1. train the meta agent
            if episode_num > self.start_adaptive_episodes:
                self._train_meta_agent(episode_num)
            # 2. sample action from the agent
            meta_state = self.get_meta_state(envs)
            device = self.device[0]
            with torch.no_grad():
                state = torch.tensor(meta_state).unsqueeze(0).float().to(device)
                logits = self.meta_agent(state)
                actions = []
                for i in range(len(envs)):
                    dist = Categorical(logits=logits[i*2:(i+1)*2])
                    action = dist.sample()
                    actions.append(action)
            ## store datasets
            self.meta_agent_datasets["state"] = meta_state
            self.meta_agent_datasets["action"] = actions
            self.meta_agent_datasets["hypervolume"] = self.current_hv_value
            # for i in range(len(probs)):
            #     logger.tb_logger.add_histogram(f'meta agent action probs {i}', probs[i], global_step=episode_num)
        elif self.meta_agent_type == "random":
            actions = [random.sample([0,1],1)[0] for _ in range(4)]

        # 3. execute the action 
        ## action_code: 0 加档，1减档，2不动
        for i in range(len(actions)):
            action_region = i
            action_type = int(actions[i])
            envs = self._execute_action(action_region, action_type, envs)
            # log action
            logger.tb_logger.add_scalar(f'{i}-th action region', action_region, global_step=episode_num)
            logger.tb_logger.add_scalar(f'{i}-th action type', action_type, global_step=episode_num)
        
        return envs
