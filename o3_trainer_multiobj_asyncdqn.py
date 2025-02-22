"""
    DQN training algorithm drawed by the paper 
    "RL-MUL: Multiplier Design Optimization with Deep Reinforcement Learning"
"""
import copy
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
from torch.multiprocessing import Pool, set_start_method
import torch.multiprocessing as mp
from o0_logger import logger
from o5_utils import Transition, MBRLTransition, MultiObjTransition, SharedMultiObjTransition
from o0_global_const import PartialProduct
from o1_environment import RefineEnv, SerialRefineEnv
from ipdb import set_trace

def compute_values(
    state_batch, action_batch, state_mask, env, q_policy, target_q_policy,
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
            q_values = q_policy(state.unsqueeze(0).float(), state_mask=state_mask[i])
            q_values = q_values.reshape((int(int_bit_width*2))*4)           
            # q_values = self.q_policy(state.unsqueeze(0)).reshape((int(self.int_bit_width*2))*4)
            state_action_values[i] = q_values[action_batch[i]]
        else:
            q_values = target_q_policy(state.unsqueeze(0).float(), is_target=True, state_mask=state_mask[i])
            state_action_values[i] = q_values.max(1)[0].detach()
    return state_action_values

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

def compute_q_loss(replay_memory, env, q_policy, target_q_policy, loss_fn, task_weight_vectors, task_index, **q_kwargs):
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
        # int_rewards_batch = self.compute_int_rewards(
        #     next_state_batch, next_state_mask
        # )
        # int_rewards_batch = int_rewards_batch / torch.tensor(np.sqrt(self.int_reward_run_mean_std.var), device=self.device)
        # train_reward_batch = reward_batch.to(self.device) + self.int_reward_scale * int_rewards_batch
        q_policy = q_policy.to(q_kwargs["device"])
        target_q_policy = target_q_policy.to(q_kwargs["device"])
        
        train_reward_batch = task_weight_vectors[0][0] * area_reward_batch + task_weight_vectors[0][1] * delay_reward_batch
        train_reward_batch = train_reward_batch.to(q_kwargs["device"])

        state_action_values = compute_values(
            state_batch, action_batch, state_mask,
            env, q_policy, target_q_policy,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
        )
        next_state_values = compute_values(
            next_state_batch, None, next_state_mask,
            env, q_policy, target_q_policy,
            q_kwargs["device"], q_kwargs["initial_partial_product"], q_kwargs["MAX_STAGE_NUM"], q_kwargs["int_bit_width"]
        )
        target_state_action_values = (next_state_values * q_kwargs["gamma"]) + train_reward_batch

        loss = loss_fn(
            state_action_values.unsqueeze(1), 
            target_state_action_values.unsqueeze(1)
        )
        info = {
            "loss": loss.item(),
            "q_values": state_action_values.detach().cpu().numpy(),
            "target_q_values": target_state_action_values.detach().cpu().numpy(),
            "is_update": True
        }
        # self.policy_optimizer.zero_grad()
        # loss.backward()
        # for param in self.q_policy.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.policy_optimizer.step()

        # info = {
        #     "q_values": state_action_values.detach().cpu().numpy(),
        #     "target_q_values": target_state_action_values.detach().cpu().numpy(),
        #     "positive_rewards_number": torch.sum(torch.gt(reward_batch.cpu(), 0).float())
        # }
        # self.rnd_int_rewards = np.mean(int_rewards_batch.cpu().numpy())
        # self.rnd_ext_rewards = np.mean(reward_batch.cpu().numpy())

        # if self.total_steps % self.update_rnd_freq == 0:
        #     rnd_loss = self.update_rnd_model(
        #         next_state_batch, next_state_mask
        #     )
    return loss, info

def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        # if param.grad is not None:
        shared_param._grad = param.grad.clone().cpu()

class AsyncDQNAlgorithm():
    def __init__(
        self,
        q_policy,
        replay_memory,
        # rnd_predictor,
        # rnd_target,
        # int_reward_run_mean_std,
        seed,
        # rnd kwargs 
        rnd_lr=3e-4,
        update_rnd_freq=10,
        int_reward_scale=1,
        # evaluate kwargs
        evaluate_freq=5,
        evaluate_num=5,
        # multi task kwargs
        task_weight_vectors=[[4,1],[4,1],[4,1],[4,1]],
        target_delay=[
            [50,200,500,1200], [50,200,500,1200],
            [50,200,500,1200], [50,200,500,1200]
        ],
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
        # bit width
        bit_width="16_bits",
        int_bit_width=16,
        str_bit_width=16,
        # env kwargs
        env_kwargs={}
    ):
        # 1. 处理好启动并行线程需要的共享变量
        # 2. run experiments，并行启动多进程，多线程需要能调用gpu，多线程传入的参数不一样，其他执行的程序是一样的，要上锁
        # 3. 按照一个episode 一个episode来，每次并行启动一个episode？ 统一更新RND model? 统一分配weight vectors？

        self.q_policy = q_policy
        self.replay_memory = replay_memory
        # self.rnd_predictor = rnd_predictor
        # self.rnd_target = rnd_target
        # self.int_reward_run_mean_std = int_reward_run_mean_std
        self.seed = seed
        # kwargs
        self.rnd_lr = rnd_lr
        self.update_rnd_freq = update_rnd_freq
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

        self.bit_width = bit_width
        self.int_bit_width = int_bit_width
        self.initial_partial_product = PartialProduct[self.bit_width][:-1]

        self.env_kwargs = env_kwargs

        self.loss_fn = nn.SmoothL1Loss()

    @staticmethod
    def run_episode(task_index, task_weight_vectors, shared_q_policy, replay_memory, env, total_steps, loss_fn, lock, **kwargs):
        log_info = {
            "reward": [],
            "loss": [],
            "q_values": [],
            "target_q_values": [],
            "avg_ppa": [],
            "task_index": task_index,
            "eps_threshold": []
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
        q_kwargs = {
            "batch_size": kwargs["batch_size"],
            "device": kwargs["device"],
            "initial_partial_product": kwargs["initial_partial_product"],
            "MAX_STAGE_NUM": kwargs["MAX_STAGE_NUM"],
            "int_bit_width": kwargs["int_bit_width"],
            "gamma": kwargs["gamma"]
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
                torch.tensor(state), total_steps.value
            )
            next_state, reward, rewards_dict = env.step(action)
            _, next_state_policy_info = q_policy.select_action(
                torch.tensor(next_state), total_steps.value
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
                replay_memory, env, q_policy, target_q_policy, 
                loss_fn, task_weight_vectors, task_index, **q_kwargs)
            if loss_info["is_update"]:
                # update shared model
                q_loss.backward()
                for param in q_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                # The critical section begins
                lock.acquire()
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
            else:
                # log info
                log_info["reward"].append(reward)
                log_info["avg_ppa"].append(rewards_dict["avg_ppa"])
                log_info["loss"].append(0)
                log_info["q_values"].append(0)
                log_info["target_q_values"].append(0)
                log_info["eps_threshold"].append(policy_info["eps_threshold"])
        
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
                    weight_area=self.task_weight_vectors[0][0],
                    weight_delay=self.task_weight_vectors[0][1],
                    task_index=i, target_delay=self.target_delay[0],
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
            "MAX_STAGE_NUM": self.MAX_STAGE_NUM
        }
        return run_episode_kwargs

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
            
            logger.tb_logger.add_scalar(f'task index {task_index} total_steps', self.total_steps[task_index].value, global_step=episode_num)
            
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
        for episode_num in range(self.total_episodes):
            # lock = mp.Lock()
            logger.log(f"starting episode {episode_num}")
            # 1. run an episode for each task agent
            with Pool(n_processing) as pool:
                def collect_info(info):
                    log_infos.append(info)
                def error_info(error):
                    raise error
                for i, task_weight in enumerate(self.task_weight_vectors):
                    run_episode_kwargs = self._get_episode_kwargs(i)
                    pool.apply_async(
                        func=AsyncDQNAlgorithm.run_episode,
                        args=(i, self.task_weight_vectors, self.q_policy, self.replay_memory, envs[i], self.total_steps[i], self.loss_fn, shared_lock),
                        kwds=run_episode_kwargs,
                        callback=collect_info,
                        error_callback=error_info
                    )
                    # pool.apply_async(
                    #     func=AsyncMultiTaskDQNAlgorithm.run_episode_v2,
                    #     args=(i),
                    #     callback=collect_info
                    # )
                pool.close()
                pool.join()
            logger.log(f"ending episode {episode_num}")
            # log data for each episode
            print(log_infos)
            self.log_global_information(episode_num, log_infos, envs)

