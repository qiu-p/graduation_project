import logging
import os
import copy
import numpy as np
from collections import deque
import json
import matplotlib.pyplot as plt

from o0_rtl_tasks import EvaluateWorker, PowerSlewConsulter
from o0_state import State, legal_FA_list, legal_HA_list

from o0_logger import logger

# fmt: on
class BaseEnv:
    def __init__(self, seed, **env_kwargs):
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


class RefineEnv(BaseEnv):
    # 注意这里完全重写了初始化函数
    def __init__(
        self,
        seed: int,
        build_path_base: str = "pybuild",
        bit_width: int = 8,
        pp_encode_type: str = "and",
        init_ct_type: str = "wallace",
        use_pp_wiring_optimize: bool = True,
        pp_wiring_init_type: str = "default",
        use_compressor_map_optimize: bool = False,
        compressor_map_init_type: str = "default",
        use_final_adder_optimize: bool = False,
        final_adder_init_type: str = "default",
        use_routing_optimize: bool = True,
        target_delay: list = [50, 250, 400, 650],
        opt_target_label: list = ["area", "delay", "power"],
        opt_target_weight: list = [2, 1, 2],
        normalize_reward_type: str = "scale",
        opt_target_scale: list = None,
        ppa_scale: float = 100,
        initial_state_pool_max_len: int = 0,
        reward_scale: float = 100,
        long_term_reward_scale: float = 1.0,
        reset_state_policy: str = "random",
        random_reset_steps: int = 201,
        store_state_type="less",
        alpha: float = 1,
        is_debug: bool = False,
        reward_type: str = "simulate",
        MAX_STAGE_NUM: int = 4,
        n_processing: int = 4,
        task_index: int = 0,
        top_name="MUL",
        evaluate_target=[],
        **env_kwargs,
    ):
        """
        强化学习环境
        Parameters:
            seed:                           随机数种子
            build_path_base:                综合仿真的工作路径
            bit_width:                      乘法器输入宽度
            pp_encode_type:                 部分积编码方式
            init_ct_type:                   初始压缩树结构
            use_pp_wiring_optimize:         弃用功能
            pp_wiring_init_type:            初始化 pp routing方式
                "default": 默认连线     "random": 随机连线
            use_compressor_map_optimize:    是否优化压缩器种类
            compressor_map_init_type:       初始化压缩器种类方法
            use_final_adder_optimize:       是否优化最终的CPA
            final_adder_init_type:          使用何种 CPA
            use_routing_optimize:           是否使用布线优化
            target_delay:                   就是target delay
            opt_target_label:               优化目标的标签列表
            opt_target_weight:              优化目标的权重
            opt_target_scale:               优化指标的scale相关的参数
            normalize_reward_type:          normalize 方式, "scale" or "normal"
        """
        self.set_seed(seed)

        self.bit_width = bit_width
        self.pp_encode_type = pp_encode_type
        self.init_ct_type = init_ct_type
        self.use_pp_wiring_optimize = use_pp_wiring_optimize
        self.pp_wiring_init_type = pp_wiring_init_type
        self.use_compressor_map_optimize = use_compressor_map_optimize
        self.action_type_num = 4
        if use_compressor_map_optimize:
            self.action_type_num += len(legal_FA_list) + len(legal_HA_list)
        self.compressor_map_init_type = compressor_map_init_type
        self.use_final_adder_optimize = use_final_adder_optimize
        self.final_adder_init_type = final_adder_init_type

        self.target_delay = target_delay
        self.opt_target_label = opt_target_label
        self.opt_target_weight = opt_target_weight
        assert len(opt_target_label) == len(opt_target_weight)

        self.normalize_reward_type = normalize_reward_type
        self.ppa_scale = ppa_scale
        self.reward_scale = reward_scale
        self.long_term_reward_scale = long_term_reward_scale
        self.reset_state_policy = reset_state_policy
        self.random_reset_steps = random_reset_steps
        self.store_state_type = store_state_type
        self.alpha = alpha
        self.is_debug = is_debug
        self.reward_type = reward_type
        self.MAX_STAGE_NUM = MAX_STAGE_NUM
        self.top_name = top_name

        self.use_routing_optimize = use_routing_optimize

        self.evaluate_target = evaluate_target
        if use_routing_optimize is True:
            self.evaluate_target += ["ppa", "activity", "power"]
        else:
            self.evaluate_target += ["ppa", "power"]
        
        if self.use_final_adder_optimize:
            self.evaluate_target.append("prefix_adder")
            self.evaluate_target.append("prefix_adder_power")
        
        if top_name != "MUL":
            self.evaluate_target=["ppa"]

        self.n_processing = n_processing
        assert n_processing >= 0

        self.build_path_base = build_path_base

        # 初始化当前状态信息
        if not os.path.exists(self.build_path_base):
            os.makedirs(self.build_path_base)
        self.eval_build_path = os.path.join(
            build_path_base, f"{bit_width}bits_{pp_encode_type}_{task_index}"
        )
        self.rtl_path = os.path.join(self.eval_build_path, "MUL.v")

        # 需要维护的当前状态
        self.cur_state: State = None
        self.initial_evaluate_worker_no_routing: EvaluateWorker = None
        self.cur_evaluate_worker: EvaluateWorker = None
        self.opt_target_scale = opt_target_scale
        self.__reset()

        self.initial_state_pool_max_len = initial_state_pool_max_len
        self.initial_state = copy.deepcopy(self.cur_state)
        self.initial_evaluate_worker = copy.deepcopy(self.cur_evaluate_worker)
        if initial_state_pool_max_len > 0:
            self.initial_state_pool = deque([], maxlen=initial_state_pool_max_len)
            self.initial_state_pool.append(
                {
                    "state": copy.deepcopy(self.cur_state),
                    "evaluate_worker": copy.deepcopy(self.cur_evaluate_worker),
                    "count": 1,
                }
            )
        else:
            self.initial_state_pool = None

    def __reset(self):
        """
        name mangling 的 __reset
        仅限于 RefineEnv 的 __init__ 中调用
        确保被继承后不会被覆盖掉
        """
        # fmt: off
        self.cur_state = State(
            self.bit_width,
            self.pp_encode_type,
            self.MAX_STAGE_NUM,
            self.use_pp_wiring_optimize,
            self.pp_wiring_init_type,
            self.use_compressor_map_optimize,
            self.compressor_map_init_type,
            self.use_final_adder_optimize,
            self.final_adder_init_type,
            top_name=self.top_name,
        )
        self.cur_state.init(self.init_ct_type)
        # 仿真初始状态 并且把初始状态的 ppa 作为 scale
        eval_build_path = os.path.join(self.build_path_base, f"reset_{self.init_ct_type}")
        rtl_path = os.path.join(eval_build_path, "MUL.v")
        self.cur_state.emit_verilog(rtl_path)
        self.cur_evaluate_worker = EvaluateWorker(
            rtl_path,
            self.evaluate_target,
            self.target_delay,
            eval_build_path,
            False, False, False, False,
            n_processing=self.n_processing,
            top_name=self.top_name,
        )
        self.cur_evaluate_worker.evaluate()
        if "power" in self.evaluate_target:
            self.cur_state.update_power_mask(self.cur_evaluate_worker)
        self.initial_evaluate_worker_no_routing = copy.deepcopy(self.cur_evaluate_worker)
        if self.use_routing_optimize:
            self.cur_state.pp_wiring_arrangement_v0(None, None, None, None, self.cur_evaluate_worker)
            self.cur_state.emit_verilog(rtl_path)
            self.cur_evaluate_worker.evaluate()
            self.cur_state.update_power_mask(self.cur_evaluate_worker)
        ppa_dict = self.cur_evaluate_worker.consult_ppa()
        if (self.opt_target_scale is None or self.opt_target_scale == "None") and self.normalize_reward_type == "scale":
            self.opt_target_scale = []
            for ppa_key in self.opt_target_label:
                self.opt_target_scale.append(ppa_dict[ppa_key])
        else:
            pass  # 什么都不需要做

    # fmt: off
    def select_state_from_pool(self, state_novelty, state_value):
        if state_novelty is None and state_value is None:
            sel_index = np.random.randint(len(self.initial_state_pool))
            initial_state = self.initial_state_pool[sel_index]["state"]
        else:
            if self.reset_state_policy == "random":
                sel_index = np.random.randint(len(self.initial_state_pool))
                initial_state = self.initial_state_pool[sel_index]["state"]
            else:
                """ TODO """
                raise NotImplementedError
        return initial_state, sel_index
    # fmt: on

    # fmt: off
    def reset(self, state_novelty=None, state_value=None):
        if self.initial_state_pool_max_len > 0:
            initial_state, sel_index = self.select_state_from_pool(state_novelty, state_value)
            self.cur_state = copy.deepcopy(self.initial_state_pool[sel_index]["state"])
            self.cur_evaluate_worker = copy.deepcopy(self.initial_state_pool[sel_index]["evaluate_worker"])
        else:
            sel_index = 0
            initial_state = copy.deepcopy(self.initial_state)
            self.cur_state = copy.deepcopy(self.initial_state)
            self.cur_evaluate_worker = copy.deepcopy(self.initial_evaluate_worker)
        return initial_state, sel_index
    # fmt: on

    def normalize_target(self, target_index, target_value, global_step=None, my_logger=None):
        if global_step != None:
            logger.tb_logger.add_scalar(
                    'opt_target_scale_{}'.format(self.opt_target_label[target_index]), self.opt_target_scale[target_index], global_step=global_step)
            if my_logger != None:
                my_logger.info('opt_target_scale_{}: {}'.format(self.opt_target_label[target_index], self.opt_target_scale[target_index]))
        if self.normalize_reward_type == "scale":
            return target_value / self.opt_target_scale[target_index]
        elif self.normalize_reward_type == "normal":
            return (
                target_value - self.opt_target_scale[target_index][0]
            ) / self.opt_target_scale[target_index][1]
        else:
            raise NotImplementedError

    # fmt: off
    def get_ppa(self, ppa_dict, global_step=None, logger=None):
        ppa = 0.0
        for ppa_key_index, ppa_key in enumerate(self.opt_target_label):
            normalized_value = self.normalize_target(ppa_key_index, ppa_dict[ppa_key], global_step, logger)
            ppa += self.opt_target_weight[ppa_key_index] * normalized_value
            logging.debug(f"env.get_ppa: {ppa_key}, value={ppa_dict[ppa_key]}, scale={self.opt_target_scale[ppa_key_index]}, normalized={normalized_value}, weight={self.opt_target_weight[ppa_key_index]}")
        ppa *= self.ppa_scale
        return ppa
    # fmt: on

    # fmt: off
    def process_reward(self, next_evaluate_worker: EvaluateWorker):
        """
        从仿真信息中获得奖励
        Parameters:
            next_evaluate_worker: 仿真器 里面需要包含仿真信息
        """
        avg_ppa_dict = next_evaluate_worker.consult_ppa()
        ppa = self.get_ppa(avg_ppa_dict)

        last_avg_ppa_dict = self.cur_evaluate_worker.consult_ppa()
        last_ppa = self.get_ppa(last_avg_ppa_dict)

        initial_avg_ppa_dict = self.initial_evaluate_worker.consult_ppa()
        initial_ppa = self.get_ppa(initial_avg_ppa_dict)
        long_term_reward = initial_ppa - ppa

        reward = last_ppa - ppa
        reward += self.long_term_reward_scale * long_term_reward

        reward_dict = {
            "reward": reward,
            "avg_ppa_dict": avg_ppa_dict,
            "avg_ppa": ppa,
        }
        return reward, reward_dict
    # fmt: on

    # fmt: off
    def step(self, action_index: int):
        action_column = int(action_index) // self.action_type_num
        action_type = int(action_index) % self.action_type_num

        next_state = copy.deepcopy(self.cur_state)
        next_state.transition(action_column, action_type)
        next_state.get_initial_pp_wiring()
        next_state.emit_verilog(self.rtl_path)
        next_evaluate_worker = EvaluateWorker(
            self.rtl_path,
            self.evaluate_target,
            self.target_delay,
            self.eval_build_path,
            n_processing=self.n_processing,
            top_name=self.top_name,
        )
        next_evaluate_worker.evaluate()
        next_state.update_power_mask(next_evaluate_worker)
        if self.use_routing_optimize:
            # 使用部分积布线优化!
            next_evaluate_worker_no_routing = copy.deepcopy(next_evaluate_worker) # 首先保存一下优化前的结果
            next_state.pp_wiring_arrangement_v0(None, None, None, None, next_evaluate_worker)
            next_state.emit_verilog(self.rtl_path)
            next_evaluate_worker.evaluate()
            next_state.update_power_mask(next_evaluate_worker)

        reward, reward_dict = self.process_reward(next_evaluate_worker)

        # 改变当前状态
        self.cur_state = copy.deepcopy(next_state)
        self.cur_evaluate_worker = copy.deepcopy(next_evaluate_worker)

        step_info_dict = {
            "next_state": next_state,
            "reward": reward,
            "reward_dict": reward_dict,
            "evaluate_worker": next_evaluate_worker,
        }
        if self.use_routing_optimize:
            step_info_dict["evaluate_worker_no_routing"] = next_evaluate_worker_no_routing
        return step_info_dict
    # fmt: on

    def mask_with_legality(self):
        """
        test only
        """
        return self.cur_state.mask_with_legality()

    def mask(self):
        """
        test only
        """
        return self.cur_state.mask()

    def get_mutual_distance(self):
        number_states = len(self.initial_state_pool)
        mutual_distances = np.zeros((number_states, number_states))
        for i in range(number_states):
            state_i: State = self.initial_state_pool[i]["state"]
            for j in range(number_states):
                state_j: State = self.initial_state_pool[j]["state"]
                if self.top_name == "MUL":
                    mutual_dis = np.linalg.norm(state_i.ct - state_j.ct, ord=2)
                else:
                    mutual_dis = np.linalg.norm(state_i.cell_map - state_j.cell_map, ord=2)
                mutual_distances[i, j] = mutual_dis
        mutual_distances = np.around(mutual_distances, decimals=2)
        return mutual_distances

    def get_ppa_full_delay_cons(self, state: State, n_processing=None):
        target_delay = []
        if self.bit_width == 8:
            for i in range(50, 1000, 10):
                target_delay.append(i)
        elif self.bit_width == 16:
            for i in range(50, 2000, 10):
                target_delay.append(i)
        elif self.bit_width == 32:
            for i in range(50, 3000, 10):
                target_delay.append(i)
        elif self.bit_width == 64:
            for i in range(50, 4000, 10):
                target_delay.append(i)
        else:
            for i in range(50, 1000, 10):
                target_delay.append(i)

        if n_processing is None:
            n_processing = self.n_processing
        eval_build_path = os.path.join(
            self.build_path_base, f"full_ppa_{self.init_ct_type}"
        )
        rtl_path = os.path.join(eval_build_path, "MUL.v")
        evaluate_worker = EvaluateWorker(
            rtl_path,
            ["ppa"],
            target_delay,
            eval_build_path,
            n_processing=n_processing,
            top_name=self.top_name,
        )
        state.emit_verilog(rtl_path)
        evaluate_worker.evaluate()
        return evaluate_worker.consult_ppa_list()

    def verilate(self):
        raise NotImplementedError

    def get_pp_len(self) -> int:
        if self.pp_encode_type == "and":
            return 2 * self.bit_width - 1
        elif self.pp_encode_type == "booth":
            return 2 * self.bit_width
        else:
            raise NotImplementedError


class RefineEnvMultiAgent(RefineEnv):
    # fmt: off
    def step(self, action_ct: int = None, action_pt: int = None, action: int = None):
        """
        Parameters:
            action_ct: 压缩树动作
            action_pt: 前缀树动作
            action: 总动作，优先使用

            action = action_pt + action_ct * (2 * pp_len ** 2)
            action_pt = action % (2 * pp_len ** 2)
            action_ct = action // (2 * pp_len ** 2)
        """
        next_state = copy.deepcopy(self.cur_state)

        if action is not None:
            offset = 2 * next_state.get_pp_len() ** 2
            action_pt = action % offset
            action_ct = action // offset

        if action_ct is not None:
            action_column = int(action_ct) // self.action_type_num
            action_type = int(action_ct) % self.action_type_num
            next_state.transition(action_column, action_type)
        if action_pt is not None:
            next_state.transition_cell_map(action_pt)

        if self.top_name == "MUL":
            next_state.get_initial_pp_wiring()
        next_state.emit_verilog(self.rtl_path)
        next_evaluate_worker = EvaluateWorker(
            self.rtl_path,
            self.evaluate_target,
            self.target_delay,
            self.eval_build_path,
            n_processing=self.n_processing,
            top_name=self.top_name,
        )
        next_evaluate_worker.evaluate()
        if self.top_name == "MUL":
            next_state.update_power_mask(next_evaluate_worker)
        next_state.update_power_mask_cell_map(next_evaluate_worker)
        if self.use_routing_optimize:
            # 使用部分积布线优化!
            next_evaluate_worker_no_routing = copy.deepcopy(next_evaluate_worker) # 首先保存一下优化前的结果
            next_state.pp_wiring_arrangement_v0(None, None, None, None, next_evaluate_worker)
            next_state.emit_verilog(self.rtl_path)
            next_evaluate_worker.evaluate()
            next_state.update_power_mask(next_evaluate_worker)

        reward, reward_dict = self.process_reward(next_evaluate_worker)

        # 改变当前状态
        self.cur_state = copy.deepcopy(next_state)
        self.cur_evaluate_worker = copy.deepcopy(next_evaluate_worker)

        step_info_dict = {
            "next_state": next_state,
            "reward": reward,
            "reward_dict": reward_dict,
            "evaluate_worker": next_evaluate_worker,
        }
        if self.use_routing_optimize:
            step_info_dict["evaluate_worker_no_routing"] = next_evaluate_worker_no_routing
        return step_info_dict
    # fmt: on


def test_env_step():
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
    )
    env = RefineEnv(
        0,
        bit_width=16,
        MAX_STAGE_NUM=7,
        build_path_base="pybuild/env_debug",
        n_processing=4,
        pp_encode_type="and",
        init_ct_type="dadda",
        normalize_reward_type="normal",
        opt_target_scale=[
            [2046.93375, 81.45554568866075],
            [2.45930775, 0.1302654988233166],
            [0.0088753, 0.001296428352243193],
        ],
    )
    with open("debuglog/log-booth", "w") as file:
        pass

    for step_index in range(100):
        mask = env.mask_with_legality()

        indices = np.where(mask)[0]
        action = np.random.choice(indices)
        with open("debuglog/log-booth", "a") as file:
            file.write(f"\n========= step {step_index} =============\n")
            file.write(
                f"action = {action}, column = {action // 4}, type = {action % 4}\n"
            )
            file.write(f"state before = \n{env.cur_state.ct} \n")
        _, reward, __ = env.step(action)
        with open("debuglog/log-booth", "a") as file:
            file.write(f"state before = \n{env.cur_state.ct} \n")
            file.write(
                f"reward = {__}, ppa = {env.get_ppa(env.cur_evaluate_worker.consult_ppa())} \n"
            )
        print(f"## step {step_index}, action = {action}, reward = {reward}")
    pass


def debug_env_step():
    env = RefineEnv(
        0,
        bit_width=16,
        MAX_STAGE_NUM=7,
        build_path_base="pybuild/env_debug",
        n_processing=4,
        init_ct_type="dadda",
    )
    env.cur_state.ct = np.asarray(
        [
            [
                0,
                0,
                0,
                2,
                2,
                4,
                4,
                5,
                7,
                8,
                9,
                9,
                10,
                12,
                12,
                13,
                13,
                13.0,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0.0,
            ],
            [
                0,
                0,
                1,
                0,
                2,
                0,
                2,
                3,
                1,
                1,
                1,
                2,
                2,
                0,
                1,
                1,
                1,
                0.0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
            ],
        ]
    )
    env.cur_state.get_initial_compressor_map()
    env.cur_state.get_initial_pp_wiring()
    mask = env.mask_with_legality()
    env.step(44)


if __name__ == "__main__":
    test_env_step()
