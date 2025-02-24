import copy
import json
import logging
import multiprocessing
import os
import re
import math
import time
from multiprocessing import Pool

import torch
import matplotlib.pyplot as plt
import numpy as np

from o0_mul_utils import (
    decompose_compressor_tree,
    get_compressor_tree,
    get_initial_partial_product,
    legal_FA_list,
    legal_HA_list,
    legalize_compressor_tree,
    write_mul,
)

from o0_adder_utils import (
    get_init_cell_map, cell_map_legalize, get_mask_map, get_level_map, remove_tree_cell
)
from o0_rtl_tasks import EvaluateWorker, PowerSlewConsulter

from o0_logger import logger
from o0_netlist import NetList
net = NetList()

from ysh_logger import get_logger

class State:
    def __init__(
        self,
        bit_width: int,
        encode_type: str,
        max_stage_num: str,
        use_pp_wiring_optimize: bool,
        pp_wiring_init_type: str,
        use_compressor_map_optimize: bool,
        compressor_map_init_type: str,
        use_final_adder_optimize: bool,
        final_adder_init_type: str,
        top_name: str = "MUL",
    ) -> None:
        '''
        encode_type: and booth
        '''
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.max_stage_num = max_stage_num

        self.use_pp_wiring_optimize = use_pp_wiring_optimize
        self.use_compressor_map_optimize = use_compressor_map_optimize
        self.use_final_adder_optimize = use_final_adder_optimize

        self.pp_wiring_init_type = pp_wiring_init_type
        self.compressor_map_init_type = compressor_map_init_type

        self.final_adder_init_type = final_adder_init_type
        self.top_name = top_name

        """
        compressor_tree = [ct32, ct22]
        compressor_mask: pp_wiring[stage][column][wire_index]
        cell_map : len pp x  len pp
        pp_wiring: pp_wiring[stage][column][wire_index]
        """
        self.initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
        self.ct = None
        self.cell_map = None
        self.compressor_map = None
        self.pp_wiring = None

        """
        extra_info
        """
        self.compressor_connect_dict = None
        self.wire_connect_dict = None
        self.wire_constant_dict = None
        self.power_mask = None

        self.power_mask_cell_map = None

    def init(self, ct_type):
        """
        为什么不写在 __init__ 里面呢？
        担心被继承并且重载后出问题
        """
        if self.top_name == "MUL":
            self.get_initial_ct(ct_type)
            if self.use_compressor_map_optimize:
                self.get_initial_compressor_map(self.compressor_map_init_type)
            else:
                self.get_initial_compressor_map("default")
            self.get_initial_pp_wiring(self.pp_wiring_init_type)
        if self.use_final_adder_optimize:
            self.get_init_cell_map()

    def get_initial_ct(self, ct_type):
        if ct_type == "random":
            ct32 = np.random.randint(0, self.get_pp_height(), self.get_pp_len())
            ct22 = np.random.randint(0, self.get_pp_height(), self.get_pp_len())
            ct32, ct22 = legalize_compressor_tree(self.initial_pp, ct32, ct22)
            ct32_decomposed, ct22_decomposed, sequence_pp, stage_num = decompose_compressor_tree(self.initial_pp, ct32, ct22)
        else:
            ct32, ct22, ct32_decomposed, ct22_decomposed, sequence_pp, stage_num = get_compressor_tree(self.initial_pp, self.bit_width, ct_type)
        ct = np.zeros([2, len(self.initial_pp)])
        ct[0] = ct32
        ct[1] = ct22
        self.ct = ct
        self.stage_num = stage_num
        self.sequence_pp = sequence_pp
        ct_decomposed = np.zeros([2, len(ct32_decomposed), len(self.initial_pp)])
        ct_decomposed[0] = ct32_decomposed
        ct_decomposed[1] = ct22_decomposed
        self.ct_decomposed = ct_decomposed

    def get_pp_len(self) -> int:
        if self.encode_type == "and":
            return 2 * self.bit_width - 1
        elif self.encode_type == "booth":
            return 2 * self.bit_width
        else:
            raise NotImplementedError

    def get_pp_height(self) -> int:
        if self.encode_type == "and":
            return self.bit_width
        elif self.encode_type == "booth":
            return (self.bit_width) // 2 + 1


    def get_initial_compressor_map(self, init_type: str = "default"):
        """
        初始化压缩器种类
        self.compressor_map[compressor_type][stage][column] = 0, 1, 2, ...
        """
        assert self.ct is not None
        # fmt: off
        self.compressor_map = np.full([2, self.get_pp_len()], -1, int)
        for column_index in range(self.get_pp_len()):
            # FA
            if init_type == "default":
                self.compressor_map[0][column_index] = 0
            elif init_type == "random":
                self.compressor_map[0][column_index] = np.random.randint(len(legal_FA_list))
            else:
                raise NotImplementedError
            # HA
            if init_type == "default":
                self.compressor_map[1][column_index] = 0
            elif init_type == "random":
                self.compressor_map[1][column_index] = np.random.randint(len(legal_HA_list))
            else:
                raise NotImplementedError
        # fmt: on

    def get_initial_pp_wiring(self, init_type: str = "default"):
        """
        初始化部分积连线
        合法的位置是wire的map (0, 1, 2, ...)
        不合法的位置是 -1
        self.pp_wiring[stage_index][column_index] = [1, 2, 0, 3, ...]
        Parameters:
            init_type = "default" or "random"
        """
        # fmt: off
        assert self.ct is not None
        self.pp_wiring = np.full([self.max_stage_num, self.get_pp_len(), self.get_pp_height()], -1)

        remain_pp = copy.deepcopy(self.initial_pp)
        ct32_decomposed = self.ct_decomposed[0]
        ct22_decomposed = self.ct_decomposed[1]
        stage_num = len(ct32_decomposed)
        for stage_index in range(stage_num):
            for column_index in range(self.get_pp_len()):
                wire_num = int(remain_pp[column_index])
                if init_type == "random":
                    random_index = [wire_index for wire_index in range(wire_num)]
                    np.random.shuffle(random_index)
                    for wire_index in range(wire_num):
                        self.pp_wiring[stage_index][column_index][wire_index] = random_index[wire_index]
                elif init_type == "default":
                    for wire_index in range(wire_num):
                        self.pp_wiring[stage_index][column_index][wire_index] = wire_index

                # update remain pp
                remain_pp[column_index] += -2 * ct32_decomposed[stage_index][column_index] - ct22_decomposed[stage_index][column_index]
                if column_index > 0:
                    remain_pp[column_index] += ct32_decomposed[stage_index][column_index - 1] + ct22_decomposed[stage_index][column_index - 1]
        # fmt: on

    def get_init_cell_map(self):
        if self.top_name == "MUL":
            self.cell_map = get_init_cell_map(self.get_pp_len(), self.final_adder_init_type)
        else:
            self.cell_map = get_init_cell_map(self.bit_width, self.final_adder_init_type)
            
    def set_pp_wiring(self, filename: str = None):
        """
        设置部分积连线
        合法的位置是wire的map (0, 1, 2, ...)
        不合法的位置是 -1
        self.pp_wiring[stage_index][column_index] = [1, 2, 0, 3, ...]
        Parameters:
            filename
        """
        # fmt: off
        assert self.ct is not None
        self.pp_wiring = np.full([self.max_stage_num, self.get_pp_len(), self.get_pp_height()], -1)

        remain_pp = copy.deepcopy(self.initial_pp)
        ct32_decomposed = self.ct_decomposed[0]
        ct22_decomposed = self.ct_decomposed[1]
        with open(filename, 'r') as file:
            file_data = file.readlines() #读取所有行
            assert self.stage_num*self.get_pp_len() == len(file_data)
            row_index = 0
            for stage_index in range(self.stage_num):
                for column_index in range(self.get_pp_len()):
                    wire_num = int(remain_pp[column_index])
                    tmp_list = file_data[row_index].split(' ') #按‘ ’切分每行的数据
                    row_index += 1
                    # tmp_list[-1] = tmp_list[-1].replace('\n','') #去掉换行符
                    if tmp_list[-1] == '\n':
                        tmp_list = tmp_list[:-1]
                    if wire_num != len(tmp_list):
                        print('row_index: ', row_index)
                        print('wire_num: ', wire_num)
                        print('len(tmp_list): ', len(tmp_list))
                        print('tmp_list: ', tmp_list)
                    assert wire_num == len(tmp_list)
                    for wire_index in range(wire_num):
                        self.pp_wiring[stage_index][column_index][wire_index] = tmp_list[wire_index]

                    # update remain pp
                    remain_pp[column_index] += -2 * ct32_decomposed[stage_index][column_index] - ct22_decomposed[stage_index][column_index]
                    if column_index > 0:
                        remain_pp[column_index] += ct32_decomposed[stage_index][column_index - 1] + ct22_decomposed[stage_index][column_index - 1]
        # fmt: on
        
    # fmt: off
    def __transition(
        self, ct: np.ndarray, action_column: int, action_type: int
    ) -> np.ndarray:
        """
        不会改变自己的状态, only for ct
        action is a number, action coding:
            action=0: add a 2:2 compressor
            action=1: remove a 2:2 compressor
            action=2: replace a 3:2 compressor
            action=3: replace a 2:2 compressor
        Input: cur_state, action
        Output: next_state
        """
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

        legalized_ct32, legalized_ct22 = legalize_compressor_tree(get_initial_partial_product(self.bit_width, self.encode_type), ct[0], ct[1])
        legalized_ct = np.zeros_like(ct)
        legalized_ct[0] = legalized_ct32
        legalized_ct[1] = legalized_ct22
        return legalized_ct
    # fmt: on

    # fmt: off
    def transition(self, action_column: int, action_type: int) -> None:
        """
        会改变自己的状态
        action is a number, action coding:
            action=0: add a 2:2 compressor
            action=1: remove a 2:2 compressor
            action=2: replace a 3:2 compressor
            action=3: replace a 2:2 compressor
        Input: cur_state, action
        Output: next_state
        """
        if action_type < 4:
            # 改变压缩树
            self.ct = self.__transition(self.ct, action_column, action_type)
        elif self.use_compressor_map_optimize:
            if action_type - 4 < len(legal_FA_list):
                # 改一个FA的实现
                target_fa_type = action_type - 4
                # $$DEBUG
                # assert (target_fa_type != self.compressor_map[0][action_column]) and (self.ct[0][action_column] > 0)
                self.compressor_map[0][action_column] = target_fa_type
            elif action_type - 4 - len(legal_FA_list) < len(legal_HA_list):
                # 改一个HA的实现
                target_ha_type = action_type - 4 - len(legal_FA_list)
                # $$DEBUG
                # assert (target_ha_type != self.compressor_map[1][action_column]) and (self.ct[1][action_column] > 0)
                self.compressor_map[1][action_column] = target_ha_type
        else:
            raise NotImplementedError
    # fmt: on
    
    def get_nextstates(self):
        action_type_num = 4
        next_states = []
        mask = self.mask_with_legality()
        get_logger('ysh').info('state_mask_with_legality: {}'.format(mask))
        for action_index in range(self.get_pp_len()*action_type_num):
            if not mask[action_index]:
                next_states.append(None)
                continue
            action_column = int(action_index) // action_type_num
            action_type = int(action_index) % action_type_num

            next_state: State = copy.deepcopy(self)
            next_state.transition(action_column, action_type)
            next_states.append(next_state)
        return next_states
    
    def _worker_get_power_coefficient(self, next_state):
        if next_state == None:
            # next_power = 100000
            next_power = math.inf
        else:
            next_power = net.report_power(self.bit_width, self.encode_type, next_state.ct, None)
        return self.cur_power - next_power
        
    def get_power_mask(self, mask_min):
        '''
        return tensor
        '''

        start_time = time.time()
        self.cur_power = net.report_power(self.bit_width, self.encode_type, self.ct, None)
        next_states = self.get_nextstates()
        # 开始并行
        cpu_worker_num = 4
        with Pool(cpu_worker_num) as p:
            power_coefficient = p.map(self._worker_get_power_coefficient, next_states)
        # power_coefficient = torch.zeros(len(next_states))
        # for i, next_state in enumerate(next_states):
        #     if next_state == None:
        #         next_power = 100000
        #     else:
        #         next_power = net.report_power(self.bit_width, self.encode_type, next_state.ct, None)
        #     power_coefficient[i] = cur_power - next_power
        if mask_min >= 1-1e-9:
            power_mask = torch.tensor(np.ones(len(next_states)))
        elif mask_min != 0:
            power_mask = torch.nn.functional.softmax(torch.Tensor(power_coefficient), dim=0)
            power_mask = power_mask.numpy()
            power_mask = (power_mask-np.min(power_mask)) / (np.max(power_mask)-np.min(power_mask)) * (1-mask_min)
            power_mask = power_mask + mask_min
            power_mask = torch.tensor(power_mask)
        else:
            power_mask = torch.nn.functional.softmax(torch.Tensor(power_coefficient), dim=0)
        end_time = time.time()
        time_consumed = end_time - start_time
        return power_mask, power_coefficient
    
    # fmt: off
    def mask_with_legality(self):
        """
        有些动作会导致结果 stage 超过 max stage
        因此需要被过滤掉

        问题: 哪些动作会让stage增加? 原来的方法是遍历 有没有更好的方法
        """
        action_type_num = 4
        if self.use_compressor_map_optimize:
            action_type_num += len(legal_FA_list) + len(legal_HA_list)

        initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
        mask = np.zeros([action_type_num * len(initial_pp)])
        remain_pp = copy.deepcopy(initial_pp)
        for column_index in range(len(remain_pp)):
            if column_index > 0:
                remain_pp[column_index] += self.ct[0][column_index - 1] + self.ct[1][column_index - 1]
            remain_pp[column_index] += - 2 * self.ct[0][column_index] - self.ct[1][column_index]

        legal_act = []
        for column_index in range(2, len(initial_pp)):
            if remain_pp[column_index] == 2:
                legal_act.append((column_index, 0))
                if (self.ct[1][column_index] >= 1):
                    legal_act.append((column_index, 3))
            if remain_pp[column_index] == 1:
                if self.ct[0][column_index] >= 1:
                    legal_act.append((column_index, 2))
                if self.ct[1][column_index] >= 1:
                    legal_act.append((column_index, 1))
        # compressor domain
        if self.use_compressor_map_optimize:
            for column_index in range(len(initial_pp)):
                # FA, offset = 4
                if self.ct[0][column_index] > 0:
                    # 首先这一列必须要有东西，不然不合法
                    legal_type_index_list = list(range(4, 4 + len(legal_FA_list)))
                    # 然后这一列当前的是不合法的，要删掉（不然会出现大量"原地不动"的动作）
                    legal_type_index_list.remove(4 + int(self.compressor_map[0][column_index]))
                    for act_type in legal_type_index_list:
                        legal_act.append((column_index, act_type))
                # HA, offset = 4 + len(legal_FA_list)
                if self.ct[1][column_index] > 0:
                    # 首先这一列必须要有东西，不然不合法
                    legal_type_index_list = list(range(4 + len(legal_FA_list), 4 + len(legal_FA_list) + len(legal_HA_list)))
                    # 然后这一列当前的是不合法的，要删掉（不然会出现大量"原地不动"的动作）
                    legal_type_index_list.remove(4 + len(legal_FA_list) + int(self.compressor_map[1][column_index]))
                    for act_type in legal_type_index_list:
                        legal_act.append((column_index, act_type))

        for act_col, action in legal_act:
            if action >= 4:
                # 是改变压缩器种类的动作
                mask[act_col * action_type_num + action] = 1
                continue
            pp = copy.deepcopy(remain_pp)
            ct = copy.deepcopy(self.ct)

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
                    mask[act_col * action_type_num + action] = 1
                    break
                elif pp[i] == 1 or pp[i] == 2:
                    mask[act_col * action_type_num + action] = 1
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
        mask = (mask != 0)
        indices = np.where(mask)[0]

        for action in indices:
            ct = copy.deepcopy(self.ct)
            action_type = action % action_type_num
            action_column = action // action_type_num
            if action_type < 4:
                next_state = self.__transition(ct, action_column, action_type)
                ct32, ct22, _, __ = decompose_compressor_tree(initial_pp, next_state[0], next_state[1])
                if len(ct32) > self.max_stage_num:
                    mask[int(action)] = 0
        mask = (mask != 0)

        return mask
    # fmt: on

    # fmt: off
    def mask(self):
        action_type_num = 4
        if self.use_compressor_map_optimize:
            action_type_num += len(legal_FA_list) + len(legal_HA_list)
        
        initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
        mask = np.zeros([action_type_num * len(initial_pp)])
        remain_pp = copy.deepcopy(initial_pp)
        for column_index in range(len(remain_pp)):
            if column_index > 0:
                remain_pp[column_index] += self.ct[0][column_index - 1] + self.ct[1][column_index - 1]
            remain_pp[column_index] += - 2 * self.ct[0][column_index] - self.ct[1][column_index]

        legal_act = []
        # ct domain
        for column_index in range(2, len(initial_pp)):
            if remain_pp[column_index] == 2:
                legal_act.append((column_index, 0))
                if (self.ct[1][column_index] >= 1):
                    legal_act.append((column_index, 3))
            if remain_pp[column_index] == 1:
                if self.ct[0][column_index] >= 1:
                    legal_act.append((column_index, 2))
                if self.ct[1][column_index] >= 1:
                    legal_act.append((column_index, 1))
        # compressor domain
        if self.use_compressor_map_optimize:
            for column_index in range(len(initial_pp)):
                # FA, offset = 4
                if self.ct[0][column_index] > 0:
                    # 首先这一列必须要有东西，不然不合法
                    legal_type_index_list = list(range(4, 4 + len(legal_FA_list)))
                    # 然后这一列当前的是不合法的，要删掉（不然会出现大量"原地不动"的动作）
                    legal_type_index_list.remove(4 + int(self.compressor_map[0][column_index]))
                    for act_type in legal_type_index_list:
                        legal_act.append((column_index, act_type))
                # HA, offset = 4 + len(legal_FA_list)
                if self.ct[1][column_index] > 0:
                    # 首先这一列必须要有东西，不然不合法
                    legal_type_index_list = list(range(4 + len(legal_FA_list), 4 + len(legal_FA_list) + len(legal_HA_list)))
                    # 然后这一列当前的是不合法的，要删掉（不然会出现大量"原地不动"的动作）
                    legal_type_index_list.remove(4 + len(legal_FA_list) + int(self.compressor_map[1][column_index]))
                    for act_type in legal_type_index_list:
                        legal_act.append((column_index, act_type))

        for act_col, action in legal_act:
            mask[act_col * action_type_num + action] = 1

        mask = (mask != 0)
        return mask
    # fmt: on

    def emit_verilog(self, verilog_path):
        """
        生成 Verilog
        """
        # fmt: off
        verilog_dir = os.path.dirname(verilog_path)
        if not os.path.exists(verilog_dir):
            os.makedirs(verilog_dir)
        if self.top_name == "MUL":
            pp_wiring = self.pp_wiring
            if self.use_compressor_map_optimize:
                if len(self.compressor_map.shape) < 4:
                    compressor_map = self.misalign_compressor_map()
                else:
                    compressor_map = self.compressor_map
            else:
                compressor_map = None

            self.compressor_connect_dict, self.wire_connect_dict, self.wire_constant_dict = write_mul(verilog_path, self.bit_width, self.ct_decomposed, pp_wiring, compressor_map, self.use_final_adder_optimize, self.cell_map)
        else:
            write_mul(verilog_path, None, None, None, None, None, self.cell_map, is_adder_only=True)
        # fmt: on

    # fmt: off
    def misalign_compressor_map(self, granularity="compressor") -> list:
        """
        将 compressor_map 对齐到某个粒度
        "compressor": 细化到对某一个compressor操作
        "slice": 
        """
        assert self.compressor_map is not None
        if granularity == "compressor":
            misaligned_compressor_map = np.full([2, self.max_stage_num, self.get_pp_len(), self.get_pp_height()], -1, int)
            initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
            ct32_decomposed, ct22_decomposed, _, __ = decompose_compressor_tree(initial_pp, self.ct[0], self.ct[1])
            if len(self.compressor_map.shape) == 2:
                for stage_index in range(len(ct32_decomposed)):
                    for column_index in range(self.get_pp_len()):
                        cp_32_num = ct32_decomposed[stage_index][column_index]
                        cp_22_num = ct22_decomposed[stage_index][column_index]
                        for index in range(int(cp_32_num)):
                            misaligned_compressor_map[0][stage_index][column_index][index] = self.compressor_map[0][column_index]
                        for index in range(int(cp_22_num)):
                            misaligned_compressor_map[1][stage_index][column_index][index] = self.compressor_map[1][column_index]
                return misaligned_compressor_map
            elif len(self.compressor_map.shape) == 3:
                for stage_index in range(len(ct32_decomposed)):
                    for column_index in range(self.get_pp_len()):
                        cp_32_num = ct32_decomposed[stage_index][column_index]
                        cp_22_num = ct22_decomposed[stage_index][column_index]
                        for index in range(int(cp_32_num)):
                            misaligned_compressor_map[0][stage_index][column_index][index] = self.compressor_map[0][stage_index][column_index]
                        for index in range(int(cp_22_num)):
                            misaligned_compressor_map[1][stage_index][column_index][index] = self.compressor_map[1][stage_index][column_index]
                return misaligned_compressor_map
            else:
                raise NotImplementedError
        elif granularity == "slice":
            misaligned_compressor_map = np.full([2, self.max_stage_num, self.get_pp_len()], -1, int)
            initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
            ct32_decomposed, ct22_decomposed, _, __ = decompose_compressor_tree(initial_pp, self.ct[0], self.ct[1])
            if len(self.compressor_map.shape) == 2:
                for stage_index in range(len(ct32_decomposed)):
                    for column_index in range(self.get_pp_len()):
                        misaligned_compressor_map[0][stage_index][column_index] = self.compressor_map[0][column_index]
                        misaligned_compressor_map[1][stage_index][column_index] = self.compressor_map[1][column_index]
                return misaligned_compressor_map
            elif len(self.compressor_map.shape) == 3:
                return copy.deepcopy(self.compressor_map)
        else:
            raise NotImplementedError
    # fmt: on

    def update_power_mask(self, worker: EvaluateWorker = None):
        self.power_mask = worker.consult_compressor_power_mask(self.archive())

    # fmt: off
    def pp_wiring_arrangement_v0(self, verilog_path=None, build_dir=None, target_delay_list=None, n_processing=None, worker:EvaluateWorker=None) -> None:
        """
        部分积布线方案 V0
        直接修改所有的 slice
        Parameters:
            verilog_path: Verilog 代码的位置
            build_dir: 综合仿真的文件夹位置
            target_delay_list: 就是 target_delay_list
            n_processing: 线程数
            worker: 仿真器 如果不是空的 那么优先从这个仿真器中获取仿真信息 否则就会自己创建仿真器
        """
        assert self.ct is not None

        initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct32_decomposed, ct22_decomposed, _, __ = decompose_compressor_tree(initial_pp, self.ct[0], self.ct[1])
        ct_decomposed = [ct32_decomposed, ct22_decomposed]
        stage_num = len(ct32_decomposed)
        column_len = len(ct32_decomposed[0])

        # 首先获取每个阶段剩余pp的信息
        remain_pp = np.zeros([stage_num + 1, initial_pp.size])
        remain_pp[0] = initial_pp
        for stage_index in range(stage_num):
            for column_index in range(column_len):
                remain_pp[stage_index + 1][column_index] = remain_pp[stage_index][column_index] - 2 * ct32_decomposed[stage_index][column_index] - ct22_decomposed[stage_index][column_index]
                if column_index > 0:
                    remain_pp[stage_index + 1][column_index] += ct32_decomposed[stage_index][column_index - 1] + ct22_decomposed[stage_index][column_index - 1]

        # 获取 activity
        if worker is None:
            worker = EvaluateWorker(verilog_path, ["ppa", "activity"], target_delay_list, build_dir, n_processing=n_processing, clear_dir=False, clear_log=False, clear_netlist=False)
            if self.use_final_adder_optimize:
                worker.target_lists.append("prefix_adder")
            worker.evaluate()
        worker.update_wire_constant_dict(self.wire_constant_dict)

        # 获取 power_slew 信息
        power_slew_consulter = PowerSlewConsulter()

        # 开始优化!
        pp_wiring = copy.deepcopy(self.pp_wiring) # TODO 这里仅为了测试方便，可以删除
        if len(self.compressor_map.shape) < 4:
            misaligned_compressor_map = self.misalign_compressor_map("slice")
        else:
            misaligned_compressor_map = self.compressor_map
        for stage_index in range(stage_num):
            for column_index in range(column_len):
            # 获取 pp 端的信息
                pp_num = int(remain_pp[stage_index][column_index])
                slice_activity = worker.consult_port_activity(stage_index, column_index, pp_num, self.bit_width, stage_num, column_len, self.wire_connect_dict)
                pp_indices = range(pp_num)
                sorted_pp_indices = sorted(pp_indices, key=lambda x:slice_activity[x])

                # 获取 port 端的信息
                port_num = int(3 * ct32_decomposed[stage_index][column_index] + 2 * ct22_decomposed[stage_index][column_index])
                ct32_num, ct22_num = ct32_decomposed[stage_index][column_index], ct22_decomposed[stage_index][column_index]
                power_slew = power_slew_consulter.consult_power_slew(int(ct32_num), int(ct22_num), misaligned_compressor_map, stage_index, column_index)
                port_indices = range(port_num)
                sorted_port_indices = sorted(port_indices, key=lambda x:power_slew[x])
                sorted_port_indices.reverse() # 降序排列

                # 连线
                pp_wiring[stage_index][column_index] = np.full_like(pp_wiring[stage_index][column_index], -1)
                for port_rank, port_index in enumerate(sorted_port_indices):
                    pp_index = sorted_pp_indices[port_rank]
                    pp_wiring[stage_index][column_index][port_index] = pp_index
                for port_rank in range(len(sorted_port_indices), len(sorted_pp_indices)):
                    pp_index = sorted_pp_indices[port_rank]
                    port_index = port_rank
                    pp_wiring[stage_index][column_index][port_index] = pp_index
        # $$DEBUG
        # assert ((pp_wiring > -1) == (self.pp_wiring > -1)).all()
        self.pp_wiring = pp_wiring
        return worker

    # fmt: off
    def pp_wiring_arrangement_v1(self, verilog_path, build_dir, target_delay_list, n_processing, n, worker: EvaluateWorker = None) -> None:
        """
        部分积布线方案 V1
        挑出前 n 大的 slice
        然后只针对这 n 个 slice 来修改 wiring
        Parameters:
            verilog_path: Verilog 代码的位置
            build_dir: 综合仿真的文件夹位置
            target_delay_list: 就是 target_delay_list
            n_processing: 线程数
            n: 挑选多少 slice 来优化
            worker: 仿真器 如果不是空的 那么优先从这个仿真器中获取仿真信息 否则就会自己创建仿真器
        """
        assert self.ct is not None

        initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct32_decomposed, ct22_decomposed, _, __ = decompose_compressor_tree(initial_pp, self.ct[0], self.ct[1])
        ct_decomposed = [ct32_decomposed, ct22_decomposed]
        stage_num = len(ct32_decomposed)
        column_len = len(ct32_decomposed[0])

        # 首先获取每个阶段剩余pp的信息
        remain_pp = np.zeros([stage_num + 1, initial_pp.size])
        remain_pp[0] = initial_pp
        for stage_index in range(stage_num):
            for column_index in range(column_len):
                remain_pp[stage_index + 1][column_index] = remain_pp[stage_index][column_index] - 2 * ct32_decomposed[stage_index][column_index] - ct22_decomposed[stage_index][column_index]
                if column_index > 0:
                    remain_pp[stage_index + 1][column_index] += ct32_decomposed[stage_index][column_index - 1] + ct22_decomposed[stage_index][column_index - 1]

        # 获取 power_mask 和 activity
        if worker is None:
            worker = EvaluateWorker(verilog_path, ["ppa", "activity", "power"], target_delay_list, build_dir, n_processing=n_processing, clear_dir=False, clear_log=False, clear_netlist=False, )
            worker.evaluate()
        worker.update_wire_constant_dict(self.wire_constant_dict)
        power_mask = worker.consult_compressor_power_mask(ct_decomposed)
        self.power_mask = power_mask
        sum_power_mask = power_mask[0] + power_mask[1]
        # 获取需要优化的下标
        flat_indices = np.argpartition(sum_power_mask.flatten(), -n)[-n:]
        indices = np.unravel_index(flat_indices, sum_power_mask.shape)
        target_indices = list(zip(*indices))
        logging.info(f"Optimizing slices: {target_indices}")

        # 获取 power_slew 信息
        power_slew_consulter = PowerSlewConsulter()

        # 开始优化!
        pp_wiring = copy.deepcopy(self.pp_wiring)  # TODO 这里仅为了测试方便，可以删除
        for slice_index in target_indices:
            stage_index, column_index = slice_index
            # 获取 pp 端的信息
            pp_num = int(remain_pp[stage_index][column_index])
            slice_activity = worker.consult_port_activity(stage_index, column_index, pp_num, self.bit_width, stage_num, column_len, self.wire_connect_dict)
            pp_indices = range(pp_num)
            sorted_pp_indices = sorted(pp_indices, key=lambda x: slice_activity[x])

            # 获取 port 端的信息
            port_num = int(3 * ct32_decomposed[stage_index][column_index] + 2 * ct22_decomposed[stage_index][column_index])
            ct32_num, ct22_num = ct32_decomposed[stage_index][column_index], ct22_decomposed[stage_index][column_index],
            power_slew = power_slew_consulter.consult_power_slew(int(ct32_num), int(ct22_num), self.compressor_map, stage_index, column_index)
            port_indices = range(port_num)
            sorted_port_indices = sorted(port_indices, key=lambda x: power_slew[x])
            sorted_port_indices.reverse()

            # 连线
            pp_wiring[stage_index][column_index] = np.full_like(pp_wiring[stage_index][column_index], -1)
            for port_rank, port_index in enumerate(sorted_port_indices):
                pp_index = sorted_pp_indices[port_rank]
                pp_wiring[stage_index][column_index][port_index] = pp_index
            for port_rank in range(len(sorted_port_indices), len(sorted_pp_indices)):
                pp_index = sorted_pp_indices[port_rank]
                pp_wiring[stage_index][column_index][port_rank] = pp_index
        # $$DEBUG
        # assert ((pp_wiring > -1) == (self.pp_wiring > -1)).all()
        self.pp_wiring = pp_wiring
        # fmt: on
        return worker

    def archive(self, return_mask=False, legality_action_mask=True) -> np.ndarray:
        """
        用于生成 np.ndarray 形式的 image state
        Parameters:
            return_mask: 是否返回 power mask 和 action mask
            legality_action_mask: 是否返回
        """
        initial_pp = get_initial_partial_product(self.bit_width, self.encode_type)
        ct32_decomposed, ct22_decomposed, _, __ = decompose_compressor_tree(initial_pp, self.ct[0], self.ct[1])
        ct_decomposed = np.asarray([ct32_decomposed, ct22_decomposed])
        
        # TODO
        archived_ct = np.zeros([2, self.max_stage_num , self.get_pp_len()])
        archived_ct[:, :len(ct32_decomposed), :] = ct_decomposed
        if return_mask == True:
            action_type_num = 4
            if self.use_compressor_map_optimize:
                action_type_num += len(legal_FA_list) + len(legal_HA_list)
                comap = self.misalign_compressor_map("slice")
            assert self.power_mask is not None
            if legality_action_mask == True:
                action_mask = self.mask_with_legality()
            else:
                action_mask = self.mask()
            indices = np.where(action_mask)[0]
            image_action_mask = np.zeros([action_type_num, self.max_stage_num, self.get_pp_len()])
            for action in indices:
                action_type = action % action_type_num
                action_column = action // action_type_num
                image_action_mask[action_type, : , action_column] = 1
            if self.use_compressor_map_optimize:
                archived_ct = np.concatenate([archived_ct, comap, self.power_mask, image_action_mask], 0)
            else:
                archived_ct = np.concatenate([archived_ct, self.power_mask, image_action_mask], 0)
        return archived_ct

    def archive_cell_map(self, return_power_mask=False) -> np.ndarray:
        assert self.cell_map is not None

        level_map = get_level_map(self.cell_map)
        mask_map = get_mask_map(self.cell_map)

        if not return_power_mask:
            cell_mask_state = np.concatenate([[self.cell_map], [level_map], mask_map], axis=0)
        else:
            cell_mask_state = np.concatenate([[self.cell_map], [level_map], mask_map, [self.power_mask_cell_map]], axis=0)
        cell_mask_state = cell_mask_state.astype(float)

        return cell_mask_state
    
    def mask_cell_map(self) -> np.ndarray:
        assert self.cell_map is not None
        mask_map = get_mask_map(self.cell_map)
        return mask_map.flatten()
    
    def transition_cell_map(self, action):
        bit_width = len(self.cell_map)
        action_type = action // (bit_width ** 2)
        action_x = (action % bit_width ** 2) // bit_width
        action_y = (action % bit_width ** 2) % bit_width 

        if action_type == 0:
            # 添加一个 cell
            assert self.cell_map[action_x, action_y] == 0
            self.cell_map[action_x, action_y] = 1
        else:
            # # 去掉一个 cell
            # assert self.cell_map[action_x, action_y] == 1
            # self.cell_map[action_x, action_y] = 0
            # 去掉一个子树
            assert self.cell_map[action_x, action_y] == 1
            self.cell_map = remove_tree_cell(self.cell_map, [action_x], [action_y])
        self.cell_map = cell_map_legalize(self.cell_map)

    def update_power_mask_cell_map(self, worker: EvaluateWorker):
        self.power_mask_cell_map = worker.consult_cell_power_mask(len(self.cell_map))
