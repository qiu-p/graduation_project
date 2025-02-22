import numpy as np
import math
import hashlib
import os
import shutil
import copy
import subprocess
import random
from collections import deque
from typing import List, Tuple

from o0_adder_utils import cell_map_legalize, adder_output_verilog_main

OpenRoadFlowPath = "/datasets/ai4multiplier/openroad_deb/OpenROAD-flow-scripts"
lef = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary.lef'
lib = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary_typical.lib'

BLACK_CELL = """module BLACK(gik, pik, gkj, pkj, gij, pij);
input gik, pik, gkj, pkj;
output gij, pij;
assign pij = pik & pkj;
assign gij = gik | (pik & gkj);
endmodule
"""

GREY_CELL = """module GREY(gik, pik, gkj, gij);
input gik, pik, gkj;
output gij;
assign gij = gik | (pik & gkj);
endmodule
"""


# yosys_script_format = """read -sv {}
# hierarchy -top main
# flatten
# proc; techmap; opt;
# abc -D {} -fast -liberty {}
# write_verilog {}
# """

abc_constr_format = """set_driving_cell BUF_X1
set_load 10.0 [all_outputs]
"""

yosys_script_format = """read -sv {}
synth -top main
dfflibmap -liberty {}
abc -D {} -constr {} -liberty {}
write_verilog {}
"""

sdc_format = """create_clock [get_ports clk] -name core_clock -period 3.0
set_all_input_output_delays
"""

openroad_tcl = """source "helpers.tcl"
source "flow_helpers.tcl"
source "Nangate45/Nangate45.vars"
set design "adder"
set top_module "main"
set synth_verilog "{}"
set sdc_file "{}"
set die_area {{0 0 80 80}}
set core_area {{0 0 80 80}}
source -echo "fast_flow.tcl"
"""

openroad_sta_tcl = """read_lef {}
read_lib {}
read_verilog {}
link_design main
set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts \"wns $path_delay\"
report_design_area
exit
"""

result_cache = {}
global_step = 0
cache_hit = 0


class State(object):
    def __init__(
        self,
        level: int,
        size: int,
        cell_map: List[List[int]],
        level_map: List[List[int]],
        min_map: List[List[int]],
        step_num: int,
        action: int,
        reward: float,
        level_bound_delta: float,
        input_bit: int,
        initial_adder_type: int = 0,
    ) -> None:
        self.initial_adder_type = initial_adder_type
        self.current_value = 0.0
        self.current_round_index = 0
        self.input_bit = input_bit
        self.cumulative_choices = []
        self.level = level
        self.cell_map = cell_map # prefix adder state; N*N array;
        self.level_map = level_map
        self.fanout_map = np.zeros((self.input_bit, self.input_bit), dtype=np.int8)
        self.min_map = min_map
        self.reward = reward
        self.size = size
        self.delay = None
        self.area = None
        self.level_bound_delta = level_bound_delta
        self.level_bound = int(math.log2(input_bit) + 1 + level_bound_delta)
        assert self.cell_map.sum() - self.input_bit == self.size

        self.available_choice_list = []
        self.available_choice = 0
        self.update_available_choice()

        self.action = action
        self.step_num = step_num

    def get_represent_int(self) -> int:
        rep_int = 0
        for i in range(1, self.input_bit):
            for j in range(i):
                if self.cell_map[i, j] == 1:
                    rep_int = rep_int * 2 + 1
                else:
                    rep_int *= 2
        self.rep_int = rep_int
        return rep_int

    def output_cell_map(self, dir: str) -> None:
        run_verilog_mid_path = os.path.join(dir, "run_verilog_mid")
        if not os.path.exists(run_verilog_mid_path):
            os.mkdir(run_verilog_mid_path)
        fdot_save = open(
            f"{run_verilog_mid_path}/adder_{self.input_bit}b_{int(self.level_map.max())}_{int(self.cell_map.sum() - self.input_bit)}_{self.hash_value}.log",
            "w",
        )
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                fdot_save.write("{}".format(str(int(self.cell_map[i, j]))))
            fdot_save.write("\n")
        fdot_save.write("\n")
        fdot_save.close()

    def output_verilog(self, dir: str, file_name: str = None) -> None:
        verilog_mid_path = dir + "/run_verilog_mid"
        if not os.path.exists(verilog_mid_path):
            os.mkdir(verilog_mid_path)
        rep_int = self.get_represent_int()
        self.hash_value = hashlib.md5(str(rep_int).encode()).hexdigest()
        self.output_cell_map(dir)
        if file_name is None:
            file_name = verilog_mid_path + "/adder_{}b_{}_{}_{}.v".format(
                self.input_bit,
                int(self.level_map.max()),
                int(self.cell_map.sum() - self.input_bit),
                self.hash_value,
            )
        self.verilog_file_name = file_name.split("/")[-1]

        verilog_file = open(file_name, "w")
        verilog_file.write("module main(a,b,s,cout);\n")
        verilog_file.write("input [{}:0] a,b;\n".format(self.input_bit - 1))
        verilog_file.write("output [{}:0] s;\n".format(self.input_bit - 1))
        verilog_file.write("output cout;\n")
        wires = set()
        for i in range(self.input_bit):
            wires.add("c{}".format(i))

        for x in range(self.input_bit - 1, 0, -1):
            last_y = x
            for y in range(x - 1, -1, -1):
                if self.cell_map[x, y] == 1:
                    assert self.cell_map[last_y - 1, y] == 1
                    if y == 0:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y - 1, y))
                    else:
                        wires.add("g{}_{}".format(x, last_y))
                        wires.add("p{}_{}".format(x, last_y))
                        wires.add("g{}_{}".format(last_y - 1, y))
                        wires.add("p{}_{}".format(last_y - 1, y))
                        wires.add("g{}_{}".format(x, y))
                        wires.add("p{}_{}".format(x, y))
                    last_y = y

        for i in range(self.input_bit):
            wires.add("p{}_{}".format(i, i))
            wires.add("g{}_{}".format(i, i))
            wires.add("c{}".format(x))
        assert 0 not in wires
        assert "0" not in wires
        verilog_file.write("wire ")

        for i, wire in enumerate(wires):
            if i < len(wires) - 1:
                verilog_file.write("{},".format(wire))
            else:
                verilog_file.write("{};\n".format(wire))
        verilog_file.write("\n")

        for i in range(self.input_bit):
            verilog_file.write("assign p{}_{} = a[{}] ^ b[{}];\n".format(i, i, i, i))
            verilog_file.write("assign g{}_{} = a[{}] & b[{}];\n".format(i, i, i, i))

        for i in range(1, self.input_bit):
            verilog_file.write("assign g{}_0 = c{};\n".format(i, i))

        for x in range(self.input_bit - 1, 0, -1):
            last_y = x
            for y in range(x - 1, -1, -1):
                if self.cell_map[x, y] == 1:
                    assert self.cell_map[last_y - 1, y] == 1
                    if y == 0:  # add grey module
                        verilog_file.write(
                            "GREY grey{}(g{}_{}, p{}_{}, g{}_{}, c{});\n".format(
                                x, x, last_y, x, last_y, last_y - 1, y, x
                            )
                        )
                    else:
                        verilog_file.write(
                            f"BLACK black{x}_{y}(g{x}_{last_y}, p{x}_{last_y}, g{last_y-1}_{y}, p{last_y-1}_{y}, g{x}_{y}, p{x}_{y});\n"
                        )
                    last_y = y

        verilog_file.write("assign s[0] = a[0] ^ b[0];\n")
        verilog_file.write("assign c0 = g0_0;\n")
        verilog_file.write("assign cout = c{};\n".format(self.input_bit - 1))
        for i in range(1, self.input_bit):
            verilog_file.write("assign s[{}] = p{}_{} ^ c{};\n".format(i, i, i, i - 1))
        verilog_file.write("endmodule")
        verilog_file.write("\n\n")

        verilog_file.write(GREY_CELL)
        verilog_file.write("\n")
        verilog_file.write(BLACK_CELL)
        verilog_file.write("\n")
        verilog_file.close()

    def run_yosys(self, dir: str, target_delay: int, save_verilog: bool = False) -> None:
        verilog_mid_path = dir + "/run_verilog_mid"
        yosys_mid_path = dir + "/run_yosys_mid"
        yosys_script_path = dir + "/run_yosys_script"

        if not os.path.exists(yosys_mid_path):
            os.mkdir(yosys_mid_path)
        dst_file_name = os.path.join(
            yosys_mid_path, self.verilog_file_name.split(".")[0] + "_yosys.v"
        )
        file_name_prefix = self.verilog_file_name.split(".")[0] + "_yosys"
        if os.path.exists(dst_file_name):
            return
        src_file_path = os.path.join(verilog_mid_path, self.verilog_file_name)

        if not os.path.exists(yosys_script_path):
            os.mkdir(yosys_script_path)
        abc_constr_file_name = os.path.join(
            yosys_script_path, "abc_constr"
        )
        fopen = open(abc_constr_file_name, "w")
        fopen.write(abc_constr_format.format())
        fopen.close()

        yosys_script_file_name = os.path.join(
            yosys_script_path, "{}.ys".format(file_name_prefix)
        )
        fopen = open(yosys_script_file_name, "w")
        fopen.write(yosys_script_format.format(src_file_path, lib, target_delay, abc_constr_file_name, lib, dst_file_name))
        fopen.close()
        _ = subprocess.check_output(
            ["yosys {}".format(yosys_script_file_name)], shell=True
        )
        if not save_verilog:
            os.remove(src_file_path)

    # openroad std
    def run_openroad(self, dir: str):
        yosys_file_name = os.path.join(
            dir, "run_yosys_mid", self.verilog_file_name.split(".")[0] + "_yosys.v"
        )

        # 1. write openroad_sta.tcl
        openroad_sta_path = os.path.join(dir, "openroad_sta")
        if not os.path.exists(openroad_sta_path):
            os.mkdir(openroad_sta_path)
        openroad_sta_tcl_file_name = os.path.join(
            openroad_sta_path, "openroad_sta.tcl"
        )
        fopen = open(openroad_sta_tcl_file_name, "w")
        fopen.write(openroad_sta_tcl.format(lef, lib, yosys_file_name))
        fopen.close()

        # open sta cmd
        sta_cmd = f'source {OpenRoadFlowPath}/env.sh\n' + f'openroad {openroad_sta_tcl_file_name} | tee ./log' # openroad sta
        
        # remove files cmd
        rm_log_cmd = 'rm -f ' + './log'
        rm_netlist_cmd = 'rm -f ' + f'{yosys_file_name}'
        rm_sta_tcl = 'rm -f ' + f'{openroad_sta_tcl_file_name}'

        # 2. execute sta cmd
        os.system(sta_cmd)

        # 3. get ppa from log
        with open('./log', 'r') as f:
            rpt = f.read().splitlines()
            for line in rpt:
                if len(line.rstrip()) < 2:
                    continue
                print("\rline", line)
                line = line.rstrip().split()
                if line[0] == 'wns':
                    delay = line[-1]
                    #delay = delay[1:]
                    continue
                if line[0] == 'Design':
                    area = line[2]
                    break
        ppa_dict = {
            "area": float(area),
            "delay": float(delay)
        }
        self.delay = ppa_dict["delay"]
        self.area = ppa_dict["area"]
        self.power = 0
        # remove log
        os.system(rm_log_cmd)
        os.system(rm_netlist_cmd)
        os.system(rm_sta_tcl)
        
        return ppa_dict["delay"], ppa_dict["area"], 0

    # def run_openroad(self, dir: str):
    #     global result_cache
    #     global cache_hit

    #     def substract_results(p):
    #         lines = p.split("\n")[-15:]
    #         area = -100.0
    #         wslack = -100.0
    #         power = 0.0
    #         note = None
    #         for line in lines:
    #             if not line.startswith("result:") and not line.startswith("Total"):
    #                 continue
    #             print("line", line)
    #             if "design_area" in line:
    #                 area = float(line.split(" = ")[-1])
    #             elif "worst_slack" in line:
    #                 wslack = float(line.split(" = ")[-1])
    #                 note = lines
    #             elif "Total" in line:
    #                 power = float(line.split()[-2])

    #         return area, wslack, power, note

    #     file_name_prefix = self.verilog_file_name.split(".")[0]
    #     hash_idx = file_name_prefix.split("_")[-1]
    #     if hash_idx in result_cache:
    #         delay = result_cache[hash_idx]["delay"]
    #         area = result_cache[hash_idx]["area"]
    #         power = result_cache[hash_idx]["power"]
    #         cache_hit += 1
    #         self.delay = delay
    #         self.area = area
    #         self.power = power
    #         return delay, area, power
    #     verilog_file_path = "{}/OpenROAD/test/adder_tmp_{}.v".format(
    #         OpenRoadPath, file_name_prefix
    #     )
    #     yosys_file_name = os.path.join(
    #         dir, "run_yosys_mid", self.verilog_file_name.split(".")[0] + "_yosys.v"
    #     )
    #     shutil.copyfile(yosys_file_name, verilog_file_path)

    #     sdc_file_path = "{}/OpenROAD/test/adder_nangate45_{}.sdc".format(
    #         OpenRoadPath, file_name_prefix
    #     )
    #     fopen_sdc = open(sdc_file_path, "w")
    #     fopen_sdc.write(sdc_format)
    #     fopen_sdc.close()
    #     fopen_tcl = open(
    #         "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(
    #             OpenRoadPath, file_name_prefix
    #         ),
    #         "w",
    #     )
    #     fopen_tcl.write(
    #         openroad_tcl.format(
    #             "adder_tmp_{}.v".format(file_name_prefix),
    #             "adder_nangate45_{}.sdc".format(file_name_prefix),
    #         )
    #     )
    #     fopen_tcl.close()

    #     command = "openroad {}/OpenROAD/test/adder_nangate45_{}.tcl".format(
    #         OpenRoadPath, file_name_prefix
    #     )
    #     print("COMMAND: {}".format(command))
    #     output = subprocess.check_output(
    #         [
    #             "openroad",
    #             "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(
    #                 OpenRoadPath, file_name_prefix
    #             ),
    #         ],
    #         cwd="{}/OpenROAD/test".format(OpenRoadPath),
    #     ).decode("utf-8")
    #     note = None
    #     retry = 0
    #     area, wslack, power, note = substract_results(output)
    #     while note is None and retry < 3:
    #         output = subprocess.check_output(
    #             [
    #                 "openroad",
    #                 "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(
    #                     OpenRoadPath, file_name_prefix
    #                 ),
    #             ],
    #             shell=True,
    #             cwd="{}/OpenROAD/test".format(OpenRoadPath),
    #         ).decode("utf-8")
    #         area, wslack, power, note = substract_results(output)
    #         retry += 1
    #     if os.path.exists(yosys_file_name):
    #         os.remove(yosys_file_name)
    #     if os.path.exists(
    #         "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(
    #             OpenRoadPath, file_name_prefix
    #         )
    #     ):
    #         os.remove(
    #             "{}/OpenROAD/test/adder_nangate45_{}.tcl".format(
    #                 OpenRoadPath, file_name_prefix
    #             )
    #         )
    #     if os.path.exists(
    #         "{}/OpenROAD/test/adder_nangate45_{}.sdc".format(
    #             OpenRoadPath, file_name_prefix
    #         )
    #     ):
    #         os.remove(
    #             "{}/OpenROAD/test/adder_nangate45_{}.sdc".format(
    #                 OpenRoadPath, file_name_prefix
    #             )
    #         )
    #     if os.path.exists(
    #         "{}/OpenROAD/test/adder_tmp_{}.v".format(OpenRoadPath, file_name_prefix)
    #     ):
    #         os.remove(
    #             "{}/OpenROAD/test/adder_tmp_{}.v".format(OpenRoadPath, file_name_prefix)
    #         )
    #     delay = 3.0 - wslack
    #     delay *= 1000
    #     self.delay = delay
    #     self.area = area
    #     self.power = power
    #     result_cache[hash_idx] = {"delay": delay, "area": area, "power": power}
    #     return delay, area, power

    def update_available_choice(self):
        up_tri_mask = np.triu(
            np.ones((self.input_bit, self.input_bit), dtype=np.int8), k=1
        )
        self.prob = np.ones((2, self.input_bit, self.input_bit), dtype=np.int8)
        self.prob[0] = np.where(self.cell_map >= 1.0, 0, self.prob[0])
        self.prob[0] = np.where(up_tri_mask >= 1.0, 0, self.prob[0])
        self.prob[1] = np.where(self.min_map <= 0.0, 0, self.prob[1])
        self.prob[1] = np.where(up_tri_mask >= 1.0, 0, self.prob[1])

        self.available_choice_list = []
        cnt = 0
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                if self.prob[1, i, j] == 1:
                    self.available_choice_list.append(
                        self.input_bit**2 + i * self.input_bit + j
                    )
                    cnt += 1
        for i in range(self.input_bit):
            for j in range(self.input_bit):
                if self.prob[0, i, j] == 1:
                    self.available_choice_list.append(i * self.input_bit + j)
                    cnt += 1
        self.available_choice = cnt

    def is_terminal(self) -> bool:
        if self.available_choice == 0:
            return True
        return False

    def legalize(
        self, cell_map: List[List[int]], min_map: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        # legalization function?
        min_map = copy.deepcopy(cell_map)
        for i in range(self.input_bit):
            min_map[i, 0] = 0
            min_map[i, i] = 0
        for x in range(self.input_bit - 1, 0, -1):
            last_y = x
            for y in range(x - 1, -1, -1):
                if cell_map[x, y] == 1:
                    cell_map[last_y - 1, y] = 1
                    min_map[last_y - 1, y] = 0
                    last_y = y
        return cell_map, min_map

    def update_fanout_map(self) -> None:
        self.fanout_map.fill(0)
        self.fanout_map[0, 0] = 0
        for x in range(1, self.input_bit):
            self.fanout_map[x, x] = 0
            last_y = x
            for y in range(x - 1, -1, -1):
                if self.cell_map[x, y] == 1:
                    self.fanout_map[last_y - 1, y] += 1
                    self.fanout_map[x, last_y] += 1
                    last_y = y

    def update_level_map(
        self, cell_map: List[List[int]], level_map: List[List[int]]
    ) -> List[List[int]]:
        level_map[1:].fill(0)
        level_map[0, 0] = 1
        for x in range(1, self.input_bit):
            level_map[x, x] = 1
            last_y = x
            for y in range(x - 1, -1, -1):
                if cell_map[x, y] == 1:
                    level_map[x, y] = (
                        max(level_map[x, last_y], level_map[last_y - 1, y]) + 1
                    )
                    last_y = y
        return level_map

    def copy(self):
        new_state = copy.deepcopy(self)
        return new_state

    def compute_reward(
        self,
        weight_area=1,
        weight_delay=1,
        ppa_scale=1,
        area_scale=1,
        delay_scale=1,
    ):
        return -ppa_scale * (
            weight_delay * self.delay / delay_scale
            + weight_area * self.area / area_scale
        )

    def get_next_state_with_random_choice(self, dir: str):
        global global_step
        global record_num
        try_step = 0
        min_metric = 1e10
        while self.available_choice > 0 and (
            (self.initial_adder_type != 0 and try_step < 4)
            or (self.initial_adder_type == 0 and try_step < 4)
        ):
            sample_prob = np.ones((self.available_choice))
            choice_idx = np.random.choice(
                self.available_choice,
                size=1,
                replace=False,
                p=sample_prob / sample_prob.sum(),
            )[0]
            random_choice = self.available_choice_list[choice_idx]
            action_type = random_choice // (self.input_bit**2)
            x = (random_choice % (self.input_bit**2)) // self.input_bit
            y = (random_choice % (self.input_bit**2)) % self.input_bit
            next_cell_map = copy.deepcopy(self.cell_map)
            next_min_map = np.zeros((self.input_bit, self.input_bit))
            next_level_map = np.zeros((self.input_bit, self.input_bit))

            if action_type == 0:
                assert next_cell_map[x, y] == 0
                next_cell_map[x, y] = 1
                next_cell_map, next_min_map = self.legalize(next_cell_map, next_min_map)
            elif action_type == 1:
                assert self.min_map[x, y] == 1
                assert self.cell_map[x, y] == 1
                next_cell_map[x, y] = 0
                next_cell_map, next_min_map = self.legalize(next_cell_map, next_min_map)
            next_level_map = self.update_level_map(next_cell_map, next_level_map)
            next_level = next_level_map.max()
            next_size = next_cell_map.sum() - self.input_bit
            next_step_num = self.step_num + 1
            action = random_choice
            reward = 0

            next_state = State(
                next_level,
                next_size,
                next_cell_map,
                next_level_map,
                next_min_map,
                next_step_num,
                action,
                reward,
                self.level_bound_delta,
                self.input_bit,
                self.initial_adder_type,
            )

            next_state.output_verilog(dir)
            next_state.run_yosys(dir, self.target_delay)
            delay, area, power = next_state.run_openroad(dir)
            global_step += 1
            print("delay = {}, area = {}".format(delay, area))

            next_state.delay = delay
            next_state.area = area
            next_state.power = power
            next_state.update_fanout_map()
            print("try_step = {}".format(try_step))
            try_step += 1

            if self.initial_adder_type == 0:
                if next_state.area < min_metric:
                    best_next_state = copy.deepcopy(next_state)
                    min_metric = next_state.area
            else:
                if next_state.area + next_state.delay <= min_metric:
                    best_next_state = copy.deepcopy(next_state)
                    min_metric = next_state.area + next_state.delay
            find = True
            if self.initial_adder_type == 0:
                if next_state.area <= self.area:
                    pass
                else:
                    find = False
            if self.initial_adder_type == 1 or self.initial_adder_type == 2:
                if next_state.area + next_state.delay <= self.area + self.delay:
                    pass
                else:
                    find = False
            if find is False:
                self.available_choice_list.remove(random_choice)
                self.available_choice -= 1
                assert self.available_choice == len(self.available_choice_list)
                continue
            self.cumulative_choices.append(action)
            return next_state

        return best_next_state

    def __repr__(self):
        return f"{self.cell_map}"

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
    def __init__(
        self,
        seed,
        q_policy,
        build_path="build",
        bit_width=8,
        target_delay=1000,
        weight_area=4,
        weight_delay=1,
        ppa_scale=100,
        area_scale=100.55,
        delay_scale=1404.668754404116,
        initial_state_pool_max_len=0,
        is_from_pool=False,
        task_index=0,
        best_state=None,  # 初始状态
        best_ppa=1e5,
        best_area=1e5,
        best_delay=1e5,
        initial_adder_type=1,
        **env_kwargs,
    ):
        super().__init__(seed, **env_kwargs)

        self.last_area = None
        self.last_delay = None
        self.last_ppa = None
        self.weight_area = weight_area
        self.weight_delay = weight_delay
        self.ppa_scale = ppa_scale

        self.delay_scale = None
        self.area_scale = None
        self.target_delay = target_delay

        self.total_steps = 0
        self.bit_width = bit_width
        self.cur_state = None
        self.found_best_info = {
            "found_best_ppa": best_ppa,
            "found_best_state": best_state,
            "found_best_area": best_area,
            "found_best_delay": best_delay,
        }

        self.q_policy = q_policy
        if q_policy is not None:
            self.device = q_policy.device

        self.task_index = task_index
        self.initial_cwd_path = os.getcwd()
        self.build_path = os.path.join(self.initial_cwd_path, build_path)
        if not os.path.exists(self.build_path):
            os.mkdir(self.build_path)

        # initial state pool
        # TODO-xxl
        self.initial_state_pool_max_len = initial_state_pool_max_len
        self.is_from_pool = is_from_pool
        self.initial_adder_type = initial_adder_type
        if initial_state_pool_max_len > 0:
            self.initial_state_pool = deque([], maxlen=initial_state_pool_max_len)
            init_state = self.reset(self.initial_adder_type)
            self.initial_state_pool.append(copy.deepcopy(init_state))

    def get_normal_init(self) -> State:
        cell_map = np.zeros((self.bit_width, self.bit_width))
        level_map = np.zeros((self.bit_width, self.bit_width))
        for i in range(self.bit_width):
            cell_map[i, i] = 1
            cell_map[i, 0] = 1
            level_map[i, i] = 1
            level_map[i, 0] = i + 1
        level = level_map.max()
        min_map = copy.deepcopy(cell_map)
        for i in range(self.bit_width):
            min_map[i, i] = 0
            min_map[i, 0] = 0
        size = cell_map.sum() - self.bit_width
        state = State(
            level, size, cell_map, level_map, min_map, 0, 0, 0, 0, self.bit_width, 0
        )
        return state

    def get_sklansky_init(self) -> State:
        cell_map = np.zeros((self.bit_width, self.bit_width))
        level_map = np.zeros((self.bit_width, self.bit_width))
        for i in range(self.bit_width):
            cell_map[i, i] = 1
            level_map[i, i] = 1
            t = i
            now = i
            x = 1
            level = 1
            while t > 0:
                if t % 2 == 1:
                    last_now = now
                    now -= x
                    cell_map[i, now] = 1
                    level_map[i, now] = max(level, level_map[last_now - 1, now]) + 1
                    level += 1
                t = t // 2
                x *= 2

        min_map = copy.deepcopy(cell_map)
        for i in range(self.bit_width):
            min_map[i, i] = 0
            min_map[i, 0] = 0

        level = level_map.max()
        size = cell_map.sum() - self.bit_width
        state = State(
            level, size, cell_map, level_map, min_map, 0, 0, 0, 0, self.bit_width, 1
        )
        state.cell_map, state.min_map = state.legalize(cell_map, min_map)
        state.update_available_choice()
        return state

    def get_brent_kung_init(self) -> State:
        def update_level_map(cell_map, level_map):
            level_map.fill(0)
            level_map[0, 0] = 1
            for x in range(1, self.bit_width):
                level_map[x, x] = 1
                last_y = x
                for y in range(x - 1, -1, -1):
                    if cell_map[x, y] == 1:
                        level_map[x, y] = (
                            max(level_map[x, last_y], level_map[last_y - 1, y]) + 1
                        )
                        last_y = y
            return level_map

        cell_map = np.zeros((self.bit_width, self.bit_width))
        level_map = np.zeros((self.bit_width, self.bit_width))
        for i in range(self.bit_width):
            cell_map[i, i] = 1
            cell_map[i, 0] = 1
        t = 2
        while t < self.bit_width:
            for i in range(t - 1, self.bit_width, t):
                cell_map[i, i - t + 1] = 1
            t *= 2
        level_map = update_level_map(cell_map, level_map)
        level = level_map.max()
        min_map = copy.deepcopy(cell_map)
        for i in range(self.bit_width):
            min_map[i, i] = 0
            min_map[i, 0] = 0
        size = cell_map.sum() - self.bit_width
        print(
            "BK level ={}, size = {}".format(
                level_map.max(), cell_map.sum() - self.bit_width
            )
        )
        state = State(
            level, size, cell_map, level_map, min_map, 0, 0, 0, 0, self.bit_width, 2
        )
        return state

    def reset(self, initial_adder_type: int, is_from_pool=False) -> State:
        if is_from_pool:
            sel_indexes = range(0, len(self.initial_state_pool))
            sel_index = random.sample(sel_indexes, 1)[0]
            init_state = self.initial_state_pool[sel_index]            
        else:
            if initial_adder_type == 0:
                init_state = self.get_normal_init()
            elif initial_adder_type == 1:
                init_state = self.get_sklansky_init()
            else:
                init_state = self.get_brent_kung_init()

        areas = []
        delays = []
        powers = []
        for td in self.target_delay:
            init_state.output_verilog(self.build_path)
            init_state.run_yosys(self.build_path, td)
            delay, area, power = init_state.run_openroad(self.build_path)
            areas.append(area)
            delays.append(delay)
            powers.append(power)
        area = np.mean(areas)
        delay = np.mean(delays)
        power = np.mean(powers)

        if self.area_scale is None:
            self.area_scale = area
            self.delay_scale = delay

        self.last_delay = delay
        self.last_area = area
        self.last_ppa = self.weight_area * (area / self.area_scale) + self.weight_delay * (
            delay / self.delay_scale
        )
        self.last_ppa *= self.ppa_scale 
        print(f"last ppa: {self.last_ppa}")
        self.cur_state = init_state

        return init_state

    def transist(self, action: int) -> State:
        state = self.cur_state
        # action space: int; type * x * y
        action_type = action // (state.input_bit**2)
        x = (action % (state.input_bit**2)) // state.input_bit
        y = (action % (state.input_bit**2)) % state.input_bit
        next_cell_map = copy.deepcopy(state.cell_map)
        next_min_map = np.zeros((state.input_bit, state.input_bit))
        next_level_map = np.zeros((state.input_bit, state.input_bit))

        if action_type == 0:
            assert next_cell_map[x, y] == 0
            next_cell_map[x, y] = 1
            next_cell_map, next_min_map = state.legalize(next_cell_map, next_min_map)
        elif action_type == 1:
            assert state.min_map[x, y] == 1
            assert state.cell_map[x, y] == 1
            next_cell_map[x, y] = 0
            next_cell_map, next_min_map = state.legalize(next_cell_map, next_min_map)
        next_level_map = state.update_level_map(next_cell_map, next_level_map)
        next_level = next_level_map.max()
        next_size = next_cell_map.sum() - state.input_bit
        next_step_num = state.step_num + 1
        reward = 0
        next_state = State(
            next_level,
            next_size,
            next_cell_map,
            next_level_map,
            next_min_map,
            next_step_num,
            action,
            reward,
            state.level_bound_delta,
            state.input_bit,
            state.initial_adder_type,
        )

        self.cur_state = next_state.copy()
        return next_state

    def step(self, action: int):
        # 状态转移
        next_state = self.transist(action)

        # 仿真
        areas = []
        delays = []
        powers = []
        for td in self.target_delay:
            next_state.output_verilog(self.build_path)
            next_state.run_yosys(self.build_path, td)
            delay, area, power = next_state.run_openroad(self.build_path)
            areas.append(area)
            delays.append(delay)
            powers.append(power)
        area = np.mean(areas)
        delay = np.mean(delays)
        power = np.mean(powers)
        # 获取奖励
        rewards_dict = {"delay": delay, "area": area, "power": power}
        ppa = self.weight_area * (area / self.area_scale) + self.weight_delay * (
            delay / self.delay_scale
        )
        ppa = self.ppa_scale * ppa

        reward = self.last_ppa - ppa
        self.last_area = area
        self.last_delay = delay
        self.last_ppa = ppa

        rewards_dict["avg_ppa"] = ppa
        rewards_dict["reward"] = reward

        return next_state, reward, rewards_dict

    def get_ppa_full_delay_cons(self, test_state):
        # get area delay list
        input_width = math.ceil(self.bit_width)
        target_delay = []
        if input_width == 8:
            for i in range(50,1000,10):
                target_delay.append(i)
        elif input_width == 16:
            for i in range(50,2000,20):
                target_delay.append(i)
        elif input_width == 31:
            for i in range(50,2000,20):
                target_delay.append(i)
        elif input_width == 64: 
            for i in range(50,2000,20):
                target_delay.append(i)
        ppas_dict = {
            "area": [],
            "delay": [],
            "power": []
        }
        for td in target_delay:
            test_state.output_verilog(self.build_path)
            test_state.run_yosys(self.build_path, td)
            delay, area, power = test_state.run_openroad(self.build_path)
            ppas_dict["area"].append(area)
            ppas_dict["delay"].append(delay)
            ppas_dict["power"].append(power)
            
        return ppas_dict

class RefineEnvGA(RefineEnv):
    def block_crossover(self, state1:State, state2:State):
        """
        直接交换某一行之前的所有行
        """
        cell_map_1 = copy.deepcopy(state1.cell_map)
        cell_map_2 = copy.deepcopy(state2.cell_map)

        input_bit = len(cell_map_1)
        random_index_pool = list(range(0, input_bit))

        while len(random_index_pool) > 0:
            selected_row = np.random.choice(random_index_pool)

            cell_map_1_selected = cell_map_1[:selected_row]
            cell_map_2_selected = cell_map_2[:selected_row]

            if np.array_equal(cell_map_1_selected, cell_map_2_selected):
                random_index_pool.remove(selected_row)
                continue
            else:
                # 交换
                print("block_crossover:selected row: ", selected_row)
                cell_map_1[:selected_row] = cell_map_2_selected
                cell_map_2[:selected_row] = cell_map_1_selected
                cell_map_1 = cell_map_legalize(cell_map_1)
                cell_map_2 = cell_map_legalize(cell_map_2)

                # 构造新状态
                new_level_map_1 = np.zeros((input_bit, input_bit))
                new_level_map_1 = state1.update_level_map(cell_map_1, new_level_map_1)
                new_level_1 = new_level_map_1.max()
                new_min_map_1 = np.zeros((input_bit, input_bit))
                cell_map_1, new_min_map_1 = state1.legalize(cell_map_1, new_min_map_1)
                step_num_1 = state1.step_num + 1
                new_size_1 = cell_map_1.sum() - state1.input_bit
                new_state_1 = State(
                    new_level_1,
                    new_size_1,
                    cell_map_1,
                    new_level_map_1,
                    new_min_map_1,
                    step_num_1,
                    -1,
                    0,
                    state1.level_bound_delta,
                    state1.input_bit,
                    state1.initial_adder_type,
                )

                new_level_map_2 = np.zeros((input_bit, input_bit))
                new_level_map_2 = state2.update_level_map(cell_map_2, new_level_map_2)
                new_level_2 = new_level_map_2.max()
                new_min_map_2 = np.zeros((input_bit, input_bit))
                cell_map_2, new_min_map_2 = state2.legalize(cell_map_2, new_min_map_2)
                step_num_2 = state2.step_num + 1
                new_size_2 = cell_map_2.sum() - state2.input_bit
                new_state_2 = State(
                    new_level_2,
                    new_size_2,
                    cell_map_2,
                    new_level_map_2,
                    new_min_map_2,
                    step_num_2,
                    -1,
                    0,
                    state1.level_bound_delta,
                    state1.input_bit,
                    state1.initial_adder_type,
                )

                return new_state_1, new_state_2
        
        # 没能找到，说明这俩是一模一样的
        print("block_crossover:cannot change")
        return None, None
    
    def column_crossover(self, state1:State, state2:State):
        """
        直接交换某一行
        """
        cell_map_1 = copy.deepcopy(state1.cell_map)
        cell_map_2 = copy.deepcopy(state2.cell_map)

        input_bit = len(cell_map_1)
        random_index_pool = list(range(0, input_bit))

        while len(random_index_pool) > 0:
            selected_row = np.random.choice(random_index_pool)

            cell_map_1_selected = cell_map_1[selected_row]
            cell_map_2_selected = cell_map_2[selected_row]

            if np.array_equal(cell_map_1_selected, cell_map_2_selected):
                random_index_pool.remove(selected_row)
                continue
            else:
                # 交换
                print("column_crossover:selected row: ", selected_row)
                cell_map_1[selected_row] = cell_map_2_selected
                cell_map_2[selected_row] = cell_map_1_selected
                cell_map_1 = cell_map_legalize(cell_map_1)
                cell_map_2 = cell_map_legalize(cell_map_2)

                # 构造新状态
                new_level_map_1 = np.zeros((input_bit, input_bit))
                new_level_map_1 = state1.update_level_map(cell_map_1, new_level_map_1)
                new_level_1 = new_level_map_1.max()
                new_min_map_1 = np.zeros((input_bit, input_bit))
                cell_map_1, new_min_map_1 = state1.legalize(cell_map_1, new_min_map_1)
                step_num_1 = state1.step_num + 1
                new_size_1 = cell_map_1.sum() - state1.input_bit
                new_state_1 = State(
                    new_level_1,
                    new_size_1,
                    cell_map_1,
                    new_level_map_1,
                    new_min_map_1,
                    step_num_1,
                    -1,
                    0,
                    state1.level_bound_delta,
                    state1.input_bit,
                    state1.initial_adder_type,
                )

                new_level_map_2 = np.zeros((input_bit, input_bit))
                new_level_map_2 = state2.update_level_map(cell_map_2, new_level_map_2)
                new_level_2 = new_level_map_2.max()
                new_min_map_2 = np.zeros((input_bit, input_bit))
                cell_map_2, new_min_map_2 = state2.legalize(cell_map_2, new_min_map_2)
                step_num_2 = state2.step_num + 1
                new_size_2 = cell_map_2.sum() - state2.input_bit
                new_state_2 = State(
                    new_level_2,
                    new_size_2,
                    cell_map_2,
                    new_level_map_2,
                    new_min_map_2,
                    step_num_2,
                    -1,
                    0,
                    state1.level_bound_delta,
                    state1.input_bit,
                    state1.initial_adder_type,
                )

                return new_state_1, new_state_2
        
        # 没能找到，说明这俩是一模一样的
        print("column_crossover:cannot change")
        return None, None

if __name__ == "__main__":
    # from o2_policy_adder import DeepQPolicy, BasicBlock
    # from ipdb import set_trace
    # bit_width = 31
    # q_policy = DeepQPolicy(BasicBlock, device="cpu", bit_width=bit_width)

    # target_q_policy = DeepQPolicy(BasicBlock, device="cpu", bit_width=bit_width)
    # env = RefineEnv(
    #     0, q_policy, bit_width=bit_width,
    #     area_scale=74, delay_scale=0.8376
    # )
    # state = env.reset(1)
    # # print(state.cell_map)
    # # print(state.available_choice_list)

    # set_trace()

    # available_choice_list = state.available_choice_list
    # next_state, reward, rewards_dict = env.step(available_choice_list[0])

    # # print(next_state)
    # print(reward)

    # args npy_data_path
    # import argparse
    from paretoset import paretoset
    from pygmo import hypervolume
    # parser = argparse.ArgumentParser(description="Testing Hypervolume")
    # parser.add_argument('--npy_data_path', type=str, default='xxx')
    # parser.add_argument('--bit_width', type=int, default=32)
    # parser.add_argument('--reference_area', type=float, default=700)
    # parser.add_argument('--reference_delay', type=float, default=0.8)
    # parser.add_argument('--build_path', type=str, default="./build/dqn32")
    

    # args = parser.parse_args()
    # # load state
    # npy_data = np.load(
    #     args.npy_data_path, allow_pickle=True
    # ).item()
    # test_state = npy_data["found_best_info"]["found_best_state"]

    # # get area delay list
    # input_width = math.ceil(args.bit_width)
    # target_delay = []
    # if input_width == 8:
    #     for i in range(50,1000,10):
    #         target_delay.append(i)
    # elif input_width == 16:
    #     for i in range(50,2000,20):
    #         target_delay.append(i)
    # elif input_width == 32: 
    #     for i in range(50,2000,20):
    #         target_delay.append(i)
    # elif input_width == 64: 
    #     for i in range(50,2000,20):
    #         target_delay.append(i)
    # area_list = []
    # delay_list = []
    # for td in target_delay:
    #     test_state.output_verilog(args.build_path)
    #     test_state.run_yosys(args.build_path, td)
    #     delay, area, power = test_state.run_openroad(args.build_path)
    #     area_list.append(area)
    #     delay_list.append(delay)

    # combine_array = []
    # for i in range(len(area_list)):
    #     point = [area_list[i], delay_list[i]]
    #     combine_array.append(point)
    # combine_array = np.array(combine_array)
    # # compute hypervolume
    # hv = hypervolume(combine_array)
    # hv_value = hv.compute([args.reference_area, args.reference_delay])

    # print(f"area list: {area_list}")
    # print(f"delay list: {delay_list}")
    # print(f"hypervolume: {hv_value}")

    # 32 bits
    # npy_data_path_have = "outputs/2024-08-06/00-26-52/logger_log/dqn_32bits_factorq_reset/dqn_adder/dqn_adder_2024_08_06_00_27_00_0000--s-3404/itr_2525.npy"
    # npy_data_path_mute = "outputs/2024-08-06/00-26-39/logger_log/dqn_32bits_factorq_reset_ea/dqn_adder/dqn_adder_2024_08_06_00_26_48_0000--s-3391/itr_2525.npy"
    # 64 bits
    npy_data_path_have = "outputs/2024-08-06/00-27-54/logger_log/dqn_64bits_factorq_reset/dqn_adder/dqn_adder_2024_08_06_00_28_03_0000--s-3466/itr_3775.npy"
    npy_data_path_mute = "outputs/2024-08-06/00-28-00/logger_log/dqn_64bits_factorq_reset_ea/dqn_adder/dqn_adder_2024_08_06_00_28_09_0000--s-3472/itr_4400.npy"

    npy_data_paths = [
        npy_data_path_have,
        npy_data_path_mute
    ]

    for npy_path in npy_data_paths:
        data = np.load(
            npy_path,
            allow_pickle=True
        ).item()
        pareto_points_area = data["testing_pareto_data"]["testing_pareto_points_area"]
        pareto_points_delay = data["testing_pareto_data"]["testing_pareto_points_delay"]
        combine_array = []
        for i in range(len(pareto_points_area)):
            point = [pareto_points_area[i], pareto_points_delay[i]]
            combine_array.append(point)
        combine_array = np.array(combine_array)
        hv = hypervolume(combine_array)
        hv_value = hv.compute([1200,1.2]) # [700,0.8] [1200,1.2]
        print(hv_value)