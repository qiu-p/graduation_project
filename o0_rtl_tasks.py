import copy
import json
import logging
import multiprocessing
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

from o0_mul_utils import (
    FA_src_list,
    HA_src_list,
    legal_FA_list,
    legal_HA_list,
)

is_source_env = False

# OpenRoadFlowPath = "/home/xiaxilin/MiraLab/OpenROAD-flow-scripts"
# lef = "/home/xiaxilin/MiraLab/nips2024/ai4-multiplier-master/data/NangateOpenCellLibrary.lef"
# lib = "/home/xiaxilin/MiraLab/nips2024/ai4-multiplier-master/data/NangateOpenCellLibrary_typical.lib"

lef = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary.lef'
lib = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary_typical.lib'
OpenRoadFlowPath = '/datasets/ai4multiplier/openroad_deb/OpenROAD-flow-scripts'

CURRENT_FILE_DIR = os.path.dirname(__file__)

lef = os.path.join(CURRENT_FILE_DIR, "dataset", 'NangateOpenCellLibrary.lef')
lib = os.path.join(CURRENT_FILE_DIR, "dataset", 'NangateOpenCellLibrary_typical.lib')
OpenRoadFlowPath = '/datasets/ai4multiplier/openroad_deb/OpenROAD-flow-scripts'


synth_script_template = """
read -sv {}
synth -top {}
dfflibmap -liberty {}
abc -D {} -constr {} -liberty {}
write_verilog {}
"""

abc_constr = """
set_driving_cell BUF_X1
set_load 10.0 [all_outputs]
"""

sta_script_template_basic = """
read_lef {}
read_lib {}
read_verilog {}
link_design {}

set period 5
create_clock -period $period [get_ports clock]

set clk_period_factor .2

set clk [lindex [all_clocks] 0]
set period [get_property $clk period]
set delay [expr $period * $clk_period_factor]
set_input_delay $delay -clock $clk [delete_from_list [all_inputs] [all_clocks]]
set_output_delay $delay -clock $clk [delete_from_list [all_outputs] [all_clocks]]
"""

sta_script_template_ppa = """
set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts "wns $path_delay"
report_design_area

set_power_activity -input -activity 0.5
report_power
"""

sta_script_template_activity = """
set nets [get_nets C0/data*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}

set nets [get_nets C0/out*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}

set nets [get_nets out*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}
"""

sta_script_template_activity_prefix_adder = """
set nets [get_nets prefix_adder_0/*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}
"""

sta_script_template_power = """
foreach inst [get_cells C0/FA*] {
    report_power -instance $inst
}

foreach inst [get_cells C0/HA*] {
    report_power -instance $inst
}
"""

sta_script_template_power_prefix_adder = """
foreach inst [get_cells prefix_adder_0/*] {
    report_power -instance $inst
}
"""

sta_script_template_power_prefix_adder_top = """
foreach inst [get_cells *] {
    report_power -instance $inst
}
"""

sta_script_template_end = """
exit
"""

sta_script_template_compressor_power = """
read_lef {}
read_lib {}
read_verilog {}
link_design {}

set period 5
create_clock -period $period [get_ports clock]

set clk_period_factor .2

set clk [lindex [all_clocks] 0]
set period [get_property $clk period]
set delay [expr $period * $clk_period_factor]
set_input_delay $delay -clock $clk [delete_from_list [all_inputs] [all_clocks]]
set_output_delay $delay -clock $clk [delete_from_list [all_outputs] [all_clocks]]

set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts "wns $path_delay"
report_design_area

set_power_activity -input -activity 0
set_power_activity -input_port {} -activity {}
report_power

exit
"""


class EvaluateWorker:
    """
    用于根据需要完成rtl级代码的评估等任务
    """

    def __init__(
        self,
        rtl_path: str,  # Verilog 文件路径
        target_lists: list,  # 评估目标列表 见下面的注释
        target_delay_list: list,
        worker_path: str,  # build 的位置
        clear_dir: bool = True,  # 完成后是否清理文件夹
        clear_log: bool = True,  # 完成后是否清理log文件
        clear_output: bool = True,  # 是否清理被重定向的out文件
        clear_netlist: bool = True,  # 是否清理被重定向的out文件
        use_existing_netlist: bool = False,  # 是否使用现有的netlist文件
        use_existing_log: bool = False,  # 是否使用现有的log文件
        n_processing: int = 1,  # 线程数量
        use_prefix_adder: bool = False,
        top_name: str = "MUL",
    ) -> None:
        """
        target_lists = [
            "ppa",          # 获取全局的 power area delay 结果
            "activity",     # 获取全局的 switching 信息
            "power"         # 获取每个模块上的 power 信息
            "prefix_adder"         # 获取和 adder 相关的信息
            "prefix_adder_power"
        ]
        """
        self.rtl_path = rtl_path
        self.target_lists = target_lists
        self.target_delay_list = target_delay_list
        self.worker_path = worker_path
        self.n_processing = n_processing
        self.clear_dir = clear_dir
        self.clear_log = clear_log
        self.clear_netlist = clear_netlist
        self.clear_output = clear_output
        self.use_existing_netlist = use_existing_netlist
        self.use_existing_log = use_existing_log
        self.use_prefix_adder = use_prefix_adder
        self.top_name = top_name

        self.results = None

    @staticmethod
    def evaluate_worker(
        worker_path_prefix: str,
        rtl_path: str,
        target_delay: int,
        target_list: list,
        clear_dir: bool = True,
        clear_log: bool = True,
        clear_output: bool = True,
        clear_netlist: bool = True,
        use_existing_netlist: bool = False,
        use_existing_log: bool = False,
        worker_id: int = 0,
        top_name="MUL",
    ):
        assert os.path.exists(rtl_path), f"not exists {rtl_path}"
        worker_path = f"{worker_path_prefix}/worker_{worker_id}"
        if not os.path.exists(worker_path):
            os.makedirs(worker_path)

        netlist_path = os.path.join(worker_path, "netlist.v")
        abc_constr_path = os.path.join(worker_path, "abc_constr")
        yosys_path = os.path.join(worker_path, "yosys.ys")
        sta_script_path = os.path.join(worker_path, "sta.tcl")
        log_path = os.path.join(worker_path, "log")
        yosys_out_path = os.path.join(worker_path, "yosys_out")

        # fmt: off
        synth_flag: bool = not use_existing_netlist or not os.path.exists(netlist_path) or not (use_existing_log and os.path.exists(use_existing_log))
        if synth_flag: # 完成综合
            synth_script = synth_script_template.format(rtl_path, top_name, lib, target_delay, abc_constr_path, lib, netlist_path)
            with open(yosys_path, "w") as file: file.write(f"{synth_script}")
            with open(abc_constr_path, "w") as file: file.write(f"{abc_constr}")
            yosys_cmd = f"yosys {yosys_path} > {yosys_out_path} 2>&1"
            logging.info(f"worker-{worker_id} synthing")
            os.system(yosys_cmd)
        
        sta_flag = synth_flag or not (use_existing_log and os.path.exists(use_existing_log))
        if sta_flag: # 完成仿真
            sta_script = sta_script_template_basic.format(lef, lib, netlist_path, top_name)
            if "ppa" in target_list:
                sta_script += sta_script_template_ppa
            if "activity" in target_list:
                sta_script += sta_script_template_activity
                if "prefix_adder" in target_list:
                    sta_script += sta_script_template_activity_prefix_adder
            if "power" in target_list:
                sta_script += sta_script_template_power
            if "prefix_adder_power" in target_list:
                if top_name == "MUL":
                    sta_script += sta_script_template_power_prefix_adder
                else:
                    sta_script += sta_script_template_power_prefix_adder_top
            sta_script += sta_script_template_end
            with open(sta_script_path, "w") as file: file.write(f"{sta_script}")
            logging.info(f"worker-{worker_id} simulating")
            if is_source_env:
                sta_cmd = (
                    f"source {OpenRoadFlowPath}/env.sh > {log_path} 2>&1\n"
                    + f"openroad {sta_script_path} > {log_path}"
                )
            else:
                sta_cmd = (
                    f"openroad {sta_script_path} > {log_path}"
                )
            os.system(sta_cmd)
        
        result = {}
        result["target_delay"] = target_delay # 多线程的时候target delay会乱 所以需要记录一下
        if "ppa" in target_list:
            with open(log_path) as file:
                rpt = file.read().splitlines()
                for line in rpt:
                    if len(line.rstrip()) < 2: continue
                    line = line.rstrip().split()
                    if line[0] == "wns": delay = line[-1]
                    if line[0] == "Design": area = line[2]
                    if line[0] == "Total":
                        power = line[-2]
                        break
            result["ppa"] = {"delay": float(delay), "area": float(area), "power": float(power)}
        if "activity" in target_list:
            activity_info = {}
            with open(log_path) as file:
                rpt = file.read().splitlines()
                line_index = 0
                while line_index < len(rpt):
                    line = rpt[line_index].split(" ")
                    if line[0] == "net_name":
                        if "/" in line[1]:
                            net_name = line[1].split("/")[-1]
                        else:
                            net_name = line[1]
                        if ("out" in net_name and "_C" not in net_name and "C0" not in line[1] or "prefix_adder_0" in line[1]):
                            line_index += 1
                            continue
                        next_line_index = line_index + 1
                        if next_line_index >= len(rpt): break
                        find_info_flag = False
                        while "net_name" not in rpt[next_line_index]:
                            if "FA" in rpt[next_line_index] or "HA" in rpt[next_line_index] or "PD0" in rpt[next_line_index]:
                                next_line = rpt[next_line_index].split(" ")
                                frequency = float(next_line[3])
                                duty = float(next_line[4])
                                activity_info[net_name] = {"frequency": frequency, "duty": duty}
                                find_info_flag = True
                                break
                            next_line_index += 1
                        if not find_info_flag:
                            activity_info[net_name] = {"frequency": 0, "duty": 0}
                    line_index += 1
            result["activity"] = activity_info
        if "power" in target_list:
            power_info_dict = {}
            with open(log_path) as file:
                rpt = file.read().splitlines()
                for line in rpt:
                    word_list = line.split(" ")
                    if "FA" in word_list[-1] or "HA" in word_list[-1]:
                        full_name = word_list[-1]
                        name = full_name.split("/")[1]
                        power = float(word_list[-2])
                        if name in power_info_dict.keys(): power_info_dict[name] += power
                        else: power_info_dict[name] = power
            result["power"] = power_info_dict

            if "prefix_adder" in target_list:
                prefix_adder_power = 0.0
                with open(log_path) as file:
                    rpt = file.read().splitlines()
                    for line in rpt:
                        word_list = line.split(" ")
                        if "prefix_adder_0" in word_list[-1] and "net_name" not in line:
                            power = float(word_list[-2])
                            prefix_adder_power += power
                result["prefix_adder_overall_power"] = prefix_adder_power


        if "prefix_adder_power" in target_list:
            pattern = r"cell_(\d+)_(\d+)_"
            prefix_adder_power_dict = {}
            with open(log_path) as file:
                rpt = file.read().splitlines()
                for line in rpt:
                    if "cell" in line and "prefix_adder_0" in line and "pin" not in line and "net_name" not in line:
                        word_list = line.split(" ")
                        match = re.search(pattern, word_list[-1])
                        x = int(match.group(1))
                        y = int(match.group(2))
                        cell_power = float(word_list[-2])
                        if (x, y) in prefix_adder_power_dict.keys():
                            prefix_adder_power_dict[(x, y)] += cell_power
                        else:
                            prefix_adder_power_dict[(x, y)] = cell_power
            result["prefix_adder_power"] = prefix_adder_power_dict

        clear_output = clear_output or clear_dir
        clear_log = clear_log or clear_dir
        clear_netlist = clear_netlist or clear_dir
        if clear_output:
            try: os.remove(yosys_out_path)
            except: pass
        if clear_log:
            try: os.remove(log_path)
            except: pass
        if clear_netlist:
            try: os.remove(netlist_path)
            except: pass
        if clear_dir:
            try:
                os.remove(abc_constr_path)
                os.remove(yosys_path)
                os.remove(sta_script_path)
                os.removedirs(worker_path)
            except: pass
        # fmt: on
        return result

    def evaluate(self):
        param_list = [
            (
                self.worker_path,
                self.rtl_path,
                self.target_delay_list[i],
                self.target_lists,
                self.clear_dir,
                self.clear_log,
                self.clear_output,
                self.clear_netlist,
                self.use_existing_netlist,
                self.use_existing_log,
                i,
                self.top_name,
            )
            for i in range(len(self.target_delay_list))
        ]
        if self.n_processing > 1:
            with multiprocessing.Pool(self.n_processing) as pool:
                results = pool.starmap_async(self.evaluate_worker, param_list)
                pool.close()
                pool.join()
            results = results.get()
        else:
            # results = [self.evaluate_worker(*(param_list[0]))]
            results = [self.evaluate_worker(*(param)) for param in param_list]
        self.results = results
        #for i, result in enumerate(results):
        #    print(i)
        #    print(result['power'].__class__)
        #    print(result['power'])
        return results

    def update_wire_constant_dict(self, wire_constant_dict):
        """
        由于仿真的时候连到vcc和gnd的net不会出现在结果中
        所以需要在生成Verilog的时候就把他们保存起来
        然后通过本函数更新到结果中
        """
        assert self.results is not None
        for i in range(len(self.results)):
            assert "activity" in self.results[i].keys()
            self.results[i]["activity"].update(wire_constant_dict)

    def consult_compressor_power(
        self,
        compressor_type_index,
        stage_index,
        column_index,
        compressor_index,
        column_num,
    ):
        """
        获取给定的类型 阶段 列数 位置的乘法器的实例总功耗
        Parameters:
            compressor_type_index: 0 for FA, 1 for HA
            stage_index: 阶段数
            column_index: 列数
            compressor_index: 压缩器位置
        """
        power = 0.0
        rtl_column_index = column_num - column_index - 1
        compressor_type = ["FA", "HA"][compressor_type_index]
        key = (
            f"{compressor_type}_s{stage_index}_c{rtl_column_index}_i{compressor_index}"
        )
        for result in self.results:
            power += result["power"][key]
        power /= len(self.results)
        return power

    def consult_compressor_power_mask(self, ct_decomposed) -> np.ndarray:
        power_mask = np.zeros_like(ct_decomposed)
        stage_num = len(ct_decomposed[0])
        column_len = len(ct_decomposed[0][0])
        # fmt: off
        for type_index in range(2):
            for stage_index in range(stage_num):
                for column_index in range(column_len):
                    ct_num = int(ct_decomposed[type_index][stage_index][column_index])
                    power = 0.0
                    for compressor_index in range(ct_num):
                        power += self.consult_compressor_power(type_index, stage_index, column_index, compressor_index, column_len)
                    power_mask[type_index][stage_index][column_index] = power
        # fmt: on
        return power_mask

    def consult_port_activity(
        self,
        stage_index: int,
        column_index: int,
        wire_num: int,
        bit_width: int,
        stage_num: int,
        column_len: int,
        wire_connect_dict: int,
    ) -> dict:
        """
        从仿真数据中获取 pp 的 switching 信息
        """
        assert self.results is not None
        switching = np.zeros([wire_num])
        rtl_column_index = column_len - column_index - 1
        # fmt: off
        for index in range(wire_num):
            freq = 0.0
            for result in self.results:
                assert "activity" in result.keys()
                activity_info_dict = result["activity"]
                if stage_index == 0:
                    port_name = f"out{rtl_column_index}[{index}]"
                else:
                    port_name = f"data{rtl_column_index}_s{stage_index}[{index}]"
                if port_name in activity_info_dict.keys():
                    freq += activity_info_dict[port_name]["frequency"]
                else:
                    if port_name in wire_connect_dict.keys():
                        port_name_root = wire_connect_dict[port_name]
                    else:
                        port_name_root = port_name
                    if (port_name_root not in activity_info_dict.keys() and port_name_root[0]):
                        # 遇到了这种情况只有一根线时, 用out0命名而不是out0[0], 导致查找 key 时找不到
                        port_name_root = re.sub(r"\[.*?\]", "", port_name_root)
                    if port_name_root not in activity_info_dict.keys():
                        # 那么大概率因为某些奇怪的原因，wire 被映射到其他地方了。那么现在只能手动遍历了
                        try_flag = False
                        if port_name[0] == "o":
                            port_name_root = port_name
                            try_column_index = re.search(r"out(\d+)", port_name).group(1)
                        else:
                            if port_name in wire_connect_dict.keys():
                                port_name_root = wire_connect_dict[port_name]
                            else:
                                port_name_root = port_name
                            if port_name_root[0] == "o":
                                try_column_index = re.search(r"out(\d+)", port_name_root).group(1)
                            else:
                                try_column_index = re.search(r"data(\d+)_", port_name_root).group(1)
                        for try_index in range(bit_width):
                            for try_stage_index in range(1, stage_num):
                                try_name = f"data{try_column_index}_s{try_stage_index}[{try_index}]"
                                if (try_name in activity_info_dict.keys() and try_name in wire_connect_dict.keys() and wire_connect_dict[try_name] == port_name_root):
                                    # 找到了
                                    port_name_root = try_name
                                    try_flag = True
                                    break
                            if try_flag:
                                break
                    if port_name_root not in activity_info_dict.keys():
                        # 还是没找到，说明被送入MUL中成为了 MUL下的out_x_C
                        if port_name_root[0] == "o":
                            # try_column_index = re.search(r'out(\d+)', port_name).group(1)
                            try_column_index = re.search(r"out(\d+)", port_name_root).group(1)
                        else:
                            try_column_index = re.search(r"data(\d+)_", port_name_root).group(1)
                        try_stage_index = stage_num
                        try_flag = False
                        for try_index in range(bit_width):
                            try_name = f"data{try_column_index}_s{try_stage_index}[{try_index}]"
                            if (try_name in wire_connect_dict.keys() and wire_connect_dict[try_name] == port_name_root):
                                try_flag = True
                                break
                        assert try_flag == True
                        port_name_root = f"out{try_column_index}_C[{try_index}]"
                        if port_name_root not in activity_info_dict.keys():
                            port_name_root = re.sub(r"\[.*?\]", "", port_name_root)
                    if port_name_root not in activity_info_dict.keys():
                        freq += 1e8
                    else:
                        freq += activity_info_dict[port_name_root]["frequency"]

            switching[index] = freq / len(self.results)
        # fmt: on
        return switching

    def consult_ppa(self):
        """
        返回的是在几个 target delay 上的平均值
        """
        assert self.results is not None
        ppa_dict = {"area": 0.0, "delay": 0.0, "power": 0.0}
        for result in self.results:
            for key in ppa_dict.keys():
                ppa_dict[key] += result["ppa"][key]
        for key in ppa_dict.keys():
            ppa_dict[key] /= len(self.results)
        return ppa_dict

    def consult_ppa_list(self):
        """
        返回的是所有 target delay 上的原始值
        """
        assert self.results is not None
        ppa_dict = {"area": [], "delay": [], "power": []}
        for result in self.results:
            for key in ppa_dict.keys():
                ppa_dict[key].append(result["ppa"][key])
        return ppa_dict

    def consult_cell_power(self, x, y):
        cell_power = 0.0
        for result in self.results:
            if (x, y) in result["prefix_adder_power"].keys():
                cell_power += result["prefix_adder_power"][(x, y)]
        cell_power /= len(self.results)
        return cell_power

    def consult_cell_power_mask(self, bit_width):
        power_mask = np.zeros([bit_width, bit_width], float)
        for x in range(1, bit_width):
            for y in range(0, x):
                power_mask[x, y] = self.consult_cell_power(x, y)
        return power_mask


class PowerSlewConsulter:
    def __init__(
        self,
        db_path: str = "./db/power_slew.json",
        use_db: bool = True,
        build_dir: str = "pybuild/power_slew",
    ):
        """
        创建一个 power slew consulter 对象
        首先它会去读取 db_path 中的数据
        如果不存在 就会去仿真得到数据并且存到 db_path
        """
        simulation_flag = not use_db
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            simulation_flag = True
        if simulation_flag:  # 如果不存在db 就去仿真它
            if not os.path.exists(build_dir):
                os.makedirs(build_dir)

            compressor_power_slew_list = [[], []]

            port_name_list = [["a", "b", "cin"], ["a", "cin"]]
            src_list = [FA_src_list, HA_src_list]
            legal_compressor_list = [legal_FA_list, legal_HA_list]
            for compressor_type_index in range(2):
                compressor_list = legal_compressor_list[compressor_type_index]
                for fa_index, fa_type in enumerate(compressor_list):
                    port_power_slew = []
                    # fmt: off
                    for port_index, port_name in enumerate(port_name_list[compressor_type_index]):
                        rtl_path = os.path.join(build_dir, "fa.v")
                        netlist_path = os.path.join(build_dir, "netlist.v")
                        abc_constr_path = os.path.join(build_dir, "abc_constr")
                        yosys_path = os.path.join(build_dir, "yosys.ys")
                        sta_script_path = os.path.join(build_dir, "sta.tcl")
                        log_path = os.path.join(build_dir, "log")
                        yosys_out_path = os.path.join(build_dir, "yosys_out")

                        with open(abc_constr_path, "w") as file:
                            file.write(f"{abc_constr}")
                        with open(rtl_path, "w") as file:
                            verilog_src = src_list[compressor_type_index][fa_index]
                            verilog_src_with_clock = self.add_clock_port_and_input(verilog_src)
                            file.write(f"{verilog_src_with_clock}")
                        synth_script = synth_script_template.format(rtl_path, fa_type, lib, 50, abc_constr_path, lib, netlist_path)
                        with open(yosys_path, "w") as file:
                            file.write(f"{synth_script}")
                        sta_script = sta_script_template_compressor_power.format(lef, lib, netlist_path, fa_type, port_name, 1)
                        # fmt: on
                        with open(sta_script_path, "w") as file:
                            file.write(f"{sta_script}")
                        yosys_cmd = f"yosys {yosys_path} > {yosys_out_path} 2>&1"
                        if is_source_env:
                            sta_cmd = (
                                f"source {OpenRoadFlowPath}/env.sh > {log_path} 2>&1\n"
                                + f"openroad {sta_script_path} > {log_path}"
                            )
                        else:
                            sta_cmd = (
                                f"openroad {sta_script_path} > {log_path}"
                            )
                        logging.info(f"simulating {fa_type}:{port_name}")
                        os.system(yosys_cmd)
                        os.system(sta_cmd)

                        with open(log_path) as file:
                            rpt = file.read().splitlines()
                            for line in rpt:
                                if len(line.rstrip()) < 2:
                                    continue
                                line = line.rstrip().split()
                                if line[0] == "Total":
                                    power = float(line[-2])
                        port_power_slew.append(power)
                    compressor_power_slew_list[compressor_type_index].append(port_power_slew)
            with open(db_path, "w") as file:
                json.dump(compressor_power_slew_list, file)
        else:
            logging.info(f"reading db {db_path}")
            with open(db_path) as file:
                compressor_power_slew_list = json.load(file)

        self.compressor_power_slew_list = compressor_power_slew_list

    def add_clock_port_and_input(self, verilog_src):
        # 匹配以 FA 或 HA 开头的模块声明和端口列表
        module_pattern = re.compile(
            r"(module\s+(FA|HA)[\w]*\s*\((.*?)\);(.*?endmodule))", re.DOTALL
        )

        def add_clock_to_module(match):
            # 模块的原始定义
            original_module = match.group(0)
            # 模块的端口列表
            ports = match.group(3)
            # 模块的主体部分（端口定义后到endmodule之间的代码）
            module_body = match.group(4)

            # 检查端口列表中是否已经包含 clock
            if "clock" not in ports:
                # 添加 clock 到端口列表的最前面
                updated_ports = f"clock, {ports}"
                original_module = original_module.replace(ports, updated_ports)

            # 检查 module_body 中是否已经有 input clock; 声明
            if "input clock;" not in module_body:
                # 查找 input 部分并在最后一个 input 声明后添加 input clock;
                updated_module_body = re.sub(
                    r"(input\s+.*?;)(?!.*input\s+clock;)",
                    r"\1\n    input clock;",
                    module_body,
                    count=1,
                    flags=re.DOTALL,
                )
                # 替换原始的 module_body
                original_module = original_module.replace(
                    module_body, updated_module_body
                )

            return original_module

        # 只替换 FA 或 HA 开头的模块部分
        updated_verilog_src = module_pattern.sub(add_clock_to_module, verilog_src)
        return updated_verilog_src

    def consult_power_slew(
        self, ct32_num, ct22_num, compressor_map, stage_index, column_index
    ):
        # fmt: off
        power_slew_list = []
        for compressor_index in range(ct32_num):
            compressor_type_index = int(compressor_map[0][stage_index][column_index])
            assert compressor_type_index != -1
            power_slew_list.append(self.compressor_power_slew_list[0][compressor_type_index][0])
            power_slew_list.append(self.compressor_power_slew_list[0][compressor_type_index][1])
            power_slew_list.append(self.compressor_power_slew_list[0][compressor_type_index][2])
        for compressor_index in range(int(ct22_num)):
            compressor_type_index = int(compressor_map[1][stage_index][column_index])
            assert compressor_type_index != -1
            power_slew_list.append(self.compressor_power_slew_list[1][compressor_type_index][0])
            power_slew_list.append(self.compressor_power_slew_list[1][compressor_type_index][1])
        return power_slew_list
        # fmt: on


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
    )
    worker = EvaluateWorker(
        "./verilate/MUL.v",
        ["ppa", "activity", "power"],
        [50, 250, 400, 650],
        "pybuild/worker_test",
        False,
        False,
        False,
        False,
        False,
        False,
        4,
    )

    worker.evaluate()
    print(worker.consult_ppa())
