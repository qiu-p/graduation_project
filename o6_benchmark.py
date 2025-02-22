import os
from o0_rtl_tasks import lef, lib, OpenRoadFlowPath
import logging
import re
import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
)

abc_constr = """
set_driving_cell BUF_X1
set_load 10.0 [all_outputs]
"""

yosys_script_template = """
{}

synth -top {}
dfflibmap -liberty {}
abc {} -constr {} -liberty {}
write_verilog {}
"""

sta_script_template_head = """
read_lef {}
read_lib {}
read_verilog {}
link_design {}

#set_max_delay -from [all_inputs] 0


set clk_name  clock
set clk_port_name {}
set clk_period 1
"""

sta_script_template = """
set clk_port [get_ports $clk_port_name]

create_clock -name $clk_name -period $clk_period $clk_port

# set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
# set path_delay [sta::format_time [[$critical_path path] arrival] 4]
# puts "wns $path_delay"
# report_design_area

# report_checks -format full

report_power

foreach inst [get_cells u_matmul*] {
    report_power -instance $inst
}

exit
"""


def gen():
    build_path = "pybuild/benchmarks/conv"
    if not os.path.exists(build_path):
        os.makedirs(build_path)

    yosys_path = os.path.join(build_path, "yosys.ys")
    abc_constr_path = os.path.join(build_path, "abc_constr")
    sta_script_path = os.path.join(build_path, "sta.tcl")
    netlist_path = os.path.join(build_path, "netlist.v")
    yosys_out_path = os.path.join(build_path, "yosys.out")
    log_path = os.path.join(build_path, "log")

    # read_cmd = (
    #     "read -sv benchmarks/tpu_like/tpu_like.small.os.v\n"
    #     + "read -sv benchmarks/MUL/MUL-8.v\n"
    # )
    read_cmd = "read -sv benchmarks/kerios/conv_layer.v\n"
    yosys_script = yosys_script_template.format(
        read_cmd, "conv_layer", lib, "-D 100", abc_constr_path, lib, netlist_path
    )
    with open(yosys_path, "w") as file:
        file.write(yosys_script)
    with open(abc_constr_path, "w") as file:
        file.write(abc_constr)
    sta_script = (
        sta_script_template_head.format(lef, lib, netlist_path, "conv_layer", "clk")
        + sta_script_template
    )
    with open(sta_script_path, "w") as file:
        file.write(sta_script)

    logging.info("start synth")
    yosys_cmd = f"yosys {yosys_path} > {yosys_out_path} 2>&1"
    os.system(yosys_cmd)
    logging.info("start simulating")
    sta_cmd = (
        f"source {OpenRoadFlowPath}/env.sh > {log_path} 2>&1\n"
        + f"openroad {sta_script_path} > {log_path}"
    )
    os.system(sta_cmd)


# gen()


def process():
    with open("pybuild/benchmarks/tpu_small/log8", "r") as file:
        content = file.read().split("\n")
    power = np.zeros([16, 16])
    cnt = 0
    for line in content:
        if "mult_ours_u1" in line:
        # if "mult_ours_u1" in line or "add_u1" in line:
        # if "mult_u1" in line:
            cnt += 1
            match = re.search(r"u_systolic_pe_matrix/pe(\d+)_(\d+)/u_mac", line)
            i = int(match.group(1))
            j = int(match.group(2))
            if i == 4 and j == 1:
                print(line)
            word_list = line.split()
            # power[i, j] += float(word_list[-5]) + float(word_list[-4])
            power[i, j] += float(word_list[-5]) + float(word_list[-4])

    print(power[4, 1])
    print(np.sum(power))
    print(cnt)
    plt.imshow(power)
    plt.colorbar()
    plt.show()

def process1():
    with open("pybuild/benchmarks/tpu_small/log9", "r") as file:
        content = file.read().split("\n")
    u_norm_power = 0
    mult_power = 0
    add_power = 0
    u_mac_power = 0
    u_matmul_power = 0
    matrix_A = 0
    matrix_B = 0
    u_cfg = 0
    u_control = 0
    u_activation = 0
    u_pool = 0
    u_systolic_pe_matrix = 0
    for line in content:
        word_list = line.split()
        if len(word_list) < 1:
            continue
        if word_list[0] == "Total":
            total_power = float(word_list[-5]) + float(word_list[-4])
        # if "mult_u" in line or "Add_u" in line:
        if "mult_ours_u1" in line:
            mult_power += float(word_list[-5]) + float(word_list[-4])
        if "add_u1" in line:
            add_power += float(word_list[-5]) + float(word_list[-4])
        if "u_mac" in line:
            u_mac_power += float(word_list[-5]) + float(word_list[-4])
        if "u_norm" in line:
            u_norm_power += float(word_list[-5]) + float(word_list[-4])
        if "u_matmul" in line:
            u_matmul_power += float(word_list[-5]) + float(word_list[-4])
        if "matrix_A" in line:
            matrix_A += float(word_list[-5]) + float(word_list[-4])
        if "matrix_B" in line:
            matrix_B += float(word_list[-5]) + float(word_list[-4])
        if "u_cfg" in line:
            u_cfg += float(word_list[-5]) + float(word_list[-4])
        if "u_control" in line:
            u_control += float(word_list[-5]) + float(word_list[-4])
        if "u_activation" in line:
            u_activation += float(word_list[-5]) + float(word_list[-4])
        if "u_pool" in line:
            u_pool += float(word_list[-5]) + float(word_list[-4])
        if "u_systolic_pe_matrix" in line:
            u_systolic_pe_matrix += float(word_list[-5]) + float(word_list[-4])        

    print(total_power)
    print(f"mul: {mult_power:.2f}")
    print(f"mul: {mult_power / total_power * 100:.2f}%")

    print(f"add: {add_power:.2f}")
    print(f"add: {add_power / total_power * 100:.2f}%")

    print(f"mul+add: {mult_power + add_power:.2f}")
    print(f"mul+add: {(mult_power + add_power) / total_power * 100:.2f}%")

    print(f"umac: {u_mac_power:.2f}")
    print(f"umac: {u_mac_power / total_power * 100:.2f}%")

    print(f"u_norm: {u_norm_power:.2f}")
    print(f"u_norm: {u_norm_power / total_power * 100:.2f}%")

    print(f"u_matmul_power: {u_matmul_power:.2f}")
    print(f"u_matmul_power: {u_matmul_power / total_power * 100:.2f}%")

    print(f"matrix_A: {matrix_A:.2f}")
    print(f"matrix_A: {matrix_A / total_power * 100:.2f}%")

    print(f"matrix_B: {matrix_B:.2f}")
    print(f"matrix_B: {matrix_B / total_power * 100:.2f}%")

    print(f"u_cfg: {u_cfg:.2f}")
    print(f"u_cfg: {u_cfg / total_power * 100:.2f}%")

    print(f"u_control: {u_control:.2f}")
    print(f"u_control: {u_control / total_power * 100:.2f}%")

    print(f"u_activation: {u_activation:.2f}")
    print(f"u_activation: {u_activation / total_power * 100:.2f}%")

    print(f"u_pool: {u_pool:.2f}")
    print(f"u_pool: {u_pool / total_power * 100:.2f}%")

    
    print(f"u_systolic_pe_matrix: {u_systolic_pe_matrix:.2f}")
    print(f"u_systolic_pe_matrix: {u_systolic_pe_matrix / total_power * 100:.2f}%")

    print(f"ram + u_matmul % = {(u_pool + u_norm_power + u_activation + u_control + u_cfg + matrix_A + matrix_B + u_mac_power) / total_power * 100:.2f}%")
# process1()

# process()

def process_3():
    over_all = float("6.50e+04")
    # pattern = r'(?P<name>\S+) (?P<module_name>\S+) (?P<switching>\S+) (?P<internal>\S+) (?P<leakage>\S+) (?P<ratio>\S+)'
    pattern =pattern = r'(?P<name>\S+) (?P<module_name>\S+) (?P<x>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) (?P<y>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) (?P<z>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) (?P<w>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) (?P<u>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    with open("benchmarks/kerios/conv.rpt") as file:
        content = file.read()
        content = re.sub(r'\s+', ' ', content).strip()
    matches = re.findall(pattern, content)

    target_list = ["mult_u1", "add_u1", "u_mac"]
    target_dict = {}
    for target in target_list:
        target_dict[target] = 0.0
    for match in matches:
        name, module_name, x, y, z, w, u = match
        for target in target_list:
            if target == name:
                target_dict[target] += float(w)
    
    for target in target_list:
        print(f"{target}: {target_dict[target]:.2f}, {target_dict[target] / over_all * 100:.2f}%")
    print(f'mult+mul: {target_dict["mult_u1"] + target_dict["add_u1"]:.2f}, {(target_dict["mult_u1"] + target_dict["add_u1"]) / over_all * 100:.2f}%')

process_3()
