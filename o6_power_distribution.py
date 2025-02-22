import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import multiprocessing

from o0_mul_utils import get_initial_partial_product, legalize_compressor_tree
from o0_rtl_tasks import EvaluateWorker
from o0_state import State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
)

def get_activity_dist(bit_width=16, ct_type="dadda", encode_type="and", final_adder_init_type="default", build_base="pybuild"):
    method_key = f"{bit_width}_{encode_type}_{ct_type}_{final_adder_init_type}"
    build_path = os.path.join(build_base, method_key)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    rtl_path = os.path.join(build_path, "MUL.v")
    state = State(bit_width, encode_type, 2 * bit_width, True, "default", True, "default", True, final_adder_init_type)
    state.init(ct_type)
    state.emit_verilog(rtl_path)

    worker = EvaluateWorker(rtl_path, ["power", "ppa", "activity", "prefix_adder", "prefix_adder_power"], [100], build_path, False, False, False, False)
    worker.evaluate()
    print(method_key, ": ", worker.consult_ppa())
    state.pp_wiring_arrangement_v0(worker=worker)
    state.emit_verilog(rtl_path)
    worker.evaluate()
    print(method_key, ": ", worker.consult_ppa())

    out_width = 2 * bit_width - 1
    frequency_arr = np.zeros([2, out_width])
    duty_arr = np.zeros([2, out_width])
    for bit_pos in range(out_width):
        if f"out{bit_pos}_C[0]" in worker.results[0]["activity"].keys():
            frequency_arr[0, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[0]"]["frequency"]
            duty_arr[0, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[0]"]["duty"]
        if f"out{bit_pos}_C[1]" in worker.results[0]["activity"].keys():
            frequency_arr[1, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[1]"]["frequency"]
            duty_arr[1, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[1]"]["duty"]
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(out_width), frequency_arr[0], "--o", label="bit array 0")
    plt.plot(range(out_width), frequency_arr[1], "--o", label="bit array 1")
    plt.title("Switching freq")
    plt.xlabel("bit position")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(out_width), duty_arr[0], "--o", label="bit array 0")
    plt.plot(range(out_width), duty_arr[1], "--o", label="bit array 1")
    plt.title("Duty")
    plt.xlabel("bit position")
    plt.legend()

    plt.suptitle(f"{bit_width}_{encode_type}_{ct_type}_routed")
    plt.tight_layout()
    plt.savefig(f"report/2024-12-08/{method_key}-routed.png")
    # plt.show()

def get_power_dist(bit_width=16, ct_type="dadda", encode_type="and", final_adder_init_type="default", build_base="pybuild"):
    method_key = f"{bit_width}_{encode_type}_{ct_type}_{final_adder_init_type}"
    build_path = os.path.join(build_base, method_key)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    rtl_path = os.path.join(build_path, "MUL.v")
    state = State(bit_width, encode_type, 2 * bit_width, True, "default", True, "default", True, final_adder_init_type)
    state.init(ct_type)
    state.emit_verilog(rtl_path)

    if bit_width == 16:
        target_delay = [50,200,500,1200]
    else:
        target_delay = [50,300,600,2000]

    worker = EvaluateWorker(rtl_path, ["power", "ppa", "prefix_adder", "prefix_adder_power"], target_delay, build_path, False, False, False, False, top_name="PrefixAdder")
    worker.evaluate()
    adder_power = worker.consult_ppa()["power"]

    worker = EvaluateWorker(rtl_path, ["power", "ppa", "prefix_adder", "prefix_adder_power"], target_delay, build_path, False, False, False, False, top_name="MUL")
    worker.evaluate()
    mul_power = worker.consult_ppa()["power"]
    mul_adder_power = 0.0
    for item in worker.results:
        mul_adder_power += item["prefix_adder_overall_power"]
    mul_adder_power /= len(worker.results)

    print(method_key, ": ", adder_power, mul_power, mul_adder_power)

    return {
        "method": method_key,
        "mul-overall": mul_power,
        "adder-only": adder_power,
        "adder-in-mul": mul_adder_power,
    }

def main1():
    save_list = []
    for bit_width in [8, 16, 32]:
        for ct_type in ["dadda", "wallace"]:
            for encode_type in ["and", "booth"]:
                for method in ["default", "brent_kung", "sklansky", "kogge_stone", "han_carlson"]:
                    save_dict = get_power_dist(bit_width, ct_type, encode_type, method, "pybuild/power_dist")
                    save_list.append(save_dict)
                
                    with open("report/2024-12-08/power_dist.json", "w") as file:
                        json.dump(save_list, file)
    df = pandas.DataFrame.from_dict(save_list)
    df.to_csv("report/2024-12-08/power_dist.csv")


def get_activity_different_ct_worker(bit_width, encode_type, ct, index, build_base="pybuild/activity_different_ct"):
    build_path = os.path.join(build_base, f"worker-{index}")
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    rtl_path = os.path.join(build_path, "MUL.v")
    state = State(bit_width, encode_type, 2 * bit_width, True, "default", True, "default", False, "default")
    state.ct = ct
    state.get_initial_pp_wiring()
    state.get_initial_compressor_map()
    state.get_init_cell_map()
    state.emit_verilog(rtl_path)

    worker = EvaluateWorker(rtl_path, ["ppa", "activity", "power", "prefix_adder", "prefix_adder_power"], [100], build_path, False, False, False, False)

    worker.evaluate()
    out_width = len(ct[0])
    frequency_arr = np.full([2, out_width], -1.0)
    duty_arr = np.full([2, out_width], -1.0)
    for bit_pos in range(out_width):
        if f"out{bit_pos}_C[0]" in worker.results[0]["activity"].keys():
            frequency_arr[0, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[0]"]["frequency"]
            duty_arr[0, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[0]"]["duty"]
        if f"out{bit_pos}_C[1]" in worker.results[0]["activity"].keys():
            frequency_arr[1, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[1]"]["frequency"]
            duty_arr[1, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C[1]"]["duty"]
        if f"out{bit_pos}_C" in worker.results[0]["activity"].keys():
            frequency_arr[0, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C"]["frequency"]
            duty_arr[0, bit_pos] = worker.results[0]["activity"][f"out{bit_pos}_C"]["duty"]

    return {
        "frequency": frequency_arr.tolist(),
        "duty": duty_arr.tolist(),
        "ct": np.asarray(ct).tolist(),
    }
    

def get_activity_different_ct(bit_width=16, encode_type="and", build_base="pybuild/activity_different_ct", num=2, n_processing=1, report_path = "report/2024-12-12"):
    param_list = []
    key = f"{bit_width}_{encode_type}"
    for i in range(num):
        pp = get_initial_partial_product(bit_width, encode_type)
        ct = np.random.randint(0, bit_width, [2, len(pp)])
        ct = legalize_compressor_tree(pp, ct[0], ct[1])
        param_list.append((bit_width, encode_type, ct, i, os.path.join(build_base, key)))
    
    if n_processing > 1:
        with multiprocessing.Pool(n_processing) as pool:
            result = pool.starmap_async(get_activity_different_ct_worker, param_list)
            pool.close()
            pool.join()

        result = result.get()
    else:
        result = []
        for param in param_list:
            result.append(get_activity_different_ct_worker(*param))

    if not os.path.exists(report_path):
        os.makedirs(report_path)
    
    with open(os.path.join(report_path, f"result-{key}.json"), "w") as file:
        json.dump(result, file)

if __name__ == "__main__":
    get_activity_different_ct(8)