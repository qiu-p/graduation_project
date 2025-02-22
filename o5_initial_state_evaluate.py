from o1_environment_speedup import SpeedUpRefineEnvWithPower
from o0_global_const import PartialProduct
from o0_global_const import (
    InitialState as wallace_initial_state,
    DaddaInitialState as dadda_initial_state,
)
import numpy as np
import os
import copy
from paretoset import paretoset
from pygmo import hypervolume
import pandas as pd
import json
from o0_state import State
from o0_rtl_tasks import EvaluateWorker
from o0_mul_utils import get_initial_partial_product, legalize_compressor_tree


def get_hv(ppas_dict, reference_point):
    area_list = ppas_dict["area"]
    delay_list = ppas_dict["delay"]
    power_list = ppas_dict["power"]
    data_points = pd.DataFrame(
        {
            "area": area_list,
            "delay": delay_list,
            "power": power_list,
        }
    )
    pareto_mask = paretoset(data_points, sense=["min", "min", "min"])
    pareto_points = data_points[pareto_mask]
    true_pareto_area_list = pareto_points["area"].values.tolist()
    true_pareto_delay_list = pareto_points["delay"].values.tolist()
    true_pareto_power_list = pareto_points["power"].values.tolist()

    combine_array = []
    for i in range(len(true_pareto_area_list)):
        point = [
            true_pareto_area_list[i],
            true_pareto_delay_list[i],
            true_pareto_power_list[i],
        ]
        combine_array.append(point)
    hv = hypervolume(combine_array)
    hv_value = hv.compute(reference_point)

    return hv_value


def get_full_ppa():
    bit_width_list = [
        "8_bits_and",
        "16_bits_and",
        "32_bits_and",
        "64_bits_and",
    ]
    bit_width_list += [
        "8_bits_booth",
        "16_bits_booth",
        "32_bits_booth",
        "64_bits_booth",
    ]
    # bit_width_list = ["8_bits_and"]
    initial_state = {"wallace": wallace_initial_state,
                     "dadda": dadda_initial_state}
    target_delay_dict = {
        "8_bits_and": [50, 250, 400, 650],
        "8_bits_booth": [50, 250, 400, 650],
        "16_bits_and": [50, 200, 500, 1200],
        "16_bits_booth": [50, 200, 500, 1200],
        "32_bits_and": [50, 300, 600, 2000],
        "32_bits_booth": [50, 300, 600, 2000],
        "64_bits_and": [50, 600, 1500, 3000],
        "64_bits_booth": [50, 600, 1500, 3000],
    }
    method_list = ["wallace", "dadda"]
    # method_list = ["dadda"]
    # method_list = ["wallace"]
    # reference_point_list = {"8_bits_and": [550,2.5,1.5],"16_bits_and": [2800,4,10], "32_bits_and": [12000,4.5,100], "64_bits_and":[50000, 5, 600]}
    reference_point_list = {
        "8_bits_and": [550, 2.5, 1.5],
        "16_bits_and": [2800, 4, 10],
        "32_bits_and": [12000, 4.5, 100],
        "64_bits_and": [50000, 5, 700],
    }  # for wallace

    for method in method_list:
        for bit_width in bit_width_list:
            build_path = f"./build/2024-10-09/{bit_width}/{method}"
            report_path = f"./report_power_no_pp_arangement/{bit_width}_{method}.json"
            report_dir = os.path.dirname(report_path)
            if not os.path.exists(build_path):
                os.makedirs(build_path)
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            target_delay = target_delay_dict[bit_width]
            env = SpeedUpRefineEnvWithPower(
                1,
                None,
                mul_booth_file="mul.test2",
                bit_width=bit_width,
                target_delay=target_delay,
                initial_state_pool_max_len=0,
                wallace_area=((517 + 551 + 703 + 595) / 4),
                wallace_delay=((1.0827 + 1.019 + 0.9652 + 0.9668) / 4),
                pp_encode_type=bit_width.split("_")[2],
                load_pool_index=3,
                reward_type="simulate",
                load_initial_state_pool_npy_path="None",
                synthesis_type="v1",
                is_debug=False,
                build_path=build_path,
                synthesis_path=build_path + "/syn",
            )
            if True:
                state = initial_state[method][bit_width]
            else:
                state = 1
            # reference_point = reference_point_list[bit_width]

            env.cur_state = copy.deepcopy(state)
            initial_partial_product = PartialProduct[env.bit_width]
            print(env.get_final_partial_product(initial_partial_product))
            _ = env.decompose_compressor_tree(
                initial_partial_product[:-1], state)
            reward_dict = env.get_reward()
            print(reward_dict)
            print("power:", np.mean(reward_dict["power"]))
            print("area:", np.mean(reward_dict["area"]))
            print("delay:", np.mean(reward_dict["delay"]))
            # ppa_full_delay_cons = env.get_ppa_full_delay_cons(state)
            # hv = get_hv(ppa_full_delay_cons, reference_point)
            save_data = {
                "reward_dict": reward_dict,
                "avg_area": np.mean(reward_dict["area"]),
                "avg_delay": np.mean(reward_dict["delay"]),
                "avg_power": np.mean(reward_dict["power"]),
                "avg_power": np.mean(reward_dict["power"]),
                # "hv": hv,
                # "ppa_full_delay_cons": ppa_full_delay_cons,
            }
            with open(report_path, "w") as log_file:
                json.dump(save_data, log_file)


def get_normal_info():
    build_path_base = "pybuild/random_search"
    log_path_base = "log/random_search.json"
    target_delay_dict = {
        "8": [50, 250, 400, 650],
        "16": [50, 200, 500, 1200],
        "32": [50, 300, 600, 2000],
        "64": [50, 600, 1500, 3000],
    }
    from tqdm import tqdm
    save_dict = {}
    for bit_width in [8, 16, 32, 64]:
        for encode_type in ["and", "booth"]:
            key = f"{bit_width}bits_{encode_type}"
            build_path = os.path.join(build_path_base, key)
            ppa_dict_list = []
            pp = get_initial_partial_product(bit_width, encode_type)
            for index in tqdm(range(200)):
                rtl_path = os.path.join(build_path, f"rtl/MUL-{index}.v")
                synth_path = os.path.join(build_path, "synth")
                state = State(bit_width, encode_type, 2 * bit_width,
                              True, "default", True, "default", False, "default")
                ct = np.random.randint(0, bit_width * 2, size=[2, len(pp)])
                ct = legalize_compressor_tree(pp, ct[0], ct[1])
                ct = np.asarray(ct)
                state.ct = ct
                state.get_initial_compressor_map()
                state.get_initial_pp_wiring()

                worker = EvaluateWorker(rtl_path, [
                                        "ppa"], target_delay_dict[f"{bit_width}"], synth_path, False, False, False, False, False, False, 4)
                state.emit_verilog(rtl_path)
                worker.evaluate()
                ppa_dict = worker.consult_ppa()
                ppa_dict_list.append(ppa_dict)
            save_dict[key] = {}
            area_list = []
            delay_list = []
            power_list = []
            for ppa_dict in ppa_dict_list:
                area_list.append(ppa_dict["area"])
                delay_list.append(ppa_dict["delay"])
                power_list.append(ppa_dict["power"])
            save_dict[key]["area"] = area_list
            save_dict[key]["delay"] = delay_list
            save_dict[key]["power"] = power_list

            save_dict[key]["avg_area"] = np.mean(area_list)
            save_dict[key]["avg_delay"] = np.mean(delay_list)
            save_dict[key]["avg_power"] = np.mean(power_list)

            save_dict[key]["var_area"] = np.std(area_list)
            save_dict[key]["var_delay"] = np.std(delay_list)
            save_dict[key]["var_power"] = np.std(power_list)

            with open(log_path_base, "w") as file:
                json.dump(save_dict, file)


def process_norm_info():
    log_path_base = "log/random_search.json"
    with open(log_path_base, "r") as file:
        save_dict = json.load(file)

    for bit_width in [8, 16, 32, 64]:
        for encode_type in ["and", "booth"]:
            key = f"{bit_width}bits_{encode_type}"
            area_list = save_dict[key]["area"]
            delay_list = save_dict[key]["delay"]
            power_list = save_dict[key]["power"]
            save_dict[key]["var_area"] = np.std(area_list)
            save_dict[key]["var_delay"] = np.std(delay_list)
            save_dict[key]["var_power"] = np.std(power_list)
            import matplotlib.pyplot as plt
            plt_index = 1
            for ppa_key in ["area", "delay", "power"]:
                ppa_list = save_dict[key][ppa_key]
                ppa_arr = np.asarray(ppa_list)
                y = (ppa_arr - np.mean(ppa_arr)) / np.std(ppa_arr)
                plt.subplot(3, 1, plt_index)
                plt_index += 1
                plt.hist(y, bins=25, density=True)
                x = np.linspace(min(y), max(y))
                plt.plot(x, np.exp( - x ** 2 / 2) / np.sqrt(np.pi))
                plt.tight_layout()
            plt.show()
    with open("log/random_search-post-processed.json", "w") as file:
                json.dump(save_dict, file)


if __name__ == "__main__":
    # get_normal_info()
    process_norm_info()
