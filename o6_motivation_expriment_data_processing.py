import argparse
import copy
import json
import logging
import multiprocessing
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
from omegaconf import DictConfig, OmegaConf
from tensorboard.backend.event_processing import event_accumulator

from o0_mul_utils import (decompose_compressor_tree, get_compressor_tree,
                          get_initial_partial_product, write_mul)
from o0_rtl_tasks import EvaluateWorker
from o0_state import State

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
)

class DataProcessor:
    def __init__(
        self,
        ct_type,
        bit_width,
        encode_type,
        log_base_dir,
        db_base_dir,
    ) -> None:
        
        self.ct_type = ct_type
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.log_base_dir = log_base_dir
        self.db_base_dir = db_base_dir

        key = f"{bit_width}_{encode_type}_{ct_type}"
        self.key = key

        self.files = [os.path.join(log_base_dir, file) for file in os.listdir(log_base_dir) if key in file and "expr2" not in file and "db" not in file]

        def sort_key(file_name:str):
            index = int((file_name.split(".")[0]).split("_")[-1])
            return index
        self.files.sort(key=sort_key)

        if ct_type in ["wallace", "dadda"]:
            pp = get_initial_partial_product(bit_width, encode_type)
            self.ct = get_compressor_tree(pp, bit_width, ct_type)
            self.pp_wiring = None
        else:
            data = np.load(f"report/2020-11-8/state-{self.bit_width}-{self.encode_type}.npy", allow_pickle=True).item()
            state_data: State = data["state"]
            self.ct = state_data.ct
            self.pp_wiring = state_data.pp_wiring

    def process(self):
        data_list = []
        draw_index = 1
        save_data_list = []
        plt.figure(figsize=[14, 16])
        for file_path in self.files:
            with open(file_path, "r") as file:
                data = json.load(file)
                data_list.append(data)
                default_power_list = []
                routed_power_list = []
                loaded_power_list = []
                for item in data["result"]:
                    default_power_list.append(item["default"]["power"])
                    routed_power_list.append(item["routed"]["power"])
                    if "loaded" in item.keys():
                        loaded_power_list.append(item["loaded"]["power"])

                
                plt.subplot(4, 3, draw_index)
                draw_index += 1
                p_FA = data["params"]["P_FA"]
                p_HA = data["params"]["P_HA"]
                plt.title(f"p_FA = {np.round(p_FA, 3)}, p_HA = {np.round(p_HA, 3)}")
                save_data_dict = {
                    "p_FA": np.round(p_FA, 3).tolist(),
                    "p_HA": np.round(p_HA, 3).tolist(),
                }

                max_value = max(default_power_list)
                min_value = min(default_power_list)
                mean_value = np.mean(default_power_list)
                ratio = np.round((max_value - min_value) / mean_value, 3)
                print(f"default mean power = {mean_value}")
                save_data_dict.update({
                    "default_max": max_value,
                    "default_min": min_value,
                    "default_mean": mean_value,
                })

                plt.hist(default_power_list, density=False, alpha=0.5, label=f"default, (max - min)/mean={ratio}")

                max_value = max(routed_power_list)
                min_value = min(routed_power_list)
                mean_value = np.mean(routed_power_list)
                print(f"routed mean power = {mean_value}")
                save_data_dict.update({
                    "routed_max": max_value,
                    "routed_min": min_value,
                    "routed_mean": mean_value,
                })
                ratio = np.round((max_value - min_value) / mean_value, 3)
                plt.hist(routed_power_list, density=False, alpha=0.5, label=f"routed, (max - min)/mean={ratio}")
                
                if "load" in self.key:
                    max_value = max(loaded_power_list)
                    min_value = min(loaded_power_list)
                    mean_value = np.mean(loaded_power_list)
                    ratio = np.round((max_value - min_value) / mean_value, 3)
                    plt.hist(loaded_power_list, density=False, alpha=0.5, label=f"loaded, (max - min)/mean={ratio}")
                
                plt.legend()

                plt.tight_layout()
                save_data_list.append(save_data_dict)
        save_path = f"./report/2024-11-12/{self.key}.png"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.suptitle(self.key, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)

        df = pandas.DataFrame.from_dict(save_data_list)
        df.to_csv(os.path.join(save_dir, f"{self.key}.csv"))

    def get_default_value(self, build_base:str="pybuild/random/default_value"):
        save_dict = {}
        for FA_type in [0, 1, 2]:
            save_dict[FA_type] = {}
            for HA_type in [0, 1]:
                save_dict[FA_type][HA_type] = {}
                target_delays = {
                    16: [50, 200, 500, 1200],
                    32: [50, 300, 600, 2000],
                }
                state = State(self.bit_width, self.encode_type, 20, True, "default", True, "default", True, "default")
                state.ct = self.ct
                state.get_initial_pp_wiring()
                state.compressor_map = np.zeros([2, 20, state.get_pp_len()])
                state.compressor_map[0] = np.full([20, state.get_pp_len()], FA_type)
                state.compressor_map[1] = np.full([20, state.get_pp_len()], HA_type)

                rtl_path = os.path.join(build_base, "MUL.v")
                worker = EvaluateWorker(
                    rtl_path,
                    ["ppa", "power", "activity"],
                    target_delays[self.bit_width],
                    build_base,
                    n_processing=4
                )
                state.emit_verilog(rtl_path)
                worker.evaluate()
                save_dict[FA_type][HA_type]["default"] = worker.consult_ppa()

                state.pp_wiring_arrangement_v0(worker=worker)
                state.emit_verilog(rtl_path)
                worker.evaluate()
                save_dict[FA_type][HA_type]["routed"] = worker.consult_ppa()
        with open(os.path.join(self.db_base_dir, f"{self.key}-db.json"), "w") as file:
            json.dump(save_dict, file)


    def process_1(self):
        with open(os.path.join(self.db_base_dir, f"{self.key}-db.json"), "r") as file:
            db = json.load(file)
        
        plt.figure(figsize=[20, 10])
        for target_p_HA_index, target_p_HA in enumerate([[1, 1], [2, 1], [1, 2]]):
            target_p_HA = np.asarray(target_p_HA)
            target_p_HA = target_p_HA / np.sum(target_p_HA)
            self.flag_routed = False
            self.flag_default = False
            self.flag_routed_1 = False
            self.flag_default_1 = False
            for file_path in self.files:
                with open(file_path, "r") as file:
                    data = json.load(file)
                p_FA = data["params"]["P_FA"]
                p_HA = data["params"]["P_HA"]
                if np.linalg.norm(np.asarray(p_HA) - target_p_HA) > 1e-2:
                    print("skip")
                    continue
                
                default_power_list = []
                routed_power_list = []
                for item in data["result"]:
                    default_power_list.append(item["default"]["power"])
                    routed_power_list.append(item["routed"]["power"])
                    
                def __hist(type):
                    if type == "default":
                        if self.flag_default == True:
                            return
                    else:
                        if self.flag_routed == True:
                                return
                    for fa_type in range(3):
                        fa_type = str(fa_type)
                        for ha_type in range(2):
                            ha_type = str(ha_type)
                            if type == "default":
                                self.flag_default = True
                                default_power = db[fa_type][ha_type]["default"]["power"]
                                plt.plot([default_power]*2, [0, 20], label=f"FA: {fa_type}, HA: {ha_type}, {default_power:.2e}", alpha=0.6)
                            else:
                                self.flag_routed = True
                                routed_power = db[fa_type][ha_type]["routed"]["power"]
                                plt.plot([routed_power]*2, [0, 20], label=f"FA: {fa_type}, HA: {ha_type}, {routed_power:.2e}", alpha=0.6)
                    

                plt.subplot(2, 3, target_p_HA_index + 1)
                plt.title(f" p_HA = {np.round(p_HA, 2)}-default")
                __hist("default")
                plt.hist(default_power_list, density=False, alpha=0.5, label=f"{np.round(p_FA, 2)}")
                plt.legend()
                plt.tight_layout()

                plt.subplot(2, 3, target_p_HA_index + 1 + 3)
                plt.title(f" p_HA = {np.round(p_HA, 3)}-routed")
                __hist("routed")
                plt.hist(routed_power_list, density=False, alpha=0.5, label=f"{np.round(p_FA, 2)}")
                plt.legend()
                plt.tight_layout()


                with open(os.path.join(self.log_base_dir, "default", f"{self.key}.json"), "r") as file:
                    data = json.load(file)
                default_power_list = []
                routed_power_list = []
                for item in data["result"]:
                    default_power_list.append(item["default"]["power"])
                    routed_power_list.append(item["routed"]["power"])
                
                if self.flag_default_1 == False:
                    plt.subplot(2, 3, target_p_HA_index + 1)
                    plt.hist(default_power_list, density=False, alpha=0.5, label=f"FA+FA_LUT")
                    plt.legend()
                    plt.tight_layout()
                    self.flag_default_1 = True

                if self.flag_routed_1 == False:
                    plt.subplot(2, 3, target_p_HA_index + 1 + 3)
                    plt.hist(routed_power_list, density=False, alpha=0.5, label=f"FA+FA_LUT")
                    plt.legend()
                    plt.tight_layout()
                    self.flag_routed_1 = True

        save_path = f"./report/2024-11-13/{self.key}-different-FA.png"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.suptitle(self.key, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)

    def process_2(self):
        self.random_norouting_power_min = 1e6
        self.random_routed_power_min = 1e6
        self.default_norouting_power_min = 1e6
        self.default_routed_power_min = 1e6
        with open(os.path.join(self.db_base_dir, f"{self.key}-db.json"), "r") as file:
            db = json.load(file)
        
        plt.figure(figsize=[20, 10])
        for target_p_FA_index, target_p_FA in enumerate([[1, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3]]):
            target_p_FA = np.asarray(target_p_FA)
            target_p_FA = target_p_FA / np.sum(target_p_FA)
            self.flag_routed = False
            self.flag_default = False
            self.flag_default_1 = False
            self.flag_routed_1 = False
            for file_path in self.files:
                with open(file_path, "r") as file:
                    data = json.load(file)
                p_FA = data["params"]["P_FA"]
                p_HA = data["params"]["P_HA"]
                if np.linalg.norm(np.asarray(p_FA) - target_p_FA) > 1e-2:
                    print("skip")
                    continue
                
                default_power_list = []
                routed_power_list = []
                for item in data["result"]:
                    default_power_list.append(item["default"]["power"])
                    routed_power_list.append(item["routed"]["power"])
                    
                def __hist(type):
                    if type == "default":
                        if self.flag_default == True:
                            return
                    else:
                        if self.flag_routed == True:
                                return
                    for fa_type in range(3):
                        fa_type = str(fa_type)
                        for ha_type in range(2):
                            ha_type = str(ha_type)
                            if type == "default":
                                self.flag_default = True
                                default_power = db[fa_type][ha_type]["default"]["power"]
                                if fa_type == "0" and ha_type == "0":
                                    self.default_norouting_power_min = min([self.default_norouting_power_min, default_power])
                                plt.plot([default_power]*2, [0, 20], label=f"FA: {fa_type}, HA: {ha_type}, {default_power:.2e}", alpha=0.6)
                            else:
                                self.flag_routed = True
                                routed_power = db[fa_type][ha_type]["routed"]["power"]
                                if fa_type == "0" and ha_type == "0":
                                    self.default_routed_power_min = min([self.default_routed_power_min, routed_power])
                                plt.plot([routed_power]*2, [0, 20], label=f"FA: {fa_type}, HA: {ha_type}, {routed_power:.2e}", alpha=0.6)
                    

                plt.subplot(2, 4, target_p_FA_index + 1)
                plt.title(f" p_FA = {np.round(p_FA, 2)}-default")
                __hist("default")
                plt.hist(default_power_list, density=False, alpha=0.5, label=f"{np.round(p_HA, 2)}")
                self.random_norouting_power_min = min([self.random_norouting_power_min, np.min(default_power_list)])
                plt.legend()
                plt.tight_layout()

                plt.subplot(2, 4, target_p_FA_index + 1 + 4)
                plt.title(f" p_FA = {np.round(p_FA, 3)}-routed")
                __hist("routed")
                plt.hist(routed_power_list, density=False, alpha=0.5, label=f"{np.round(p_HA, 2)}")
                self.random_routed_power_min = min([self.random_routed_power_min, np.min(routed_power_list)])
                plt.legend()
                plt.tight_layout()

                with open(os.path.join(self.log_base_dir, "default", f"{self.key}.json"), "r") as file:
                    data = json.load(file)
                default_power_list = []
                routed_power_list = []
                for item in data["result"]:
                    default_power_list.append(item["default"]["power"])
                    routed_power_list.append(item["routed"]["power"])
                
                if self.flag_default_1 == False:
                    plt.subplot(2, 4, target_p_FA_index + 1)
                    plt.hist(default_power_list, density=False, alpha=0.5, label=f"FA+FA_LUT")
                    self.random_norouting_power_min = min([self.random_norouting_power_min, np.min(default_power_list)])
                    plt.legend()
                    plt.tight_layout()
                    self.flag_default_1 = True

                if self.flag_routed_1 == False:
                    plt.subplot(2, 4, target_p_FA_index + 1 + 4)
                    self.random_routed_power_min = min([self.random_routed_power_min, np.min(routed_power_list)])
                    plt.hist(routed_power_list, density=False, alpha=0.5, label=f"FA+FA_LUT")
                    plt.legend()
                    plt.tight_layout()
                    self.flag_routed_1 = True
                

        save_path = f"./report/2024-11-13/{self.key}-different-HA.png"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.suptitle(self.key, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)

        return {
            "label": self.key,
            "random - no routing": self.random_norouting_power_min,
            "default - no routing": self.default_norouting_power_min,
            "impr - no routing":f"{(self.default_norouting_power_min - self.random_norouting_power_min)/self.default_norouting_power_min * 100:.2f}",
            "random - routed": self.random_routed_power_min,
            "default - routed": self.default_routed_power_min,
            "impr - routed":f"{(self.default_routed_power_min - self.random_routed_power_min)/self.default_routed_power_min * 100:.2f}",
        }

class DataProcessor_1(DataProcessor):
    def process_1(self):
        self.random_norouting_power_min = 1e6
        self.random_routed_power_min = 1e6
        self.default_norouting_power_min = 1e6
        self.default_routed_power_min = 1e6

        print("111", os.path.join(self.db_base_dir, f"{self.key}-db.json"))
        with open(os.path.join(self.db_base_dir, f"{self.key}-db.json"), "r") as file:
            db = json.load(file)
        
        plt.figure(figsize=[12, 10])
        plt.cla()
        print("DataProcessor_1", self.files)
        for file in self.files:
            with open(file, "r") as file:
                data = json.load(file)
            p_FA = data["params"]["P_FA"]
            no_routing_power = []
            routed_power = []
            for item in data["result"]:
                no_routing_power.append(item["default"]["power"])
                routed_power.append(item["routed"]["power"])
            
            plt.subplot(2, 1, 1)
            plt.hist(no_routing_power, bins=15, label=f"p_FA = {np.round(p_FA, 3)}", alpha=0.7)
            self.random_norouting_power_min = min(self.random_norouting_power_min, np.min(no_routing_power))

            plt.subplot(2, 1, 2)
            plt.hist(routed_power, bins=15, label=f"p_FA = {np.round(p_FA, 3)}", alpha=0.7)
            self.random_routed_power_min = min(self.random_routed_power_min, np.min(routed_power))
        
        with open(os.path.join(self.db_base_dir, "default", f"{self.key}.json")) as file:
            data = json.load(file)
            p_FA = data["params"]["P_FA"]
            no_routing_power = []
            routed_power = []
            for item in data["result"]:
                no_routing_power.append(item["default"]["power"])
                routed_power.append(item["routed"]["power"])
            plt.subplot(2, 1, 1)
            plt.hist(no_routing_power, bins=15, label=f"p_FA = {np.round(p_FA, 3)}", alpha=0.7)

            plt.subplot(2, 1, 2)
            plt.hist(routed_power, bins=15, label=f"p_FA = {np.round(p_FA, 3)}", alpha=0.7)
            

        

        for fa in range(2):
            for ha in range(1):
                if not (fa == 0 and ha == 0):
                    print(66666666666)
                    self.random_routed_power_min = min(self.random_routed_power_min, db[f"{fa}"][f"{ha}"]["routed"]["power"])
                    self.random_norouting_power_min = min(self.random_norouting_power_min, db[f"{fa}"][f"{ha}"]["default"]["power"])

                    no_routing_default = db[f"{fa}"][f"{ha}"]["default"]["power"]
                    routed_default = db[f"{fa}"][f"{ha}"]["routed"]["power"]

                    plt.subplot(2, 1, 1)
                    plt.plot([no_routing_default]*2, [0, 20], "--o", label=f"fa:{fa}, ha:{ha}")
                    plt.legend()

                    plt.subplot(2, 1, 2)
                    plt.plot([routed_default]*2, [0, 20], "--o", label=f"fa:{fa}, ha:{ha}")
                    plt.legend()

        no_routing_default = db["0"]["0"]["default"]["power"]
        routed_default = db["0"]["0"]["routed"]["power"]
        plt.subplot(2, 1, 1)
        plt.plot([no_routing_default]*2, [0, 20], "--o", label="no_routing_default")
        self.default_norouting_power_min = no_routing_default
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot([routed_default]*2, [0, 20], "--o", label="routed_default")
        self.default_routed_power_min = routed_default
        plt.legend()


        save_path = f"./report/2024-11-13/{self.key}-different-FA.png"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.suptitle(self.key, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path)

        return {
            "label": self.key,
            "random - no routing": self.random_norouting_power_min,
            "default - no routing": self.default_norouting_power_min,
            "impr - no routing":f"{(self.default_norouting_power_min - self.random_norouting_power_min)/self.default_norouting_power_min * 100:.2f}",
            "random - routed": self.random_routed_power_min,
            "default - routed": self.default_routed_power_min,
            "impr - routed":f"{(self.default_routed_power_min - self.random_routed_power_min)/self.default_routed_power_min * 100:.2f}",
        }

# if __name__ == "__main__":
#     params = [
#         ["dadda", 16, "and", "log/random_refine"],
#         ["dadda", 16, "booth", "log/random_refine"],
#         ["loaded", 16, "and", "log/random_refine"],
#         ["loaded", 16, "booth", "log/random_refine"],
#         ["dadda", 32, "and", "log/random_refine"],
#         ["dadda", 32, "booth", "log/random_refine"],
#     ]
#     save_list = []
#     for i in params:
#         print(i)
#         processor = DataProcessor(*i)
#         # processor.get_default_value()
#         # processor.process()
#         save_list.append(processor.process_2())
#         # processor.process_1()
#     df = pandas.DataFrame.from_dict(save_list)
#     df.to_csv("report/2024-11-13/random和default对比.csv")


if __name__ == "__main__":
    params = [
        ["dadda", 16, "and", "log/random_refine/LUT_COMB_only", "log/random_refine"],
        ["dadda", 16, "booth", "log/random_refine/LUT_COMB_only", "log/random_refine"],
        ["loaded", 16, "and", "log/random_refine/LUT_COMB_only", "log/random_refine"],
        ["loaded", 16, "booth", "log/random_refine/LUT_COMB_only", "log/random_refine"],
        ["dadda", 32, "and", "log/random_refine/LUT_COMB_only", "log/random_refine"],
        ["dadda", 32, "booth", "log/random_refine/LUT_COMB_only", "log/random_refine"],
    ]
    save_list = []
    for i in params:
        print(i)
        processor = DataProcessor_1(*i)
        # processor.get_default_value()
        # processor.process()
        save_list.append(processor.process_1())
        # processor.process_1()
    df = pandas.DataFrame.from_dict(save_list)
    df.to_csv("report/2024-11-13/random和default对比-all.csv")
