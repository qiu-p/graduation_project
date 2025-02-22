import argparse
import copy
import json
import multiprocessing
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
# import statsmodels.api as sm
from omegaconf import DictConfig, OmegaConf
from scipy import stats
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
from tensorboard.backend.event_processing import event_accumulator


class DataProcessor:
    def __init__(
        self,
        ct_type,
        bit_width,
        encode_type,
        log_base_dir,
    ) -> None:

        self.ct_type = ct_type
        self.bit_width = bit_width
        self.encode_type = encode_type
        self.log_base_dir = log_base_dir

        key = f"expr2_{bit_width}_{encode_type}_{ct_type}"
        self.key = key
        # keys = [
        #     f"expr2_{bit_width}_{encode_type}_{ct_type}"
        #     for encode_type in ["and", "booth"]
        #     for ct_type in ["dadda", "loaded"]
        # ]

        self.filepath = [
            os.path.join(log_base_dir, file)
            for file in os.listdir(log_base_dir)
            if "expr2" in file and "json" in file and key in file
        ][0]
        print(self.filepath)

    # fmt: off
    def process(self):
        data_list = []
        draw_index = 1
        # plt.figure(figsize=[14, 16])

        with open(self.filepath, "r") as file:
            data = json.load(file)

        random_power_matrix = np.zeros([100, 100])
        routing_power_arr_default = np.zeros(100)
        routing_power_arr_routed = np.zeros(100)
        for item in data["result"]:
            if item["label"] == "random":
                compressor_map_index = item["info"]["compressor_map_index"]
                pp_wiring_index = item["info"]["pp_wiring_index"]
                random_power_matrix[compressor_map_index][pp_wiring_index] = item["data"]["power"]
            elif item["label"] == "routing":
                compressor_map_index = item["info"]["compressor_map_index"]
                routing_power_arr_default[compressor_map_index] = item["default"]["power"]
                routing_power_arr_routed[compressor_map_index] = item["routed"]["power"]
        
        mean = np.mean(random_power_matrix.reshape(-1))
        std_dev = np.std(random_power_matrix.reshape(-1))

        z_score = (np.mean(routing_power_arr_routed) - mean) / std_dev
        percentile = stats.norm.cdf(z_score)
        print(f"分位数 {percentile:.3}")

        plt.figure(figsize=[12, 5])
        plt.subplot(1, 2, 1)
        
        plt.imshow(random_power_matrix)
        plt.colorbar()
        plt.xlabel("comap_index")
        plt.ylabel("routing_index")
        # plt.title(f"percentile of mean of routed = {percentile}")
        plt.tight_layout()
        ####

        plt.subplot(1, 2, 2)
        def hist(arr: np.ndarray, bins, label):
            max_value = np.max(arr)
            min_value = np.min(arr)
            mean_value = np.mean(arr)
            ratio = np.round((max_value - min_value) / mean_value, 4)
            if bins is not None:
                plt.hist(arr.reshape(-1), bins=bins, alpha=0.5, density=True, label=f"{label}, (max-min)/mean={ratio}")
            else:
                plt.hist(arr.reshape(-1), alpha=0.5, density=True, label=f"{label}, (max-min)/mean={ratio}")


        # if self.encode_type == "booth":
        if False:
            hist(random_power_matrix, 20, "random")
            hist(routing_power_arr_default, 10, "default routing")
            hist(routing_power_arr_routed, 10, "routed")
        else:
            hist(random_power_matrix, 50, "random")
            hist(routing_power_arr_default, 20, "default routing")
            hist(routing_power_arr_routed, 20, "routed")
        
        plt.legend()
        plt.suptitle(f"{self.bit_width}_{self.encode_type}_{self.ct_type}", fontsize=20, fontweight="bold")
        plt.tight_layout()

        plt.savefig(os.path.join("report/2024-11-12/", f"{self.key}.png"))
        # plt.show()

        # 效用分析
        N = 100
        X_values = [f"label_x_{i}" for i in range(N)]
        Y_values = [f"label_y_{i}" for i in range(N)]
        X_grid, Y_grid = np.meshgrid(X_values, Y_values)
        # F_values = random_power_matrix[:N, :N].flatten()
        F_values = random_power_matrix.flatten()

        X_flat = X_grid.flatten()
        Y_flat = Y_grid.flatten()

        # 创建 DataFrame
        data = pd.DataFrame({
            'X': X_flat,
            'Y': Y_flat,
            'F': F_values
        })

        # 计算总体均值
        overall_mean = np.mean(random_power_matrix)

        # 计算compressor_map和pp_wiring的主效应及交互效应
        compressor_map_means = np.mean(random_power_matrix, axis=1)  # 对每行（compressor_map水平）求平均
        pp_wiring_means = np.mean(random_power_matrix, axis=0)       # 对每列（pp_wiring水平）求平均

        # 计算平方和
        SS_compressor_map = np.sum((compressor_map_means - overall_mean) ** 2) * random_power_matrix.shape[1]
        SS_pp_wiring = np.sum((pp_wiring_means - overall_mean) ** 2) * random_power_matrix.shape[0]
        SS_interaction = np.sum((random_power_matrix - compressor_map_means[:, None] - pp_wiring_means + overall_mean) ** 2)

        # 计算总平方和
        SS_total = np.sum((random_power_matrix - overall_mean) ** 2)

        # 输出结果
        print()
        print("SS_compressor_map (主效应):", SS_compressor_map)
        print("SS_pp_wiring (主效应):", SS_pp_wiring)
        print("SS_interaction (交互效应):", SS_interaction)
        print("SS_total (总平方和):", SS_total)

        # 计算各效应的相对贡献比例并转换为百分制
        contribution_compressor_map = (SS_compressor_map / SS_total) * 100
        contribution_pp_wiring = (SS_pp_wiring / SS_total) * 100
        contribution_interaction = (SS_interaction / SS_total) * 100

        print()
        print("compressor_map 对总变异的贡献: {:.2f}%".format(contribution_compressor_map))
        print("pp_wiring 对总变异的贡献: {:.2f}%".format(contribution_pp_wiring))
        print("交互效应对总变异的贡献: {:.2f}%".format(contribution_interaction))

        return {
            "label": self.key,
            "percentile": f"{percentile:.2e}",
            "comap contribution": f"{contribution_compressor_map:.2f}%",
            "routing contribution": f"{contribution_pp_wiring:.2f}%",
            "interaction contribution": f"{contribution_interaction:.2f}%",
        }

    def process_1(self):
        with open(self.filepath, "r") as file:
            data = json.load(file)

        random_power_matrix = np.zeros([100, 100])
        routing_power_arr_default = np.zeros(100)
        routing_power_arr_routed = np.zeros(100)
        for item in data["result"]:
            if item["label"] == "random":
                compressor_map_index = item["info"]["compressor_map_index"]
                pp_wiring_index = item["info"]["pp_wiring_index"]
                random_power_matrix[compressor_map_index][pp_wiring_index] = item["data"]["power"]
            elif item["label"] == "routing":
                compressor_map_index = item["info"]["compressor_map_index"]
                routing_power_arr_default[compressor_map_index] = item["default"]["power"]
                routing_power_arr_routed[compressor_map_index] = item["routed"]["power"]
        

# fmt: off
if __name__ == "__main__":
    params = [
        ["dadda", 16, "and", "log/random_refine"],
        ["dadda", 16, "booth", "log/random_refine"],
        ["loaded", 16, "and", "log/random_refine"],
        ["loaded", 16, "booth", "log/random_refine"],
        # ["dadda", 32, "and", "log/random_refine"],
        # ["dadda", 32, "booth", "log/random_refine"],
    ]
    result_list = []
    for i in params:
        print("\n=========")
        print(i)
        processor = DataProcessor(*i)
        result_list.append(processor.process())
        # processor.process_1()
    
    # df = pandas.DataFrame.from_dict(result_list)
    # df.to_csv("./report/2024-11-12/expr2.csv")
    print("done")
