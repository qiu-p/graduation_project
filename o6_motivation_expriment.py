"""
motivation实验专用文件
之前的太多太杂了，单独放一个文件统一管理
"""

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

"""
在这里设置 logging 等级
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
)


class Expr_0:
    def __init__(
        self,
        random_num,
        ct_type,
        max_stage_num,
        state_npy_path,
        bit_width,
        target_delay_list,
        encode_type,
        p_FA,
        p_HA,
        #
        build_base_dir,
        log_base_dir,
        n_processing,
        index,
    ) -> None:
        self.ct_type = ct_type
        self.random_num = random_num
        self.p_FA: np.ndarray = np.asarray(p_FA) / np.sum(p_FA)
        self.p_HA = np.asarray(p_HA) / np.sum(p_HA)
        self.max_stage_num = max_stage_num
        self.encode_type = encode_type
        self.bit_width = bit_width
        self.index = index

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        self.build_base_dir = os.path.join(build_base_dir, formatted_time+f"_{self.bit_width}_{self.encode_type}_{self.ct_type}")
        self.log_base_dir = log_base_dir
        self.target_delay_list = target_delay_list

        if not os.path.exists(build_base_dir):
            os.makedirs(build_base_dir)

        if not os.path.exists(log_base_dir):
            os.makedirs(log_base_dir)

        self.n_processing = n_processing

        if ct_type in ["wallace", "dadda"]:
            pp = get_initial_partial_product(bit_width, encode_type)
            self.ct = get_compressor_tree(pp, bit_width, ct_type)
            self.pp_wiring = None
        else:
            data = np.load(state_npy_path, allow_pickle=True).item()
            state_data: State = data["state"]
            self.ct = state_data.ct
            self.pp_wiring = state_data.pp_wiring

    @staticmethod
    def wrapped_run(
        bit_width,
        ct,
        pp_wiring,
        encode_type,
        p_FA,
        p_HA,
        max_stage_num,
        build_base_dir,
        target_delay_list,
        seed,
        id,
    ):
        np.random.seed(seed)
        info = {}
        state = State(
            bit_width,
            encode_type,
            max_stage_num,
            True,
            "default",
            True,
            "default",
            True,
            "default",
        )
        build_dir = os.path.join(build_base_dir, f"sample_{id}")
        rtl_path = os.path.join(build_dir, "MUL.v")
        worker = EvaluateWorker(
            rtl_path,
            ["ppa", "power", "activity"],
            target_delay_list,
            build_dir,
            False,
            False,
            False,
            False,
            n_processing=1,
        )
        compressor_map = np.zeros([2, max_stage_num, state.get_pp_len()])
        for stage_index in range(max_stage_num):
            for column_index in range(state.get_pp_len()):
                compressor_map[0][stage_index][column_index] = np.random.choice(
                    [0, 1, 2], p=p_FA
                )
                compressor_map[1][stage_index][column_index] = np.random.choice(
                    [0, 1], p=p_HA
                )
        state.ct = ct
        state.compressor_map = copy.deepcopy(compressor_map)
        if pp_wiring is not None:
            state.pp_wiring = pp_wiring
            state.emit_verilog(rtl_path)
            worker.evaluate()
            info["loaded"] = worker.consult_ppa()

        state.get_initial_pp_wiring()
        state.emit_verilog(rtl_path)
        worker.evaluate()
        info["default"] = worker.consult_ppa()

        state.pp_wiring_arrangement_v0(None, None, None, None, worker)
        state.emit_verilog(rtl_path)
        worker.evaluate()
        info["routed"] = worker.consult_ppa()

        return info

    def run(self):
        parameters_list = [
            (
                self.bit_width,
                self.ct,
                self.pp_wiring,
                self.encode_type,
                self.p_FA,
                self.p_HA,
                self.max_stage_num,
                self.build_base_dir,
                self.target_delay_list,
                np.random.randint(0, 1000 * self.random_num),
                i,
            )
            for i in range(self.random_num)
        ]
        with multiprocessing.Pool(self.n_processing) as pool:
            result = pool.starmap_async(self.wrapped_run, parameters_list)
            pool.close()
            pool.join()

            result = result.get()
        with open(
            os.path.join(
                self.log_base_dir,
                f"{self.bit_width}_{self.encode_type}_{self.ct_type}_{self.index}.json",
            ),
            "w",
        ) as file:
            json.dump(
                {
                    "params": {
                        "P_FA": self.p_FA.tolist(),
                        "P_HA": self.p_HA.tolist(),
                    },
                    "result": result,
                },
                file,
            )


if __name__ == "__main__":
    # python ./o6_motivation_expriment.py --random_num 1 --ct_type dadda --max_stage_num 7 --state_npy_path "temp/state-16-and.npy" --bit_width 16 --target_delay_list 50 200 500 1200 --encode_type and --p_FA 3.0 1.0 1.0 --p_HA 1.0 1.0 --build_base_dir "pybuild/random/debug" --log_base_dir "log/random/debug" --n_processing 8
    # nohup python ./o6_motivation_expriment.py --random_num 1 --ct_type dadda --max_stage_num 32 --state_npy_path "None" --bit_width 16 --target_delay_list 50 200 500 1200  --encode_type and --p_FA 1.0 1.0 1.0 --p_HA 1.0 1.0 --build_base_dir "pybuild/random/debug" --log_base_dir "log/random/debug" --n_processing 1 --index 0 > cmd_outs/random_0.out 2>&1 &

    parser = argparse.ArgumentParser(
        description="Initialize Expr_0 class with command line arguments."
    )

    # 添加命令行参数
    parser.add_argument("--random_num", type=int, required=True, help="Random number.")
    parser.add_argument(
        "--ct_type", type=str, required=True, help="Compressor tree type."
    )
    parser.add_argument(
        "--max_stage_num", type=int, required=True, help="Maximum stage number."
    )
    parser.add_argument(
        "--state_npy_path", type=str, required=True, help="Path to state numpy file."
    )
    parser.add_argument("--bit_width", type=int, required=True, help="Bit width.")
    parser.add_argument(
        "--target_delay_list",
        type=int,
        nargs="+",
        required=True,
        help="List of target delays.",
    )
    parser.add_argument("--encode_type", type=str, required=True, help="Encode type.")
    parser.add_argument(
        "--p_FA", type=float, nargs="+", required=True, help="Probabilities for FA."
    )
    parser.add_argument(
        "--p_HA", type=float, nargs="+", required=True, help="Probabilities for HA."
    )
    parser.add_argument(
        "--build_base_dir", type=str, required=True, help="Build base directory."
    )
    parser.add_argument(
        "--log_base_dir", type=str, required=True, help="Log base directory."
    )
    parser.add_argument(
        "--n_processing", type=int, required=True, help="Number of processing threads."
    )
    parser.add_argument("--index", type=int, required=True, help="index")

    args = parser.parse_args()

    # 处理概率参数 p_FA 和 p_HA
    p_HA = np.asarray(args.p_HA)
    p_HA /= np.sum(p_HA)
    p_FA = np.asarray(args.p_FA)
    p_FA /= np.sum(p_FA)

    print(args.target_delay_list)
    # 实例化 Expr_0 类
    expr1 = Expr_0(
        args.random_num,
        args.ct_type,
        args.max_stage_num,
        args.state_npy_path,
        args.bit_width,
        args.target_delay_list,
        args.encode_type,
        p_FA,
        p_HA,
        args.build_base_dir,
        args.log_base_dir,
        args.n_processing,
        args.index,
    )

    expr1.run()
    print("done")
