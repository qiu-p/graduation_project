import json
import logging
import os

from o0_rtl_tasks import EvaluateWorker
from o0_state import State
import pandas
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
)

yosys_defalt = """
module MUL(a,b,clock,out);
    input clock;
	input[{}:0] a;
	input[{}:0] b;
	output[{}:0] out;
    assign out = a * b;
endmodule
"""



def get_ppa_full_delay(bit_width):
    target_delay = []
    if bit_width == 8:
        for i in range(50, 1000, 10):
            target_delay.append(i)
    elif bit_width == 16:
        for i in range(50, 2000, 10):
            target_delay.append(i)
    elif bit_width == 32:
        for i in range(50, 3000, 10):
            target_delay.append(i)
    elif bit_width == 64:
        for i in range(50, 4000, 10):
            target_delay.append(i)
            
    return target_delay

def get_data():

    for bit_width in [8]:
        save_data_dict = {}
        
        build_base = f"build/baseline-{bit_width}bit"
        report_path = "report/2024-12-05"
        if not os.path.exists(build_base):
            os.makedirs(build_base)
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        
        full_target_delay = get_ppa_full_delay(bit_width)

        # 先测试 yosys 默认
        work_path = os.path.join(build_base, "yosys-default")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        rtl_path = os.path.join(work_path, "MUL.v")
        with open(rtl_path, "w") as file:
            file.write(yosys_defalt.format(bit_width - 1, bit_width - 1, 2 * bit_width - 2))
        worker = EvaluateWorker(rtl_path, ["ppa"], full_target_delay, work_path, False, False, False, False, n_processing=32)
        worker.evaluate()
        save_data_dict["yosys"] = worker.consult_ppa_list()

        # 再测各种方法
        for encode_type in  ["and", "booth"]:
            for ct_type in ["wallace", "dadda"]:
                for method in ["default", "brent_kung", "sklansky", "kogge_stone", "han_carlson"]:
                    method_key = f"{encode_type}_{ct_type}_{method}"
                    state = State(bit_width, encode_type, 32, True, "default", True, "default", True, method)
                    state.init(ct_type)
                    work_path = os.path.join(build_base, method_key)
                    if not os.path.exists(work_path):
                        os.makedirs(work_path)
                    rtl_path = os.path.join(work_path, "MUL.v")
                    state.emit_verilog(rtl_path)
                    worker = EvaluateWorker(rtl_path, ["ppa"], full_target_delay, work_path, False, False, False, False, n_processing=32)
                    worker.evaluate()
                    save_data_dict[method_key] = worker.consult_ppa_list()

                    log_path = os.path.join(report_path, f"{bit_width}-baselines.json")
                    with open(log_path, "w") as file:
                        json.dump(save_data_dict, file)
get_data()

import matplotlib.pyplot as plt


def draw_pareto(bit_width, encode_type, ct_type):
    data_path = f"report/2024-12-06/{bit_width}-baselines.json"
    with open(data_path, "r") as file:
        data = json.load(file)

    plt.figure(figsize=[10, 10])
    for key in data.keys():
        if key != "yosys" and (encode_type not in key or ct_type not in key):
            continue
        power = data[key]["power"]
        area = data[key]["area"]
        delay = data[key]["delay"]

        plt.subplot(3, 1, 1)
        plt.plot(area, delay, "--o", label=key)
        plt.xlabel("area")
        plt.ylabel("delay")
        plt.title("area-delay")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(power, delay, "--o", label=key)
        plt.xlabel("power")
        plt.ylabel("delay")
        plt.title("power-delay")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(power, area, "--o", label=key)
        plt.xlabel("power")
        plt.ylabel("area")
        plt.title("power-area")
        plt.legend()
    plt.suptitle(f"{bit_width}-{encode_type}-{ct_type}")
    plt.tight_layout()
    plt.savefig(f"report/2024-12-06/{bit_width}-{encode_type}-{ct_type}.png")
    # plt.show()
    print(111)

def get_table(bit_width, encode_type):
    data_path = f"report/2024-12-06/{bit_width}-baselines.json"
    with open(data_path, "r") as file:
        data = json.load(file)
    save_list = []


    power = data["yosys"]["power"]
    area = data["yosys"]["area"]
    delay = data["yosys"]["delay"]
    save_dict = {}
    save_dict["method"] = "yosys"
    save_dict["power"] = min(power)
    save_dict["power impr %"]= 0

    save_dict["area"] = min(area)
    save_dict["area impr %"]= 0

    save_dict["delay"] = min(delay)
    save_dict["delay impr %"] = 0

    default_power = min(power)
    default_area = min(area)
    default_delay = min(delay)

    save_list.append(save_dict)
    
    for key in data.keys():
        save_dict = {}
        if key == "yosys" or (encode_type not in key):
            continue
        power = data[key]["power"]
        area = data[key]["area"]
        delay = data[key]["delay"]

        save_dict["method"] = key
        save_dict["power"] = min(power)
        save_dict["power impr %"] = (default_power - min(power)) / default_power * 100

        save_dict["area"] = min(area)
        save_dict["area impr %"] = (default_area - min(area)) / default_area * 100

        save_dict["delay"] = min(delay)
        save_dict["delay impr %"] = (default_delay - min(delay)) / default_delay * 100

        save_list.append(save_dict)
    print(111)
    return save_list

# for bit_width in [8, 16, 32]:
#     for encode_type in ["and", "booth"]:
#         for ct_type in ["wallace", "dadda"]:
#             draw_pareto(bit_width, encode_type, ct_type)


for bit_width in [8, 16, 32]:
    for encode_type in ["and", "booth"]:
        save_list = get_table(bit_width, encode_type)
        df = pandas.DataFrame.from_dict(save_list)
        df.to_csv(f"report/2024-12-06/{bit_width}-{encode_type}.csv")