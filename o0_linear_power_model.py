import logging.config
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
import json
import datetime
import dateutil.tz
import queue
import multiprocessing

# import gurobipy as gp
# from gurobipy import GRB

from o0_mul_utils import (
    legalize_compressor_tree,
    decompose_compressor_tree,
    get_initial_partial_product,
    get_default_pp_wiring,
    get_target_delay,
)
from o0_state import State
from o0_rtl_tasks import EvaluateWorker


def get_power(
    # 全加器半加器
    f: np.ndarray,
    h: np.ndarray,
    initial_pp: np.ndarray,
    pp_wiring: np.ndarray,
    # P_f
    beta_a,
    beta_b,
    beta_c,
    beta,
    # P_h
    alpha_a,
    alpha_b,
    alpha,
    # T_f
    nu_s_a,
    nu_s_b,
    nu_s_c,
    nu_c_a,
    nu_c_b,
    nu_c_c,
    # T_h
    mu_s_a,
    mu_s_b,
    mu_c_a,
    mu_c_b,
):
    P = 0.0
    S, N = f.shape

    # pp
    pp = np.zeros([S, N])
    pp[0] = initial_pp
    for i in range(S - 1):
        pp[i + 1, 0] = pp[i, 0] - 2 * f[i, 0] - h[i, 0]
        for j in range(1, N):
            pp[i + 1, j] = pp[i, j] - 2 * f[i, j] - h[i, j] + f[i, j - 1] + h[i, j - 1]
    T = np.zeros([S, N, int(np.max(pp))]).tolist()
    tild_T = np.zeros([S, N, int(np.max(pp))]).tolist()

    # T
    for j in range(N):
        for k in range(int(pp[0, j])):
            T[0][j][k] = 1.0

    for i in range(S - 1):
        for j in range(N):
            for k in range(int(pp[i, j])):
                next_k = pp_wiring[i][j][k]
                # tild_T[i, j, next_k] = T[i, j, k]
                tild_T[i][j][next_k] = T[i][j][k]
            if j + 1 < N:
                kappa = pp[i, j + 1] - 2 * f[i, j + 1] - h[i, j + 1]
                kappa = int(kappa)
            else:
                kappa = 0
            # fs
            for k in range(int(f[i, j])):
                T[i + 1][j][k] = (
                    nu_s_a * tild_T[i][j][3 * k]
                    + nu_s_b * tild_T[i][j][3 * k + 1]
                    + nu_s_c * tild_T[i][j][3 * k + 2]
                )

            # fc
            if j + 1 < N:
                for k in range(int(f[i, j])):
                    T[i + 1][j + 1][k + kappa] = (
                        nu_c_a * tild_T[i][j][3 * k]
                        + nu_c_b * tild_T[i][j][3 * k + 1]
                        + nu_c_c * tild_T[i][j][3 * k + 2]
                    )

            # hs
            for k in range(int(h[i, j])):
                T[i + 1][j][k + int(f[i, j])] = (
                    mu_s_a * tild_T[i][j][3 * int(f[i, j]) + 2 * k]
                    + mu_s_b * tild_T[i][j][3 * int(f[i, j]) + 2 * k + 1]
                )

            # hc
            if j + 1 < N:
                for k in range(int(h[i, j])):
                    T[i + 1][j + 1][k + int(f[i, j]) + kappa] = (
                        mu_c_a * tild_T[i][j][3 * int(f[i, j]) + 2 * k]
                        + mu_c_b * tild_T[i][j][3 * int(f[i, j]) + 2 * k + 1]
                    )

            # remain
            for k in range(int(pp[i, j]) - 3 * int(f[i, j]) - 2 * int(h[i, j])):
                T[i + 1][j][k + int(f[i, j]) + int(h[i, j])] = tild_T[i][j][
                    3 * int(f[i, j]) + 2 * int(h[i, j]) + k
                ]
    P = 0.0
    for i in range(S - 1):
        for j in range(N):
            P_f = 0.0
            P_h = 0.0
            for k in range(int(f[i, j])):
                P_f += (
                    beta
                    + beta_a * tild_T[i][j][3 * k]
                    + beta_b * tild_T[i][j][3 * k + 1]
                    + beta_c * tild_T[i][j][3 * k + 2]
                )
            for k in range(int(h[i, j])):
                P_h += (
                    alpha
                    + alpha_a * tild_T[i][j][3 * int(f[i, j]) + 2 * k]
                    + alpha_b * tild_T[i][j][3 * int(f[i, j]) + 2 * k + 1]
                )
            P += P_f + P_h

    return P


def least_square(
    bit_width=8,
    encode_type="and",
    data_path="report/random_data/data-8bits_and-2025_01_15_13_09_01.json",
):
    from sympy import symbols, pprint, simplify, expand

    beta_a, beta_b, beta_c, beta = symbols("beta[a], beta[b], beta[c], beta")
    alpha_a, alpha_b, alpha = symbols("alpha[a], alpha[b], alpha")
    nu_s_a, nu_s_b, nu_s_c = symbols("nu_s[a], nu_s[b], nu_s[c]")
    nu_c_a, nu_c_b, nu_c_c = symbols("nu_c[a], nu_c[b], nu_c[c]")
    mu_s_a, mu_s_b = symbols("mu_s[a], mu_s[b]")
    mu_c_a, mu_c_b = symbols("mu_c[a], mu_c[b]")

    with open(data_path) as file:
        data = json.load(file)
    pp = get_initial_partial_product(bit_width, encode_type)
    # for item in data:
    for item in data[:1]:
        ct = item["ct"]
        pp_wiring = item["pp_wiring"]
        f, h, _, __ = decompose_compressor_tree(pp, ct[0], ct[1])
        p = get_power(
            f.astype(int),
            h.astype(int),
            pp,
            pp_wiring,
            # P_f
            beta_a,
            beta_b,
            beta_c,
            beta,
            # P_h
            alpha_a,
            alpha_b,
            alpha,
            # T_f
            nu_s_a,
            nu_s_b,
            nu_s_c,
            nu_c_a,
            nu_c_b,
            nu_c_c,
            # T_h
            mu_s_a,
            mu_s_b,
            mu_c_a,
            mu_c_b,
        )
        # print(p)
        pprint(expand(p))
    # print(data[0])
    return 0


def gradient_decent(
    bit_width=16,
    encode_type="and",
    data_path="report/random_data/data-16bits_and-2025_01_15_13_13_09.json",
    batch_size=64,
    max_iter=5000,
    log_base="report/power_model",
    power_scale=1e3,
    log_freq=50,
    lr=1e-2,
    device="cuda:0",
    seed=0,
):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import random

    if not os.path.exists(log_base):
        os.makedirs(log_base)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    log_path = os.path.join(log_base, timestamp, "tb_event")
    tb__logger = SummaryWriter(log_path)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    class PowerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.beta_a = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.beta_b = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.beta_c = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.beta = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            # P_h
            self.alpha_a = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.alpha_b = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.alpha = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            # T_f
            self.nu_s_a = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.nu_s_b = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.nu_s_c = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.nu_c_a = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.nu_c_b = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.nu_c_c = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            # T_h
            self.mu_s_a = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.mu_s_b = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.mu_c_a = torch.nn.Parameter(torch.rand([1], requires_grad=True))
            self.mu_c_b = torch.nn.Parameter(torch.rand([1], requires_grad=True))

        def forward(self, f, h, pp, pp_wiring):
            p = get_power(
                f.astype(int),
                h.astype(int),
                pp,
                pp_wiring,
                # P_f
                self.beta_a,
                self.beta_b,
                self.beta_c,
                self.beta,
                # P_h
                self.alpha_a,
                self.alpha_b,
                self.alpha,
                # T_f
                self.nu_s_a,
                self.nu_s_b,
                self.nu_s_c,
                self.nu_c_a,
                self.nu_c_b,
                self.nu_c_c,
                # T_h
                self.mu_s_a,
                self.mu_s_b,
                self.mu_c_a,
                self.mu_c_b,
            )
            return p

    def log(power_model: PowerModel, step_index):
        sace_dict = {
            # P_f
            "beta_a": power_model.beta_a.detach().item(),
            "beta_b": power_model.beta_b.detach().item(),
            "beta_c": power_model.beta_c.detach().item(),
            "beta": power_model.beta.detach().item(),
            # P_h
            "alpha_a": power_model.alpha_a.detach().item(),
            "alpha_b": power_model.alpha_b.detach().item(),
            "alpha": power_model.alpha.detach().item(),
            # T_f
            "nu_s_a": power_model.nu_s_a.detach().item(),
            "nu_s_b": power_model.nu_s_b.detach().item(),
            "nu_s_c": power_model.nu_s_c.detach().item(),
            "nu_c_a": power_model.nu_c_a.detach().item(),
            "nu_c_b": power_model.nu_c_b.detach().item(),
            "nu_c_c": power_model.nu_c_c.detach().item(),
            # T_h
            "mu_s_a": power_model.mu_s_a.detach().item(),
            "mu_s_b": power_model.mu_s_b.detach().item(),
            "mu_c_a": power_model.mu_c_a.detach().item(),
            "mu_c_b": power_model.mu_c_b.detach().item(),
        }
        log_file_path = os.path.join(log_path, f"paramters-{bit_width}bits_{encode_type}-{step_index}step.json")
        with open(log_file_path, 'w') as file:
            json.dump(sace_dict, file)


    power_model = PowerModel().to(device)
    optimizer = torch.optim.Adam(power_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)

    with open(data_path) as file:
        data = json.load(file)
    pp = get_initial_partial_product(bit_width, encode_type)

    for step_index in range(max_iter):
        logging.info(f"step {step_index}")
        batch = random.sample(data, batch_size)
        loss = 0.0
        error_list = []

        optimizer.zero_grad()
        for item in batch:
            ct = item["ct"]
            pp_wiring = item["pp_wiring"]
            f, h, _, __ = decompose_compressor_tree(pp, ct[0], ct[1])
            p = power_model(f, h, pp, pp_wiring)
            true_p = power_scale * item["power"]
            loss += (p - true_p) ** 2
            error_list.append(abs(p.detach().item() - true_p) / true_p * 100)

        loss.backward()
        optimizer.step()
        scheduler.step()
        tb__logger.add_scalar("loss", loss.detach().item(), step_index)
        tb__logger.add_scalar("avg error (%)", np.mean(error_list), step_index)
        tb__logger.add_scalar("lr", optimizer.param_groups[0]["lr"], step_index)

        if step_index % log_freq == 0:
            log(power_model, step_index)
    log(power_model, step_index)
    return 0


def target_func(
    x,
    # P_f
    beta_a,
    beta_b,
    beta_c,
    beta,
    # P_h
    alpha_a,
    alpha_b,
    alpha,
    # T_f
    nu_s_a,
    nu_s_b,
    nu_s_c,
    nu_c_a,
    nu_c_b,
    nu_c_c,
    # T_h
    mu_s_a,
    mu_s_b,
    mu_c_a,
    mu_c_b,
):
    f, h, initial_pp, pp_wiring = x
    return get_power(
        f,
        h,
        initial_pp,
        pp_wiring,
        # P_f
        beta_a,
        beta_b,
        beta_c,
        beta,
        # P_h
        alpha_a,
        alpha_b,
        alpha,
        # T_f
        nu_s_a,
        nu_s_b,
        nu_s_c,
        nu_c_a,
        nu_c_b,
        nu_c_c,
        # T_h
        mu_s_a,
        mu_s_b,
        mu_c_a,
        mu_c_b,
    )


def compress_pp_wiring(pp_wiring: np.ndarray):
    compressed_wiring = []
    for i in range(len(pp_wiring)):
        if (pp_wiring[i] == -1).all():
            break
        compressed_wiring.append([])
        for j in range(len(pp_wiring[i])):
            compressed_wiring[i].append([])
            k = 0
            while k < len(pp_wiring[i][j]) and pp_wiring[i][j][k] != -1:
                compressed_wiring[i][j].append(int(pp_wiring[i][j][k]))
                k += 1
    return compressed_wiring


def get_data_worker(bit_width, encode_type, ct, pp_wiring, build_base, index):
    state = State(
        bit_width, encode_type, 2 * bit_width, True, None, True, None, False, None
    )
    state.ct = ct
    state.get_initial_compressor_map()
    state.pp_wiring = pp_wiring

    work_base = os.path.join(build_base, f"worker-{index}")
    rtl_path = os.path.join(work_base, "MUL.v")
    worker = EvaluateWorker(
        rtl_path,
        ["ppa"],
        get_target_delay(bit_width),
        work_base,
        False,
        False,
        False,
        False,
        n_processing=1,
    )

    state.emit_verilog(rtl_path)
    worker.evaluate()
    ppa_list = worker.consult_ppa_list()

    info = {
        "bit_width": bit_width,
        "encode_type": encode_type,
        "ct": np.asarray(ct).tolist(),
        # "pp_wiring": np.asarray(pp_wiring).tolist(),
        "pp_wiring": compress_pp_wiring(pp_wiring),
        "index": index,
        "ppa_list": ppa_list,
        "power_list": ppa_list["power"],
        "power": np.mean(ppa_list["power"]),
    }

    return info


def slice_routing_array_to_matrix(slice_routing):
    routing_matrix = np.zeros(shape=[len(slice_routing), len(slice_routing)])
    for i in range(len(slice_routing)):
        j = int(slice_routing[i])
        routing_matrix[i, j] = 1


def get_data(
    bit_width,
    encode_type,
    num=200,
    build_base="pybuild/random_data",
    report_base="report/random_data",
    n_processing=8,
    seed=0,
):
    np.random.seed(seed)
    key = f"{bit_width}bits_{encode_type}"
    build_path = os.path.join(build_base, key)

    pp = get_initial_partial_product(bit_width, encode_type)
    param_list = []
    for i in range(num):
        ct = np.random.randint(0, bit_width, size=[2, len(pp)])
        ct = np.asarray(legalize_compressor_tree(pp, ct[0], ct[1])).tolist()
        pp_wiring = get_default_pp_wiring(len(pp), pp, ct, "random")
        param = (
            bit_width,
            encode_type,
            ct,
            pp_wiring,
            build_path,
            i,
        )
        param_list.append(param)

    with multiprocessing.Pool(n_processing) as pool:
        result = pool.starmap_async(get_data_worker, param_list)
        pool.close()
        pool.join()
    result = result.get()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(report_base):
        os.makedirs(report_base)
    report_path = os.path.join(report_base, f"data-{key}-{timestamp}.json")
    with open(report_path, "w") as file:
        json.dump(result, file)


# if __name__ == "__main__":
def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
    )
    # get_data(8, "and", 2000)
    # nohup python ./o0_linear_power_model.py > o0_linear_power_model-8-and.out 2>&1 &

    # get_data(8, "booth", 2000)
    # nohup python ./o0_linear_power_model.py > o0_linear_power_model-8-booth.out 2>&1 &

    # get_data(16, "and", 2000)
    # nohup python ./o0_linear_power_model.py > o0_linear_power_model-16-and.out 2>&1 &

    # get_data(16, "booth", 2000)
    # nohup python ./o0_linear_power_model.py > o0_linear_power_model-16-booth.out 2>&1 &

    get_data(32, "and", 2000)
    # nohup python ./o0_linear_power_model.py > o0_linear_power_model-32-and.out 2>&1 &

    # get_data(32, "booth", 2000)
    # nohup python ./o0_linear_power_model.py > o0_linear_power_model-32-booth.out 2>&1 &



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s",
    )
    gradient_decent(8, "booth", "report/random_data/data-8bits_booth-2025_01_15_13_12_18.json")

    # gradient_decent()
    # gradient_decent(16, "booth", "report/random_data/data-16bits_booth-2025_01_15_13_14_12.json")
    # gradient_decent(32, "and", "report/random_data/data-32bits_and-2025_01_16_02_43_32.json")
    # gradient_decent(32, "booth", "report/random_data/data-32bits_booth-2025_01_16_02_39_29.json")
