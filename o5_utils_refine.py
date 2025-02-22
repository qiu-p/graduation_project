import os
import random
import time
import datetime
import dateutil.tz
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import namedtuple, deque
from os.path import join
import os.path as osp

from o0_logger import logger


"""
    Replay Buffer
"""
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "mask", "next_state_mask")
)


class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


MaskTransition = namedtuple(
    "MaskTransition",
    (
        "state",
        "next_state",
        "action",
        "reward",
    ),
)


class MaskReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(MaskTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


MARLTransition = namedtuple(
    "MARLTransition",
    (
        "ct_image_state",
        "ct_mask",

        "next_ct_image_state",
        "next_ct_mask",
        
        "pt_image_state",
        "next_pt_image_state",
        
        "action_ct",
        "action_pt",
        
        "reward",
    ),
)


class MARLReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(MARLTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

PowerMaskTransition = namedtuple(
    "PowerMaskTransition", ("state", "action", "next_state", "reward", "mask", "next_state_mask", "next_state_power_mask")
)

class PowerMaskReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(PowerMaskTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""
    RunningMeanStd
"""


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def set_global_seed(seed=None):
    if seed is None:
        seed = int(time.time()) % 4096
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Set CuDNN to be deterministic. Notice that this may slow down the training.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True
    return seed


"""
    Set Logger
"""


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(
    exp_prefix,
    exp_id=0,
    seed=0,
    base_log_dir=None,
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log directories be under this directory.
    :param exp_id: The number of the specific experiment run within this experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id, seed)

    if base_log_dir is None:
        raise NotImplementedError

    log_dir = join(base_log_dir, exp_prefix, exp_name)

    if osp.exists(log_dir):
        logger.log("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def setup_logger(
    exp_prefix="default",
    variant=None,
    text_log_file="debug.log",
    variant_log_file="variant.json",
    tabular_log_file="progress.csv",
    snapshot_mode="last",
    snapshot_gap=1,
    log_tabular_only=False,
    log_dir=None,
    script_name=None,
    **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to

        base_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param variant: 实验参数字典
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param script_name: If set, save the script name to this.
    :return:
    """

    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        logger.log("Variant:")
        # logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = join(log_dir, tabular_log_file)
    text_log_path = join(log_dir, text_log_file)
    logger.add_text_output(text_log_path)

    if first_time:
        logger.add_tabular_output(tabular_log_path)

    else:
        logger._add_output(
            tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode="a"
        )
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if script_name is not None:
        with open(join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


"""
    RunningMeanStd
"""


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


if __name__ == "__main__":
    pass
