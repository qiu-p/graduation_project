
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
    OpenROAD Synthesis Config
"""
# lef lib path
lef = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary.lef'
lib = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary_typical.lib'

# EasyMacPath
# EasyMacPath = '/datasets/ai4multiplier/rl-mul-code/rl-16mul-code/easymac_backup'

# v1 easymac no booth any
# EasyMacPath = '/datasets/ai4multiplier/rl-mul-code/easymac_test/easymac_backup/target/scala-2.12/easymac-assembly-0.0.1.jar'
# v2 easymac booth any
EasyMacPath = '/datasets/ai4multiplier/rl-mul-code/easymac_latest/easymac/target/scala-2.12/easymac-assembly-0.0.1.jar'

BenchmarkPath = '/datasets/ai4multiplier/rl-mul-code/easymac_test/easymac_backup/benchmarks/16x16/ppa.txt'
EasyMacTarPath = '/datasets/ai4multiplier/rl-mul-code/rl-16mul-code/easymac.tar.gz'
# OpenRoadFlowPath
OpenRoadFlowPath = '/datasets/ai4multiplier/openroad_deb/OpenROAD-flow-scripts'

lef = '/home/xiaxilin/MiraLab/nips2024/ai4-multiplier-master/data/NangateOpenCellLibrary.lef'
lib = '/home/xiaxilin/MiraLab/ai4mul-power/ai4-multiplier-master/utils/NangateOpenCellLibrary_typical.lib'
OpenRoadFlowPath = '/home/xiaxilin/MiraLab/OpenROAD-flow-scripts'
OpenRoadPath = '/home/xiaxilin/MiraLab/OpenROAD'

# abc constraint 是设计约束吗？
def abc_constr_gen(ys_path):
    abc_constr_file = os.path.join(ys_path, 'abc_constr')
    with open(abc_constr_file, 'w') as f:
        f.write('set_driving_cell BUF_X1')
        f.write('\n')
        f.write('set_load 10.0 [all_outputs]')
        f.write('\n')

# abc synthesis scripts gen
def abc_syn_script_gen(ys_path, syn_script):
    syn_script_file = os.path.join(ys_path, 'syn_script')
    with open(syn_script_file, 'w') as f:
        f.write(f'{syn_script}')
        f.write('\n')
    
# sta 的运行脚本，OpenSTA timing analysis
def sta_scripts_gen(ys_path):
    sta_file = os.path.join(ys_path, 'openroad_sta.tcl')
    with open(sta_file, 'w') as f:
        f.write('read_lef ' + str(lef))
        f.write('\n')
        f.write('read_lib ' + str(lib))
        f.write('\n')
        f.write('read_verilog ./netlist.v')
        f.write('\n')
        f.write('link_design MUL')
        # xilin-modify-begin
        f.write('\n')
        f.write("set period 5")
        f.write('\n')
        f.write("create_clock -period $period [get_ports clock]")
        f.write('\n')
        f.write("set clk_period_factor .2")
        f.write('\n')
        f.write("set clk [lindex [all_clocks] 0]")
        f.write('\n')
        f.write("set period [get_property $clk period]")
        f.write('\n')
        f.write("set delay [expr $period * $clk_period_factor]")
        f.write('\n')
        f.write("set_input_delay $delay -clock $clk [delete_from_list [all_inputs] [all_clocks]]")
        f.write('\n')
        f.write("set_output_delay $delay -clock $clk [delete_from_list [all_outputs] [all_clocks]]")
        f.write('\n')
        # xilin-modify-end
        f.write('\n')
        f.write('set_max_delay -from [all_inputs] 0')
        f.write('\n')
        f.write('set critical_path [lindex [find_timing_paths -sort_by_slack] 0]')
        f.write('\n')
        f.write('set path_delay [sta::format_time [[$critical_path path] arrival] 4]')
        f.write('\n')
        f.write('puts \"wns $path_delay\"')
        #f.write('report_wns')
        f.write('\n')
        f.write('report_design_area')
        f.write('\n')
        # xilin-modify-begin
        f.write('set_power_activity -input -activity 0.5\n')
        f.write('report_power\n')
        # xilin-modify-end
        f.write('exit')

# OpenRoad 脚本，ys 文件是啥？yosys, synthesis with target delay
def ys_scripts_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}/rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -liberty ' + str(lib))
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')

def ys_scripts_gen_v5(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}/rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -liberty ' + str(lib))
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')

# OpenRoad 脚本，ys 文件是啥？yosys, synthesis with target delay v2 copy from laiyao paper
def ys_scripts_v2_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}/rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('flatten') # add flatten
        f.write('\n')
        f.write('opt') # add opt
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -fast -liberty ' + str(lib)) # fast
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')

# OpenRoad 脚本，ys 文件是啥？yosys, synthesis with target delay v2 copy from laiyao paper
def ys_scripts_v3_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}/rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('flatten') # add flatten
        f.write('\n')
        f.write('opt') # add opt
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -liberty ' + str(lib)) # fast
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')

def ys_scripts_v5_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}/rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -fast -liberty ' + str(lib))
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')
        
def ys_scripts_v4_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}/rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('flatten') # add flatten
        f.write('\n')
        f.write('opt') # add opt
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -script ./syn_script -liberty ' + str(lib)) # fast
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')

# 读取 .v 文件，执行 openroad synthesis 脚本，然后再执行 openroad sta timing analysis
def get_ppa(ys_path):  
    synthesis_cmd = 'cd ' + ys_path + ' \n' + f'source {OpenRoadFlowPath}/env.sh\n' + 'yosys ./syn_with_target_delay.ys' # open road synthesis
    sta_cmd = 'cd ' + ys_path + ' \n' + f'source {OpenRoadFlowPath}/env.sh\n' + 'openroad openroad_sta.tcl | tee ./log' # openroad sta
    rm_log_cmd = 'rm -f ' + ys_path + '/log'
    rm_netlist_cmd = 'rm -f ' + ys_path + '/netlist.v'
    
    # execute synthesis cmd
    os.system(synthesis_cmd)  
    os.system(sta_cmd)

    # get ppa
    with open(ys_path + '/log', 'r') as f:
        rpt = f.read().splitlines()
        for line in rpt:
            if len(line.rstrip()) < 2:
                continue
            line = line.rstrip().split()
            if line[0] == 'wns':
                delay = line[-1]
                #delay = delay[1:]
                continue
            if line[0] == 'Design':
                area = line[2]
# xilin-modify-begin
                continue
            if line[0] == 'Total':
                power = line[-2]
                internal_power = line[1]
                switching_power = line[2]
                leakage_power = line[3]
                break
    ppa_dict = {
        "area": float(area),
        "delay": float(delay),
        "power": float(power),
        "internal_power": float(internal_power),
        "switching_power": float(switching_power),
        "leakage_power": float(leakage_power),
    }
# xilin-modify-end
    # remove log
    os.system(rm_log_cmd)
    os.system(rm_netlist_cmd)
    
    return ppa_dict

"""
    Replay Buffer
"""
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward','mask','next_state_mask','state_ct32','state_ct22','next_state_ct32','next_state_ct22','rewards_dict'))
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward','mask','next_state_mask','state_ct32','state_ct22','next_state_ct32','next_state_ct22'))
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','mask','next_state_mask'))

class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NStepReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)
        self.tmp_memory = []

    def push(self, is_last, *args):
        '''Save a transition'''
        self.tmp_memory.append(Transition(*args))
        if is_last:
            self.memory.append(self.tmp_memory) # save list of transitions; list of tuple; 
            self.tmp_memory = [] # clear tmp memory

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


MBRLTransition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','mask','next_state_mask','normalize_area','normalize_delay'))
class MBRLReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(MBRLTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

MBRLMultiObjTransition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','mask','next_state_mask','normalize_area','normalize_delay', 'area_reward', 'delay_reward'))
class MBRLMultiObjReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(MBRLMultiObjTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class MBRLTrajectoryReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)
        self.tmp_memory = []

    def push(self, is_last, *args):
        '''Save a transition'''
        self.tmp_memory.append(MBRLTransition(*args))
        if is_last:
            self.memory.append(self.tmp_memory) # save list of transitions; list of tuple; 
            self.tmp_memory = [] # clear tmp memory

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

MultiObjTransition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','mask','next_state_mask','area_reward', 'delay_reward'))
class MultiObjReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(MultiObjTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
PDMultiObjTransition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','mask','next_state_mask','area_reward', 'delay_reward', 'weight'))
class PDMultiObjReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(PDMultiObjTransition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
# SharedMultiObjTransition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward','mask','next_state_mask','area_reward', 'delay_reward'))

class SharedMultiObjTransition(object):
    # Define your transition class as needed
    def __init__(self, state, action, next_state, reward, mask, next_state_mask, area_reward, delay_reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.mask = mask
        self.next_state_mask = next_state_mask
        self.area_reward = area_reward
        self.delay_reward = delay_reward

class SharedMultiObjReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = mp.Manager().list(deque([], maxlen=capacity))

    def push(self, *args):
        '''Save a transition'''
        # self.memory.append(SharedMultiObjTransition(*args))
        try:
            # Clone CUDA tensors to CPU tensors before storing
            # args = tuple(x.cpu() if torch.is_tensor(x) and x.is_cuda else x for x in args)
            self.memory.append(SharedMultiObjTransition(*args)) # numpy array
        except Exception as e:
            print(f"Error in push: {e}")
            raise
    def sample(self, batch_size):
        memory_list = list(self.memory)
        return random.sample(memory_list, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class SharedMultiObjVectorConditionTransition(object):
    # Define your transition class as needed
    def __init__(self, state, action, next_state, reward, mask, next_state_mask, area_reward, delay_reward, weight_vector, delay_condition):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.mask = mask
        self.next_state_mask = next_state_mask
        self.area_reward = area_reward
        self.delay_reward = delay_reward
        self.weight_vector = weight_vector
        self.delay_condition = delay_condition

class SharedMultiObjVectorConditionReplayMemory(object):
    def __init__(self, capacity=10000):
        self.memory = mp.Manager().list(deque([], maxlen=capacity))

    def push(self, *args):
        '''Save a transition'''
        # self.memory.append(SharedMultiObjTransition(*args))
        try:
            # Clone CUDA tensors to CPU tensors before storing
            # args = tuple(x.cpu() if torch.is_tensor(x) and x.is_cuda else x for x in args)
            self.memory.append(SharedMultiObjVectorConditionTransition(*args)) # numpy array
        except Exception as e:
            print(f"Error in push: {e}")
            raise
    def sample(self, batch_size):
        memory_list = list(self.memory)
        return random.sample(memory_list, batch_size)
    
    def __len__(self):
        return len(self.memory)

"""
    Set Seed
"""
############## set global seeds ##############
def set_global_seed(seed=None):
    if seed is None:
        seed = int(time.time()) % 4096
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set CuDNN to be deterministic. Notice that this may slow down the training.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
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
        base_log_dir = _LOCAL_LOG_DIR

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
        logger._add_output(tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode='a')
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
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
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
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


if __name__ == "__main__":
    replay_buffer = SharedMultiObjReplayMemory()
    for _ in range(100):
        replay_buffer.push(
            np.array([[1,2]]),2,3,4,5,6,7,8
        )
    transitions = replay_buffer.sample(64)
    batch = {
        "state": [],
        "action": [],
        "next_state": [],
        "reward": [],
        "mask": [],
        "next_state_mask": [],
        "area_reward": [],
        "delay_reward": []
    }
    for transition in transitions:
        batch["state"].append(transition.state)
    state_batch = np.concatenate(batch["state"])
    state_batch = torch.tensor(state_batch)
    print(state_batch.shape)
    # batch = SharedMultiObjTransition(*zip(*transitions))
    # print(batch)