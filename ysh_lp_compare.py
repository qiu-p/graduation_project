import numpy as np
import scipy.sparse as sp
import copy
import os
from ysh_tools.utils import make_dir

from o0_rtl_tasks import EvaluateWorker, PowerSlewConsulter
from o0_state import State, legal_FA_list, legal_HA_list
from o0_mul_utils import (
    decompose_compressor_tree,
    get_compressor_tree,
    get_initial_partial_product,
    legal_FA_list,
    legal_HA_list,
    legalize_compressor_tree,
    write_mul,
)

class Compare:
    def __init__(
        self,
        state: State,
        target_delay,
        output_base_dir: str,
        task_index: int = 0
        ):
        self.top_name = 'MUL'
        self.n_processing = 4
        
        self.evaluate_target = []
        use_routing_optimize = True
        if use_routing_optimize is True:
            self.evaluate_target += ["ppa", "activity", "power"]
        else:
            self.evaluate_target += ["ppa", "power"]
        self.target_delay = target_delay
        
        self.bit_width = state.bit_width
        self.pp_encode_type = state.encode_type
        self.cur_state = state

        self.output_dir = '{}/{}bits_{}_{}/'.format(output_base_dir, self.bit_width, self.pp_encode_type, task_index)
        self.wire_map_filename = self.output_dir + 'wire_map.txt'
        make_dir(self.output_dir, 'i')
        self.eval_build_path_1 = os.path.join(
            self.output_dir, f"{self.bit_width}bits_{self.pp_encode_type}_1"
        )
        self.rtl_path_1 = os.path.join(self.eval_build_path_1, "MUL.v")
        self.eval_build_path_2 = os.path.join(
            self.output_dir, f"{self.bit_width}bits_{self.pp_encode_type}_2"
        )
        self.rtl_path_2 = os.path.join(self.eval_build_path_2, "MUL.v")
    
    def compare(self):
        state_1 = copy.deepcopy(self.cur_state)
        state_1.get_initial_pp_wiring()
        state_1.emit_verilog(self.rtl_path_1)
        evaluate_worker_1 = EvaluateWorker(
            self.rtl_path_1,
            self.evaluate_target,
            self.target_delay,
            self.eval_build_path_1,
            n_processing=self.n_processing,
            top_name=self.top_name,
        )
        evaluate_worker_1.evaluate()
        state_1.update_power_mask(evaluate_worker_1)
        
        state_2 = copy.deepcopy(self.cur_state)
        state_2.set_pp_wiring(self.wire_map_filename)
        state_2.emit_verilog(self.rtl_path_2)
        evaluate_worker_2 = EvaluateWorker(
            self.rtl_path_2,
            self.evaluate_target,
            self.target_delay,
            self.eval_build_path_2,
            n_processing=self.n_processing,
            top_name=self.top_name,
        )
        evaluate_worker_2.evaluate()
        state_2.update_power_mask(evaluate_worker_2)
        print(evaluate_worker_1.consult_ppa())
        print(evaluate_worker_2.consult_ppa())

if __name__ == '__main__':
    task_index = 1
    
    bit_width = 32
    # pp_encode_type = 'and'
    pp_encode_type = 'booth'
    if task_index == 0:
        init_ct_type = 'wallace'
    elif task_index == 1:
        init_ct_type = 'dadda'
    MAX_STAGE_NUM = 2 * bit_width
    state = State(
        bit_width,
        pp_encode_type,
        MAX_STAGE_NUM,
        use_pp_wiring_optimize=True,
        pp_wiring_init_type='default',
        use_compressor_map_optimize=False,
        compressor_map_init_type='default',
        use_final_adder_optimize=False,
        final_adder_init_type='default',
        top_name='MUL',
    )
    state.init(init_ct_type)
    
    target_delay = [50,200,500,1200]
    output_base_dir = 'ysh_output'
    
    my = Compare(state, target_delay, output_base_dir, task_index)

    my.compare()