import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from ysh_tools.utils import make_dir

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


'''
yosys
'''

if __name__ == '__main__':
    task_index = 0 # 0: wallace  1: dadda

    bit_width = 8
    pp_encode_type = 'and'
    # pp_encode_type = 'booth'
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
    print('ct:', state.ct)
    print('ct_decomposed:', state.ct_decomposed)
    print('stage_num:', state.stage_num)
    print('initial_pp:', state.initial_pp)
    print('sequence_pp:', state.sequence_pp)