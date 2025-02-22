import numpy as np  
import collections
import string
import pickle
from scipy.stats import pearsonr

def write_list(list, prefix=''):
    str_value = ''
    width = 7
    for item in list:
        str_value += ' '
        str_value += str(item).ljust(width)
    str_add = '{}len: {:0>2d}\n{}data: {}\n'.format(prefix, len(list), prefix, str_value)
    return str_add

def write_deque(deque, prefix):
    str_add = '{}len: {:0>2d} \t'.format(prefix, len(deque))
    str_add += 'value class: {}\n'.format(str(deque[0].__class__))
    for i, v in enumerate(deque):
        str_add += process_value(i, 'data[{}]'.format(i), v, prefix)    
    return str_add

def write_ndarray(ndarray, prefix):
    str_value = np.array2string(ndarray)
    str_add = '{}shape: {}\n{}data: {}\n'.format(prefix, ndarray.shape, prefix, str_value)
    return str_add

def process_value(i: int, key, value, prefix='\t', i_tmpl:str=None):
    if i_tmpl == None:
        str_i = str(i).rjust(2, '0')
    else:
        str_i = i_tmpl.format(i)
    str_add = '{}{} {} \t{}\n'.format(prefix, str_i, str(key), str(value.__class__))

    if isinstance(value, np.float64):
        value = value.astype(float)
    
    prefix_value = prefix + '\t'
    if isinstance(value, float) or isinstance(value, int):  # float
        str_add += '{}{}\n'.format(prefix_value, value)
    elif isinstance(value, list):                           # list
        str_add += write_list(value, prefix_value)
    elif isinstance(value, collections.deque):              # deque
        str_add += write_deque(value, prefix_value)
    elif isinstance(value, dict):                           # dict
        for i1, (key1, value1) in enumerate(value.items()):
            str_add += process_value(i1, key1, value1, prefix_value)
    elif isinstance(value, np.ndarray):                     # np.ndarray
        str_add += write_ndarray(value, prefix_value)
    
    return str_add

def get_data_structure(save_path, data_dic):
    str_to_write = ''
    for i, (key, value) in enumerate(data_dic.items()):
        str_to_write += process_value(i, key, value, prefix='')

    with open(save_path, 'w') as f:
            f.write(str_to_write)

# no power mask
input_path1 = './outputs/2024-12-16/15-42-21/logger_log/dqn_8bits_factor_action/refine_debug/refine_debug_2024_12_16_15_42_41_0000--s-2525/itr_5000.npy'
output_path1 = 'ysh_dataread1.txt'
# with power mask
input_path2 = './outputs/2024-12-16/16-15-46/logger_log/dqn_8bits_factor_action/refine_debug/refine_debug_2024_12_16_16_16_07_0000--s-434/itr_5000.npy'
output_path2 = 'ysh_dataread2.txt'
input_path = './outputs/2024-12-15/23-10-53/logger_log/dqn_8bits_factor_action/refine_debug/refine_debug_2024_12_15_23_11_01_0000--s-381/itr_4400.npy'

data = np.load(input_path2, allow_pickle=True)
    
data_dic = data.item()
print('data class: ', data.__class__)
print('data_dic class: ', data_dic.__class__)

get_data_structure(output_path2, data_dic)