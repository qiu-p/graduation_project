import numpy as np
import torch 
from o2_policy import MBRLPPAModel, BasicBlock

from ipdb import set_trace

# # data_path = "outputs/2023-09-18/15-13-21/logger_log/dqn_16bits/dqn16bits_reset/dqn16bits_reset_2023_09_18_15_13_27_0000--s-0/itr_250.npy"

# data_path = "outputs/2023-11-11/00-43-49/logger_log/dqn_16bits/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool/dqn16bits_reset_rnd_sum_int_scale3_fix_mask_reset_random_20_pool_2023_11_11_00_43_56_0000--s-1/itr_2500.npy"

# da = np.load(data_path, allow_pickle=True).item()

# set_trace()
# print(da['replay_buffer'])

ppa_model_path = "./offline_sl/saved_models/ground_truth_model/ppa_model_16_bits_v1syn_mannualverilog.pkl"
ppa_model = MBRLPPAModel(
    BasicBlock
).to("cuda:0")
ppa_model.load_state_dict(torch.load(ppa_model_path))
print(ppa_model)
