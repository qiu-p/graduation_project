import numpy as np
from pygmo import hypervolume

# input npy files and reference point 16 bits booth
# npy_files = [
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-06/00-30-26/logger_log/dqn_16bits/dqn16bits_booth/dqn16bits_booth_2024_01_06_00_30_31_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-06/00-34-44/logger_log/dqn_16bits/dqn16bits_booth/dqn16bits_booth_2024_01_06_00_34_50_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-06/00-35-22/logger_log/dqn_16bits/dqn16bits_booth/dqn16bits_booth_2024_01_06_00_35_28_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-06/00-35-26/logger_log/dqn_16bits/dqn16bits_booth/dqn16bits_booth_2024_01_06_00_35_32_0000--s-1/itr_5000.npy"
# ]
# reference_point = [2700,1.8]

# input npy files and reference point 16 bits booth
# npy_files = [
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-07/01-15-30/logger_log/dqn_8bits_booth/dqn8bits_booth/dqn8bits_booth_2024_01_07_01_15_36_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-07/01-15-57/logger_log/dqn_8bits_booth/dqn8bits_booth/dqn8bits_booth_2024_01_07_01_16_04_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-07/01-16-05/logger_log/dqn_8bits_booth/dqn8bits_booth/dqn8bits_booth_2024_01_07_01_16_11_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-07/01-16-12/logger_log/dqn_8bits_booth/dqn8bits_booth/dqn8bits_booth_2024_01_07_01_16_19_0000--s-1/itr_5000.npy"
# ]
# reference_point = [900,1.4]

# input npy files and reference point 16 bits and
# npy_files = [
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-09/01-09-33/logger_log/dqn_16bits_and/dqn16bits_and/dqn16bits_and_2024_01_09_01_09_39_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-09/01-09-39/logger_log/dqn_16bits_and/dqn16bits_and/dqn16bits_and_2024_01_09_01_09_46_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-09/01-09-44/logger_log/dqn_16bits_and/dqn16bits_and/dqn16bits_and_2024_01_09_01_09_51_0000--s-1/itr_5000.npy",
#     "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-09/01-09-51/logger_log/dqn_16bits_and/dqn16bits_and/dqn16bits_and_2024_01_09_01_09_58_0000--s-1/itr_5000.npy"
# ]
# reference_point = [2800,1.6]

# input npy files and reference point 8 bits and
npy_files = [
    "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-08/00-52-39/logger_log/dqn_8bits_and/dqn8bits_and/dqn8bits_and_2024_01_08_00_52_46_0000--s-1/itr_5000.npy",
    "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-08/00-52-59/logger_log/dqn_8bits_and/dqn8bits_and/dqn8bits_and_2024_01_08_00_53_06_0000--s-1/itr_5000.npy",
    "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-08/00-53-06/logger_log/dqn_8bits_and/dqn8bits_and/dqn8bits_and_2024_01_08_00_53_13_0000--s-1/itr_5000.npy",
    "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-01-08/00-53-15/logger_log/dqn_8bits_and/dqn8bits_and/dqn8bits_and_2024_01_08_00_53_22_0000--s-1/itr_5000.npy"
]
reference_point = [700,1.2]

# get pareto points
merged_pareto_array = []
for npy_file in npy_files:
    da = np.load(npy_file, allow_pickle=True).item()
    pareto_area_points = da["pareto_area_points"] # list
    pareto_delay_points = da["pareto_delay_points"] # list
    for i in range(len(pareto_area_points)):
        point = [pareto_area_points[i], pareto_delay_points[i]]
        merged_pareto_array.append(point)
merged_pareto_array = np.array(merged_pareto_array)

# compute hypervolume
hv = hypervolume(merged_pareto_array)
hv_value = hv.compute(reference_point)

print(hv_value)