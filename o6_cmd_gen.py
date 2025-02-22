template = """
nohup python ./o6_motivation_expriment.py --random_num 100 --ct_type {} --max_stage_num {} --state_npy_path "{}" --bit_width {} --target_delay_list {} --encode_type {} --p_FA {} --p_HA {} --build_base_dir "pybuild/random" --log_base_dir "log/random_refine/LUT_COMB_only" --n_processing 12 --index {} > cmd_outs/{} 2>&1 &
sleep 2s
echo "deploying {}-th task"
"""

target_delay_dict = {
    16: [50, 200, 500, 1200],
    32: [50, 300, 600, 2000],
}

counter = 0
script = ""
# for p_FA in ["1.0 1.0 1.0", "3.0 1.0 1.0", "1.0 3.0 1.0", "1.0 1.0 3.0"]:
# for p_FA in ["1.0 1.0 0.0"]:
for p_FA in ["2.0 1.0 0.0", "1.0 2.0 0.0", "3.0 1.0 0.0", "1.0 3.0 0.0"]:
    # for  p_HA in ["1.0 1.0", "2.0 1.0", "1.0 2.0"]:
    for  p_HA in ["1.0 0.0"]:
        # for ct_type in ["dadda", "loaded"]:
        # for ct_type in ["loaded"]:
        for ct_type in ["dadda"]:
            # script += template.format(ct_type, 16, "report/2020-11-8/state-16-booth.npy", 16, "50 200 500 1200", "booth", p_FA, p_HA, counter, f"random_{counter}.out", counter)
            # counter += 1

            # script += template.format(ct_type, 16, "report/2020-11-8/state-16-and.npy", 16, "50 200 500 1200", "and", p_FA, p_HA, counter, f"random_{counter}.out", counter)
            # counter += 1

            script += template.format(ct_type, 32, "report/2020-11-8/state-32-booth.npy", 32, "50 300 600 2000", "booth", p_FA, p_HA, counter, f"random_{counter}_32.out", counter)
            counter += 1

            script += template.format(ct_type, 32, "report/2020-11-8/state-32-and.npy", 32, "50 300 600 2000", "and", p_FA, p_HA, counter, f"random_{counter}_32.out", counter)
            counter += 1

with open("script.sh", "w") as file:
    file.write(script)
