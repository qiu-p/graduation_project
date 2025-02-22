template = """
nohup python ./o6_motivation_expriment_1.py --random_num 100 --ct_type {} --max_stage_num {} --state_npy_path "{}" --bit_width {} --target_delay_list {} --encode_type {} --build_base_dir "pybuild/random" --log_base_dir "log/random_refine" --n_processing 12 --index {} > cmd_outs/expr2_{} 2>&1 &
sleep 2s
echo "deploying {}-th task"
"""

target_delay_dict = {
    16: [50, 200, 500, 1200],
    32: [50, 300, 600, 2000],
}

counter = 0
script = ""
for ct_type in ["dadda", "loaded"]:
    script += template.format(
        ct_type,
        16,
        "report/2020-11-8/state-16-booth.npy",
        16,
        "50 200 500 1200",
        "booth",
        counter,
        f"random_{counter}.out",
        counter,
    )
    counter += 1

    script += template.format(
        ct_type,
        16,
        "report/2020-11-8/state-16-and.npy",
        16,
        "50 200 500 1200",
        "and",
        counter,
        f"random_{counter}.out",
        counter,
    )
    counter += 1


with open("script1.sh", "w") as file:
    file.write(script)
