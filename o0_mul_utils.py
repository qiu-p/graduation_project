import copy
import json
import multiprocessing
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from o0_adder_utils import adder_output_verilog_all, get_init_cell_map, adder_output_verilog_from_ct_v1

is_debug = False

FA_verilog_src_1 = """
module FA (a, b, cin, sum, cout);
    input a;
    input b;
    input cin;
    output sum;
    output cout;
    wire  a_xor_b = a ^ b; 
    wire  a_and_b = a & b; 
    wire  a_and_cin = a & cin; 
    wire  b_and_cin = b & cin; 
    wire  _T_1 = a_and_b | b_and_cin;
    assign sum = a_xor_b ^ cin;
    assign cout = _T_1 | a_and_cin; 
endmodule
"""

FA_verilog_src_2 = """
module FA_1 (a, b, cin, sum, cout);
    input a;
    input b;
    input cin;
    output sum;
    output cout;

    assign sum = (a ^ b) ^ cin;
    assign cout = (a & b) | (cin & (a ^ b));

endmodule
"""

FA_verilog_src_3 = """
module LUT3(
    input I0,    // 输入0
    input I1,    // 输入1
    input I2,    // 输入2
    output O     // 输出
);
    parameter [7:0] INIT = 8'h00;  // 8位查找表初始化参数

    assign O = INIT[{I2, I1, I0}];  // 三个输入决定输出值

endmodule


module FA_LUT(a, b, cin, sum, cout);

input a,b;
input cin;
output sum,cout;

LUT3 #(
      .INIT(8'h96)  // Specify LUT Contents    查找表值
   ) LUT3_inst1 (
      .O(sum),   // LUT general output 结果输出端
      .I0(cin), // LUT input
      .I1(b), // LUT input
      .I2(a)  // LUT input
   );

LUT3 #(
      .INIT(8'hE8)  // Specify LUT Contents 	查找表值
   ) LUT3_inst2 (
      .O(cout),   // LUT general output 进位输出端
      .I0(cin), // LUT input
      .I1(b), // LUT input
      .I2(a)  // LUT input
   );

endmodule
"""

FA_verilog_src = FA_verilog_src_1 + "\n" + FA_verilog_src_2 + "\n" + FA_verilog_src_3
legal_FA_list = ["FA", "FA_LUT", "FA_1"]
# legal_FA_list = ["FA", "FA_LUT"]
FA_src_list = [FA_verilog_src_1, FA_verilog_src_3, FA_verilog_src_2]

HA_verilog_src_1 = """module HA (a, cin, sum, cout);
    input a;
    input cin;
    output sum;
    output cout;
    assign sum = a ^ cin; 
    assign cout = a & cin; 
endmodule\n"""

HA_verilog_src_2 = """

module LUT2(
    input I0,    // 输入0
    input I1,    // 输入1
    output O     // 输出
);
    parameter [3:0] INIT = 4'b0000;  // 4位查找表初始化参数

    assign O = INIT[{I1, I0}];  // 两个输入决定输出值

endmodule

module HA_LUT(a, cin, sum, cout);
    input a;
    input cin;
    output sum;
    output cout;

    LUT2 #(
        .INIT(4'b0110)  // sum = a XOR b 的查找表
    ) LUT2_inst1 (
        .O(sum),   // 输出和
        .I0(a),    // LUT 输入
        .I1(cin)     // LUT 输入
    );

    LUT2 #(
        .INIT(4'b1000)  // cout = a AND b 的查找表
    ) LUT2_inst2 (
        .O(cout),   // 输出进位
        .I0(a),     // LUT 输入
        .I1(cin)      // LUT 输入
    );

endmodule
"""

HA_verilog_src = HA_verilog_src_1 + HA_verilog_src_2
legal_HA_list = ["HA", "HA_LUT"]
# legal_HA_list = ["HA"]
HA_src_list = [HA_verilog_src_1, HA_verilog_src_2]


def get_initial_partial_product(bit_width: int, encode_type: str) -> np.ndarray:
    if encode_type == "and":
        pp = np.zeros([bit_width * 2 - 1])
        for i in range(0, bit_width):
            pp[i] = i + 1
        for i in range(bit_width, bit_width * 2 - 1):
            pp[i] = bit_width * 2 - 1 - i
    elif encode_type == "booth":
        pp = np.zeros([bit_width * 2])
        if bit_width % 2 == 0:
            max = bit_width / 2 + 1
        else:
            max = bit_width / 2
        j = 3
        pos1, pos2 = {}, {}
        for i in range(0, bit_width + 4):
            pos1[i] = 1
        pos1[bit_width + 4] = 2
        for i in range(bit_width + 5, bit_width * 2, 2):
            pos1[i] = j
            pos1[i + 1] = j
            if j < max:
                j = j + 1
        k = 2
        for i in range(0, bit_width * 2, 2):
            pos2[i] = k
            pos2[i + 1] = k
            if k < max:
                k = k + 1
        for i in range(0, bit_width * 2):
            pp[i] = pos2[i] - pos1[i] + 1
    else:
        raise NotImplementedError

    return pp.astype(int)


def get_wallace_tree(initial_pp: np.ndarray, bit_width: int):
    pp_len = len(initial_pp)
    max_stage_num = pp_len
    stage_num = 0

    sequence_pp = np.zeros([1, pp_len])
    sequence_pp[0] = copy.deepcopy(initial_pp)
    ct32_decomposed = np.zeros([1, pp_len])
    ct22_decomposed = np.zeros([1, pp_len])
    target = np.asarray([2 for i in range(pp_len)])

    while stage_num < max_stage_num:
        # 构造 ct
        for i in range(0, pp_len):
            if sequence_pp[stage_num][i] % 3 == 0:
                ct32_decomposed[stage_num][i] = sequence_pp[stage_num][i] // 3
                ct22_decomposed[stage_num][i] = 0
            elif sequence_pp[stage_num][i] % 3 == 1:
                ct32_decomposed[stage_num][i] = sequence_pp[stage_num][i] // 3
                ct22_decomposed[stage_num][i] = 0
            elif sequence_pp[stage_num][i] % 3 == 2:
                ct32_decomposed[stage_num][i] = sequence_pp[stage_num][i] // 3
                if stage_num == 0:
                    ct22_decomposed[stage_num][i] = 0
                else:
                    ct22_decomposed[stage_num][i] = 1
        # 构造下一阶段的 pp
        sequence_pp = np.r_[sequence_pp, np.zeros([1, pp_len])]
        sequence_pp[stage_num + 1][0] = (
            sequence_pp[stage_num][0] - ct32_decomposed[stage_num][0] * 2 - ct22_decomposed[stage_num][0]
        )
        for i in range(1, pp_len):
            sequence_pp[stage_num + 1][i] = (
                sequence_pp[stage_num][i]
                + ct32_decomposed[stage_num][i - 1]
                + ct22_decomposed[stage_num][i - 1]
                - ct32_decomposed[stage_num][i] * 2
                - ct22_decomposed[stage_num][i]
            )
        stage_num += 1

        # 判断是否终止
        if (sequence_pp[stage_num] <= target).all():
            break

        ct32_decomposed = np.r_[ct32_decomposed, np.zeros([1, pp_len])]
        ct22_decomposed = np.r_[ct22_decomposed, np.zeros([1, pp_len])]

    assert stage_num < max_stage_num, "Exceed max stage num! Set max_stage_num larger"
    ct32 = np.sum(ct32_decomposed, axis=0)
    ct22 = np.sum(ct22_decomposed, axis=0)
    return ct32, ct22, ct32_decomposed, ct22_decomposed, sequence_pp.astype(int), stage_num


def _get_dadda_tree(initial_pp: np.ndarray, bit_width: int):
    pp_len = len(initial_pp)
    max_stage_num = pp_len
    stage_num = 0

    d = []
    d_j = 2
    for j in range(max_stage_num):
        d.append(d_j)
        d_j = int(np.floor(1.5 * d_j))

    remain_pp = copy.deepcopy(initial_pp)

    ct32 = np.zeros(pp_len)
    ct22 = np.zeros(pp_len)

    for j in range(max_stage_num - 1, 0 - 1, -1):
        d_j = d[j]
        i = 0

        while i <= len(remain_pp) - 1:
            if remain_pp[i] <= d_j:
                i += 1
                continue
            elif remain_pp[i] == d_j + 1:
                ct22[i] += 1
                remain_pp[i + 1] += 1
                remain_pp[i] -= 1
                i += 1
                continue
            else:
                ct32[i] += 1
                if i + 1 < len(remain_pp):
                    remain_pp[i + 1] += 1
                remain_pp[i] -= 2
                continue

    return ct32, ct22

def get_dadda_tree(initial_pp: np.ndarray, bit_width: int):
    ct32, ct22 = _get_dadda_tree(initial_pp, bit_width)
    ct32_decomposed, ct22_decomposed, sequence_pp, stage_num = decompose_compressor_tree(initial_pp, ct32, ct22)
    return ct32, ct22, ct32_decomposed, ct22_decomposed, sequence_pp, stage_num

def get_compressor_tree(pp: np.ndarray, bit_width: int, compressor_tree_type: str):
    if compressor_tree_type == "wallace":
        return get_wallace_tree(pp, bit_width)
    elif compressor_tree_type == "dadda":
        return get_dadda_tree(pp, bit_width)
    else:
        raise NotImplementedError


def get_final_partial_product(initial_partial_product: np.ndarray, ct):
    final_partial_product = np.zeros(len(initial_partial_product))
    ct32, ct22 = ct
    for i in range(1, len(initial_partial_product)):
        final_partial_product[i] = (
            initial_partial_product[i]
            + ct32[i - 1]
            + ct22[i - 1]
            - 2 * ct32[i]
            - ct22[i]
        )

    return final_partial_product


def decompose_compressor_tree(initial_pp, ct32, ct22):
    assert len(initial_pp) == len(ct32) and len(initial_pp) == len(ct22)
    ct32_remain = copy.deepcopy(ct32)
    ct22_remain = copy.deepcopy(ct22)
    stage_num = 0
    ct32_decomposed = np.zeros([1, len(ct32)])
    ct22_decomposed = np.zeros([1, len(ct22)])

    ct32_decomposed[0] = copy.deepcopy(ct32)
    ct22_decomposed[0] = copy.deepcopy(ct22)
    partial_products = np.zeros([1, len(initial_pp)])
    partial_products[0] = copy.deepcopy(initial_pp)

    # decompose each column sequentially
    for i in range(0, len(initial_pp)):
        j = 0  # j denotes the stage index, i denotes the column index
        while j <= stage_num:  # the condition is impossible to satisfy
            if is_debug:
                print('col: {}, stage: {}'.format(i, j))

            # j-th stage i-th column
            ct32_decomposed[j][i] = ct32_remain[i]
            ct22_decomposed[j][i] = ct22_remain[i]
            # initial j-th stage partial products
            if j == 0:  # 0th stage
                partial_products[j][i] = initial_pp[i]
            else:
                partial_products[j][i] = partial_products[j - 1][i]
            if is_debug and i<=2:
                print('ct32_decomposed: {}'.format(ct32_decomposed))
                print('ct22_decomposed: {}'.format(ct22_decomposed))
                print('partial_products: {}'.format(partial_products))
                print('ct32_decomposed[j][i]: {}, ct22_decomposed[j][i]: {}'.format(ct32_decomposed[j][i], ct22_decomposed[j][i]))
                print('partial_products[j][i]: {}'.format(partial_products[j][i]))

            # when to break
            if ct32_decomposed[j][i]==0 and ct22_decomposed[j][i]==0:
                if i == 0:
                    partial_products[j][i] = (
                        partial_products[j][i]
                    )
                else:
                    partial_products[j][i] = (
                        partial_products[j][i]
                        + ct32_decomposed[j][i - 1]
                        + ct22_decomposed[j][i - 1]
                    )
            elif (
                3 * ct32_decomposed[j][i] + 2 * ct22_decomposed[j][i]
            ) <= partial_products[j][i]:
                # update j-th stage partial products for the next stage
                if i == 0:
                    partial_products[j][i] = (
                        partial_products[j][i]
                        - ct32_decomposed[j][i] * 2
                        - ct22_decomposed[j][i]
                    )
                else:
                    partial_products[j][i] = (
                        partial_products[j][i]
                        - ct32_decomposed[j][i] * 2
                        - ct22_decomposed[j][i]
                        + ct32_decomposed[j][i - 1]
                        + ct22_decomposed[j][i - 1]
                    )
                # update the next state compressors
                ct32_remain[i] -= ct32_decomposed[j][i]
                ct22_remain[i] -= ct22_decomposed[j][i]
                if is_debug and i<=2:
                    print('3*ct32_decomposed[j][i]+2*ct22_decomposed[j][i] <= partial_products[j][i]')
                    print('ct32_decomposed: {}'.format(ct32_decomposed))
                    print('ct22_decomposed: {}'.format(ct22_decomposed))
                    print('partial_products: {}'.format(partial_products))
            else:
                if j == stage_num:
                    # print(f"j {j} stage num: {stage_num}")
                    # add initial next stage partial products and cts
                    stage_num += 1
                    ct32_decomposed = np.r_[ct32_decomposed, np.zeros([1, len(ct32)])]
                    ct22_decomposed = np.r_[ct22_decomposed, np.zeros([1, len(ct22)])]
                    partial_products_add = np.zeros([1, len(initial_pp)])
                    partial_products_add[0] = copy.deepcopy(partial_products[j])
                    partial_products = np.r_[partial_products, partial_products_add]
                # assign 3:2 first, then assign 2:2
                # only assign the j-th stage i-th column compressors
                if ct32_decomposed[j][i] >= partial_products[j][i] // 3:
                    ct32_decomposed[j][i] = partial_products[j][i] // 3
                    if partial_products[j][i] % 3 == 2:
                        if ct22_decomposed[j][i] >= 1:
                            ct22_decomposed[j][i] = 1
                    else:
                        ct22_decomposed[j][i] = 0
                    if is_debug and i<=2:
                        print('ct32_decomposed[j][i] >= partial_products[j][i] // 3')
                        print('ct32_decomposed: {}'.format(ct32_decomposed))
                        print('ct22_decomposed: {}'.format(ct22_decomposed))
                else:
                    ct32_decomposed[j][i] = ct32_decomposed[j][i]
                    if (
                        ct22_decomposed[j][i]
                        >= (partial_products[j][i] - ct32_decomposed[j][i] * 3) // 2
                    ):
                        ct22_decomposed[j][i] = (
                            partial_products[j][i] - ct32_decomposed[j][i] * 3
                        ) // 2
                    else:
                        ct22_decomposed[j][i] = ct22_decomposed[j][i]
                    if is_debug and i<=2:
                        print('ct32_decomposed[j][i] < partial_products[j][i] // 3')
                        print('ct32_decomposed: {}'.format(ct32_decomposed))
                        print('ct22_decomposed: {}'.format(ct22_decomposed))

                # update partial products
                if i == 0:
                    partial_products[j][i] = (
                        partial_products[j][i]
                        - ct32_decomposed[j][i] * 2
                        - ct22_decomposed[j][i]
                    )
                else:
                    partial_products[j][i] = (
                        partial_products[j][i]
                        - ct32_decomposed[j][i] * 2
                        - ct22_decomposed[j][i]
                        + ct32_decomposed[j][i - 1]
                        + ct22_decomposed[j][i - 1]
                    )
                if is_debug and i<=2:
                    print('partial_products: {}'.format(partial_products))
                ct32_remain[i] = ct32_remain[i] - ct32_decomposed[j][i]
                ct22_remain[i] = ct22_remain[i] - ct22_decomposed[j][i]
            j += 1
    
    stage_num += 1
    sequence_pp = np.insert(partial_products, 0, initial_pp, axis=0).astype(int)
    if is_debug:
        print('\n\nct32_remain', ct32_remain)
        print('ct22_remain', ct22_remain)

    return ct32_decomposed, ct22_decomposed, sequence_pp, stage_num


def legalize_compressor_tree(initial_pp, ct32, ct22):
    """
    这里会尽可能多的去使用 3:2 压缩器, 尽可能少地使用 2:2 压缩器
    """
    assert len(ct32) == len(initial_pp) and len(ct22) == len(initial_pp)
    ct32 = copy.deepcopy(ct32)
    ct22 = copy.deepcopy(ct22)
    for column_index in range(0, len(initial_pp)):
        # 首先, 压缩器数量就不能是负的
        ct32[column_index] = max(0, ct32[column_index])
        ct22[column_index] = max(0, ct22[column_index])
        if column_index == 0:
            remain_pp = (
                initial_pp[column_index] - 2 * ct32[column_index] - ct22[column_index]
            )
        else:
            remain_pp = (
                initial_pp[column_index]
                + ct32[column_index - 1]
                + ct22[column_index - 1]
                - 2 * ct32[column_index]
                - ct22[column_index]
            )
        if remain_pp < 1:
            # 当前列的 ct 太多了 先尝试删除 2:2 压缩器
            if ct22[column_index] + remain_pp >= 1:
                # ct22 是够的
                ct22[column_index] += remain_pp - 1
            else:
                # ct22 不太够 尝试删除 3:2 压缩器
                remain_pp += ct22[column_index]
                ct22[column_index] = 0
                if remain_pp % 2 == 0:
                    ct32[column_index] -= (2 - remain_pp) // 2
                else:
                    ct32[column_index] -= (1 - remain_pp) // 2
        elif remain_pp > 2:
            # 当前列的 ct 太少了 先尝试替换 2:2 为 3:2
            if remain_pp - ct22[column_index] <= 2:
                # ct22 是够的
                ct22[column_index] -= remain_pp - 2
                ct32[column_index] += remain_pp - 2
            else:
                # ct22 不太够的 尝试添加 3:2
                # 先替换
                ct32[column_index] += ct22[column_index]
                remain_pp -= ct22[column_index]
                ct22[column_index] = 0
                # 再添加
                if remain_pp % 2 == 0:
                    ct32[column_index] += (remain_pp - 2) / 2
                else:
                    ct32[column_index] += (remain_pp - 1) / 2

    remain_pp = copy.deepcopy(initial_pp)
    remain_pp[0] = initial_pp[0] - 2 * ct32[0] - ct22[0]
    for column_index in range(1, len(initial_pp)):
        remain_pp[column_index] = (
            initial_pp[column_index]
            + ct32[column_index - 1]
            + ct22[column_index - 1]
            - 2 * ct32[column_index]
            - ct22[column_index]
        )
    remain_pp = np.asarray(remain_pp)
    # $$DEBUG
    # assert (remain_pp >= 1).all() and (remain_pp <= 2).all(), "legalize fail"
    return ct32, ct22


def write_booth_selector(input_width: int) -> str:
    """
    input_width: 乘数和被乘数的位宽.
    booth 编码器, 相当于图 11.80 中的 encoder + selector.

    在 module 中，各个参数为：
        y   : 被乘数
        x   : 从乘数中取出的三位数
        pp  : 部分积，没算上符号位。
            注意, 这里的负的部分积正常来说应当是取反加一, 但是这里单独加入的取反操作, 将加一融合到了压缩树中。
        sgn : 符号位.

    需务必注意 -0 和 0 的区别.
    """
    booth_selector = (
        f"""
    module BoothEncoder (
        y,
        x,
        pp,
        sgn
    );
        parameter bitwidth = {input_width};

        input wire[bitwidth - 1: 0] y;
        input wire[3 - 1: 0]x;
        
        output wire sgn;
        output wire[bitwidth: 0] pp;

        wire[bitwidth: 0] y_extend;
        wire[bitwidth: 0] y_extend_shifted;
    """
        + r"""

        assign y_extend = {1'b0, y};
        assign y_extend_shifted = {y, 1'b0};
    """
        + f"""

        wire single, double, neg;

        assign single = x[0] ^ x[1];
        assign double = (x[0] & x[1] & (~x[2])) | ((~x[0]) & (~x[1]) & x[2]);
        assign neg = x[2];

        wire[bitwidth: 0] single_extend, double_extend, neg_extend;

        genvar i;
        generate
        for (i = 0; i < {input_width} + 1; i = i + 1) begin : bit_assign
            assign single_extend[i] = single;
            assign double_extend[i] = double;
            assign neg_extend[i] = neg;
        end
        endgenerate

        assign pp = neg_extend ^ ((single_extend & y_extend) | (double_extend & y_extend_shifted));
        assign sgn = neg;

    endmodule
    """
    )
    return booth_selector


def instance_booth_selector(
    name: str, y: str, x: str, pp: str, sgn: str, format="\t"
) -> str:
    """
    例化模块.

    name: 模块的名称.
    format: 用来传入前面有几个 \t 之类的格式相关内容.

    其余的和 write_booth_selector 的模块中定义相同.
    """
    booth_selector_str = (
        f"{format}BoothEncoder {name}(.y({y}), .x({x}), .pp({pp}), .sgn({sgn}));\n"
    )
    return booth_selector_str


def write_production_booth(input_width: int) -> str:
    """
    输入: 乘法器位宽 (注意, 这里假设了位宽是偶数且大于4)
    输出: booth_production 字符串.

    结构参考了图 11.82 (b)

    部分积的个数为 input_width // 2 + 1, 每次错开两位.
    最后一个部分积有 input_width 位, 其余的有 input_width + 1 位 (因为需要包含左移后的数).
    为了兼容, 这里仍给最后一个部分积然定义了 input_width + 1 位, 但是第一位悬空了没用.

    并且在右侧为了兼容, 给没有 s 的列添加了 0 作为补齐.

    总输出位数为: input_width + 2 * (input_width // 2) = 2 * input_width, 比 and 方式的多了 1 位.

    xxx_wos := with out sign (去掉了符号相关的xxx)
    """

    wire_constant_dict = {}

    str_width = 2 * input_width
    num_pp = input_width // 2 + 1
    len_pp_wos = input_width + 1
    len_output = []

    # step_0: 输入输出端口
    booth_pp_str = "module production (\n\tx,\n\ty"
    for i in range(str_width):
        booth_pp_str += f",\n\tout{i}"
    booth_pp_str += "\n);\n"

    # step_1: 设置输入位宽
    booth_pp_str += f"\tinput wire[{input_width} - 1: 0] x;\n"
    booth_pp_str += f"\tinput wire[{input_width} - 1: 0] y;\n\n"

    # step_2: 设置输出位宽
    # 前 input_width - 4 个输出
    tmp_output_len = 2
    for i in range((input_width - 4) // 2):
        booth_pp_str += f"\toutput wire[{tmp_output_len} - 1: 0] out{2 * i};\n"
        len_output.append(tmp_output_len)

        booth_pp_str += f"\toutput wire[{tmp_output_len + 1} - 1: 0] out{2 * i + 1};\n"
        len_output.append(tmp_output_len + 1)

        tmp_output_len += 1
    booth_pp_str += "\n"

    # 单独处理中间四位
    booth_pp_str += f"\toutput wire[{num_pp} - 1: 0] out{input_width - 4};\n"
    len_output.append(num_pp)
    booth_pp_str += f"\toutput wire[{num_pp} - 1: 0] out{input_width - 3};\n"
    len_output.append(num_pp)
    booth_pp_str += f"\toutput wire[{num_pp} - 1: 0] out{input_width - 2};\n"
    len_output.append(num_pp)
    booth_pp_str += f"\toutput wire[{num_pp} - 1: 0] out{input_width - 1};\n"
    len_output.append(num_pp)
    booth_pp_str += "\n"

    # 后 input_width 个输出
    tmp_output_len = input_width // 2
    for ii in range(input_width // 2):
        booth_pp_str += (
            f"\toutput wire[{tmp_output_len + 1} - 1: 0] out{input_width + 2 * ii};\n"
        )
        len_output.append(tmp_output_len + 1)

        booth_pp_str += f"\toutput wire[{tmp_output_len + 1} - 1: 0] out{input_width + 2 * ii + 1};\n"
        len_output.append(tmp_output_len + 1)

        tmp_output_len -= 1

    booth_pp_str += "\n"

    # step_3: 产生部分积。 pp_wos_xx 代表的是不算上符号位，且2补数末尾不加1的部分积
    # 单独处理 x_pp_0
    booth_pp_str += "\twire[3 - 1: 0] x_pp_0;\n"
    booth_pp_str += "\tassign x_pp_0 = {x[1: 0], 1'b0};\n"  # 补个0
    booth_pp_str += f"\twire[{input_width + 1} - 1: 0] pp_wos_0;\n"  # 部分积
    booth_pp_str += "\twire sgn_0;\n"  # 符号位
    booth_pp_str += instance_booth_selector(
        f"booth_selector_{0}", "y", f"x_pp_{0}", f"pp_wos_{0}", f"sgn_{0}"
    )  # 例化
    booth_pp_str += "\n"

    # x_pp_1 到 x_pp_{num_pp - 2}
    for i in range(1, num_pp - 1):
        booth_pp_str += f"\twire[3 - 1: 0] x_pp_{i};\n"
        booth_pp_str += f"\tassign x_pp_{i} = x[{i * 2 + 1}: {i * 2 - 1}];\n"  # 根据将 x 的位连接到 x_pp_xxx
        booth_pp_str += f"\twire[{input_width + 1} - 1: 0] pp_wos_{i};\n"  # 部分积
        booth_pp_str += f"\twire sgn_{i};\n"  # 符号位

        booth_pp_str += instance_booth_selector(
            f"booth_selector_{i}", "y", f"x_pp_{i}", f"pp_wos_{i}", f"sgn_{i}"
        )  # 例化
        booth_pp_str += "\n"

    # 单独处理 x_pp_{num_pp - 1}
    booth_pp_str += f"\twire[3 - 1: 0] x_pp_{num_pp - 1};\n"
    booth_pp_str += (
        f"\tassign x_pp_{num_pp - 1} = "
        + "{"
        + f"2'b00, x[{input_width - 1}]"
        + "};\n"  # 补0
    )
    booth_pp_str += f"\twire[{input_width + 1} - 1: 0] pp_wos_{num_pp - 1};\n"  # 部分积
    booth_pp_str += f"\twire sgn_{num_pp - 1};\n"  # 符号位
    booth_pp_str += instance_booth_selector(
        f"booth_selector_{num_pp - 1}",
        "y",
        f"x_pp_{num_pp - 1}",
        f"pp_wos_{num_pp - 1}",
        f"sgn_{num_pp - 1}",
    )  # 例化
    booth_pp_str += "\n"

    # step_4: 赋值给输出 (部分积)
    # out0 ~ out{input_width - 1}
    offset = input_width // 2
    tmp_wos_len = 1
    for i in range(0, input_width // 2):
        for j in range(tmp_wos_len):
            booth_pp_str += f"\tassign out{2 * i}[{j}] = pp_wos_{j + offset}[{len_pp_wos - 2 * j - 1} - 1];\n"
        booth_pp_str += "\n"
        for j in range(tmp_wos_len + 1):
            booth_pp_str += f"\tassign out{2 * i + 1}[{j}] = pp_wos_{j + offset - 1}[{len_pp_wos - 2 * j} - 1];\n"

        booth_pp_str += "\n"
        offset -= 1
        tmp_wos_len += 1

    # 剩下的
    offset = 2
    tmp_wos_len = input_width // 2
    for i in range(0, input_width // 2):
        for j in range(tmp_wos_len):
            booth_pp_str += f"\tassign out{2 * i + input_width}[{j}] = pp_wos_{j}[{len_pp_wos - 2 * j - offset}];\n"
        booth_pp_str += "\n"
        for j in range(tmp_wos_len):
            booth_pp_str += f"\tassign out{2 * i + 1 + input_width}[{j}] = pp_wos_{j}[{len_pp_wos - 2 * j - 1 - offset}];\n"

        booth_pp_str += "\n"
        offset += 2
        tmp_wos_len -= 1
    booth_pp_str += "\n"

    # step_5: 赋值给输出 (符号位)
    # 左
    for i in range(input_width // 2 - 2):
        booth_pp_str += (
            f"\tassign out{2 * i}[{len_output[2 * i]} - 1] = ~sgn_{num_pp - i - 2};\n"
        )
        booth_pp_str += (
            f"\tassign out{2 * i + 1}[{len_output[2 * i + 1]} - 1] = 1'b1;\n\n"
        )
        wire_constant_dict[f"out{2 * i + 1}[{len_output[2 * i]}]"] = {
            "duty": 0.5,
            "frequency": 1e8,
        }
    booth_pp_str += "\n"

    # 中间那四个特殊的
    booth_pp_str += (
        f"\tassign out{input_width - 4}[{len_output[input_width - 4]} - 1] = ~sgn_0;\n"
    )
    booth_pp_str += f"\tassign out{input_width - 4}[{len_output[input_width - 4] - 1} - 1] = ~sgn_1;\n"

    booth_pp_str += (
        f"\tassign out{input_width - 3}[{len_output[input_width - 3]} - 1] = sgn_0;\n"
    )

    booth_pp_str += (
        f"\tassign out{input_width - 2}[{len_output[input_width - 2]} - 1] = sgn_0;\n"
    )
    booth_pp_str += "\n"

    # 右
    for i in range(input_width // 2):
        booth_pp_str += f"\tassign out{2 * i + input_width}[{len_output[2 * i + input_width]} - 1] = 1'b0;\n"  # 补0
        wire_constant_dict[
            f"out{2 * i + input_width}[{len_output[2 * i + input_width - 1]}]"
        ] = {
            "duty": 0.5,
            "frequency": 0.5,
        }
        booth_pp_str += f"\tassign out{2 * i + input_width + 1}[{len_output[2 * i + input_width + 1]} - 1] = sgn_{num_pp - i - 2};\n"
    booth_pp_str += "\n"
    # 收尾
    booth_pp_str += "endmodule\n"
    return booth_pp_str, wire_constant_dict


def instance_FA(instance_name, port1, port2, port3, outport1, outport2, fa_type):
    assert fa_type < len(legal_FA_list)
    FA_name = legal_FA_list[fa_type]
    FA_str = "\t{} {}(.a({}),.b({}),.cin({}),.sum({}),.cout({}));\n".format(
        FA_name, instance_name, port1, port2, port3, outport1, outport2
    )
    return FA_str


def instance_HA(instance_name, port1, port2, outport1, outport2, ha_type):
    assert ha_type < len(legal_HA_list)
    HA_name = legal_HA_list[ha_type]
    HA_str = "\t{} {}(.a({}),.cin({}),.sum({}),.cout({}));\n".format(
        HA_name, instance_name, port1, port2, outport1, outport2
    )
    return HA_str


def update_remain_pp(ct, stage, final_stage_pp):
    ct32 = ct[0][stage][:]
    ct22 = ct[1][stage][:]

    str_width = ct.shape[2]
    initial_state = np.zeros((str_width))

    for i in range(str_width):
        if i == str_width - 1:
            initial_state[i] = final_stage_pp[i] - ct32[i] - ct22[i]
        else:
            initial_state[i] = (
                final_stage_pp[i] - ct32[i] - ct22[i] - ct32[i + 1] - ct22[i + 1]
            )
    initial_state = initial_state.astype(int)
    return initial_state


def update_final_pp(ct, stage, mult_type):
    ct32 = np.sum(ct[0][: stage + 1][:], axis=0)
    ct22 = np.sum(ct[1][: stage + 1][:], axis=0)

    str_width = len(ct32)
    input_width = (str_width + 1) // 2

    initial_state = get_initial_partial_product(input_width, mult_type)
    initial_state = initial_state[::-1]
    for i in range(str_width):
        if i == str_width - 1:
            initial_state[i] = initial_state[i] - 2 * ct32[i] - ct22[i]
        else:
            initial_state[i] = (
                initial_state[i] - 2 * ct32[i] - ct22[i] + ct32[i + 1] + ct22[i + 1]
            )
    initial_state = initial_state.astype(int)
    return initial_state


def write_CT(input_width, mult_type, ct, pp_wiring, compressor_map):
    """
    input:
        *input_width:乘法器位宽
        *ct: 压缩器信息，shape为2*stage*str_width
    """
    assert pp_wiring is not None, "only for debug"
    stage, str_width = ct.shape[1], ct.shape[2]
    wire_connect_dict = {}
    compressor_connect_dict = {"HA": {}, "FA": {}}

    # 输入输出端口
    ct_str = "module Compressor_Tree(a,b"
    for i in range(str_width):
        ct_str += ",data{}_s{}".format(i, stage)
    ct_str += ");\n"

    # 位宽
    ct_str += "\tinput[{}:0] a;\n".format(input_width - 1)
    ct_str += "\tinput[{}:0] b;\n".format(input_width - 1)

    final_state = update_final_pp(ct, stage, mult_type)
    initial_state = get_initial_partial_product(input_width, mult_type)
    initial_state = initial_state[::-1]

    # TODO: 根据每列最终的部分积确定最终的输出位宽
    for i in range(str_width):
        ct_str += "\toutput[{}:0] data{}_s{};\n".format(
            int(final_state[i]) - 1, i, stage
        )

    # 调用production模块，产生部分积
    ct_str += "\n\t//pre-processing block : production\n"
    for i in range(str_width):
        ct_str += "\t(* keep *) wire[{}:0] out{};\n".format(
            int(initial_state[i]) - 1, i
        )
        # ct_str += "\twire[{}:0] out{};\n".format(int(initial_state[i]) - 1, i)

    if mult_type == "booth":
        ct_str += "\tproduction PD0(.x(a),.y(b)"
    else:
        ct_str += "\tproduction PD0(.a(a),.b(b)"
    for i in range(str_width):
        ct_str += ",.out{}(out{})".format(i, i)
    ct_str += ");"

    # 生成每个阶段的压缩树
    num_tmp = 0
    for stage_num in range(stage):
        ct_str += "\n\t//****The {}th stage****\n".format(stage_num + 1)
        final_stage_pp = update_final_pp(ct, stage_num, mult_type)

        remain_pp = update_remain_pp(ct, stage_num, final_stage_pp)

        if stage_num < stage - 1:
            # 最后一个阶段被声明为 out
            for i in range(str_width):
                ct_str += "\t(* keep *)wire[{}:0] data{}_s{};\n".format(
                    # ct_str += "\twire[{}:0] data{}_s{};\n".format(
                    final_stage_pp[i] - 1,
                    i,
                    stage_num + 1,
                )

        for j in range(str_width):
            FA_index = 0
            HA_index = 0
            if stage_num == 0:
                port_list = []
                for k in range(ct[0][stage_num][j]):
                    port_list.append("out{}[{}]".format(j, 3 * k))
                    port_list.append("out{}[{}]".format(j, 3 * k + 1))
                    port_list.append("out{}[{}]".format(j, 3 * k + 2))

                for k in range(ct[1][stage_num][j]):
                    port_list.append(
                        "out{}[{}]".format(j, 3 * ct[0][stage_num][j] + 2 * k)
                    )
                    port_list.append(
                        "out{}[{}]".format(j, 3 * ct[0][stage_num][j] + 2 * k + 1)
                    )
                for k in range(remain_pp[j]):
                    port_list.append(
                        "out{}[{}]".format(
                            j, 3 * ct[0][stage_num][j] + 2 * ct[1][stage_num][j] + k
                        )
                    )
                port_index = 0
                for k in range(ct[0][stage_num][j]):
                    port1 = port_list[pp_wiring[stage_num][j][port_index]]
                    port2 = port_list[pp_wiring[stage_num][j][port_index + 1]]
                    port3 = port_list[pp_wiring[stage_num][j][port_index + 2]]
                    assert pp_wiring[stage_num][j][port_index] >= 0
                    assert pp_wiring[stage_num][j][port_index + 1] >= 0
                    assert pp_wiring[stage_num][j][port_index + 2] >= 0
                    port_index += 3

                    outport1 = "data{}_s{}[{}]".format(j, stage_num + 1, k)
                    if j != 0:
                        outport2 = "data{}_s{}[{}]".format(
                            j - 1,
                            stage_num + 1,
                            k
                            + ct[0][stage_num][j - 1]
                            + ct[1][stage_num][j - 1]
                            + remain_pp[j - 1],
                        )
                    else:
                        ct_str += "\twire[0:0] tmp{};\n".format(num_tmp)
                        outport2 = "tmp{}".format(num_tmp)
                        num_tmp += 1
                    FA_name = f"FA_s{stage_num}_c{j}_i{FA_index}"
                    if compressor_map is not None:
                        FA_type = compressor_map[0][stage_num][j][FA_index]
                    else:
                        FA_type = 0
                    ct_str += instance_FA(
                        FA_name, port1, port2, port3, outport1, outport2, FA_type
                    )
                    FA_index += 1
                    compressor_connect_dict["FA"][FA_name] = {
                        "port_1": port1,
                        "port_2": port2,
                        "port_3": port3,
                    }
                for k in range(ct[1][stage_num][j]):
                    port1 = port_list[pp_wiring[stage_num][j][port_index]]
                    port2 = port_list[pp_wiring[stage_num][j][port_index + 1]]
                    assert pp_wiring[stage_num][j][port_index] >= 0
                    assert pp_wiring[stage_num][j][port_index + 1] >= 0
                    port_index += 2

                    outport1 = "data{}_s{}[{}]".format(
                        j, stage_num + 1, ct[0][stage_num][j] + k
                    )
                    if j != 0:
                        outport2 = "data{}_s{}[{}]".format(
                            j - 1,
                            stage_num + 1,
                            k
                            + ct[0][stage_num][j - 1]
                            + ct[1][stage_num][j - 1]
                            + ct[0][stage_num][j]
                            + remain_pp[j - 1],
                        )
                    else:
                        ct_str += "\twire[0:0] tmp{};\n".format(num_tmp)
                        outport2 = "tmp{}".format(num_tmp)
                        num_tmp += 1
                    wire_connect_dict[outport1] = outport1
                    wire_connect_dict[outport2] = outport2
                    HA_name = f"HA_s{stage_num}_c{j}_i{HA_index}"
                    if compressor_map is not None:
                        HA_type = compressor_map[1][stage_num][j][HA_index]
                    else:
                        HA_type = 0
                    ct_str += instance_HA(
                        HA_name, port1, port2, outport1, outport2, HA_type
                    )
                    HA_index += 1
                    compressor_connect_dict["HA"][HA_name] = {
                        "port_1": port1,
                        "port_2": port2,
                    }
                # remain_ports
                for k in range(remain_pp[j]):
                    port_this = port_list[pp_wiring[stage_num][j][port_index]]
                    assert pp_wiring[stage_num][j][port_index] >= 0
                    port_index += 1

                    port_next = "data{}_s{}[{}]".format(
                        j,
                        stage_num + 1,
                        ct[0][stage_num][j] + ct[1][stage_num][j] + k,
                    )
                    if port_this in wire_connect_dict.keys():
                        wire_connect_dict[port_next] = wire_connect_dict[port_this]
                    else:
                        wire_connect_dict[port_next] = port_this
                    ct_str += "\tassign {} = {};\n".format(
                        port_next,
                        port_this,
                    )
            else:
                port_list = []
                for k in range(ct[0][stage_num][j]):
                    port_list.append("data{}_s{}[{}]".format(j, stage_num, 3 * k))
                    port_list.append("data{}_s{}[{}]".format(j, stage_num, 3 * k + 1))
                    port_list.append("data{}_s{}[{}]".format(j, stage_num, 3 * k + 2))
                for k in range(ct[1][stage_num][j]):
                    port_list.append(
                        "data{}_s{}[{}]".format(
                            j, stage_num, 3 * ct[0][stage_num][j] + 2 * k
                        )
                    )
                    port_list.append(
                        "data{}_s{}[{}]".format(
                            j, stage_num, 3 * ct[0][stage_num][j] + 2 * k + 1
                        )
                    )
                for k in range(remain_pp[j]):
                    port_list.append(
                        "data{}_s{}[{}]".format(
                            j,
                            stage_num,
                            3 * ct[0][stage_num][j] + 2 * ct[1][stage_num][j] + k,
                        )
                    )
                port_index = 0
                for k in range(ct[0][stage_num][j]):
                    port1 = port_list[pp_wiring[stage_num][j][port_index]]
                    port2 = port_list[pp_wiring[stage_num][j][port_index + 1]]
                    port3 = port_list[pp_wiring[stage_num][j][port_index + 2]]
                    assert pp_wiring[stage_num][j][port_index] >= 0
                    assert pp_wiring[stage_num][j][port_index + 1] >= 0
                    assert pp_wiring[stage_num][j][port_index + 2] >= 0
                    port_index += 3

                    outport1 = "data{}_s{}[{}]".format(j, stage_num + 1, k)
                    if j != 0:
                        outport2 = "data{}_s{}[{}]".format(
                            j - 1,
                            stage_num + 1,
                            k
                            + ct[0][stage_num][j - 1]
                            + ct[1][stage_num][j - 1]
                            + remain_pp[j - 1],
                        )
                    else:
                        ct_str += "\twire[0:0] tmp{};\n".format(num_tmp)
                        outport2 = "tmp{}".format(num_tmp)
                        num_tmp += 1
                    wire_connect_dict[outport1] = outport1
                    wire_connect_dict[outport2] = outport2
                    FA_name = f"FA_s{stage_num}_c{j}_i{FA_index}"
                    if compressor_map is not None:
                        FA_type = compressor_map[0][stage_num][j][FA_index]
                    else:
                        FA_type = 0

                    ct_str += instance_FA(
                        FA_name, port1, port2, port3, outport1, outport2, FA_type
                    )
                    FA_index += 1
                    compressor_connect_dict["FA"][FA_name] = {
                        "port_1": port1,
                        "port_2": port2,
                        "port_3": port3,
                    }
                for k in range(ct[1][stage_num][j]):
                    port1 = port_list[pp_wiring[stage_num][j][port_index]]
                    port2 = port_list[pp_wiring[stage_num][j][port_index + 1]]
                    assert pp_wiring[stage_num][j][port_index] >= 0
                    assert pp_wiring[stage_num][j][port_index + 1] >= 0
                    port_index += 2

                    outport1 = "data{}_s{}[{}]".format(
                        j, stage_num + 1, ct[0][stage_num][j] + k
                    )
                    if j != 0:
                        outport2 = "data{}_s{}[{}]".format(
                            j - 1,
                            stage_num + 1,
                            k
                            + ct[0][stage_num][j - 1]
                            + ct[1][stage_num][j - 1]
                            + ct[0][stage_num][j]
                            + remain_pp[j - 1],
                        )
                    else:
                        ct_str += "\twire[0:0] tmp{};\n".format(num_tmp)
                        outport2 = "tmp{}".format(num_tmp)
                        num_tmp += 1

                    wire_connect_dict[outport1] = outport1
                    wire_connect_dict[outport2] = outport2

                    HA_name = f"HA_s{stage_num}_c{j}_i{HA_index}"
                    if compressor_map is not None:
                        HA_type = compressor_map[1][stage_num][j][HA_index]
                    else:
                        HA_type = 0
                    ct_str += instance_HA(
                        HA_name, port1, port2, outport1, outport2, HA_type
                    )
                    HA_index += 1
                    compressor_connect_dict["HA"][HA_name] = {
                        "port_1": port1,
                        "port_2": port2,
                    }
                # remain_ports
                for k in range(remain_pp[j]):
                    port_this = port_list[pp_wiring[stage_num][j][port_index]]
                    assert pp_wiring[stage_num][j][port_index] >= 0
                    port_index += 1
                    port_next = "data{}_s{}[{}]".format(
                        j,
                        stage_num + 1,
                        ct[0][stage_num][j] + ct[1][stage_num][j] + k,
                    )
                    if port_this in wire_connect_dict.keys():
                        wire_connect_dict[port_next] = wire_connect_dict[port_this]
                    else:
                        wire_connect_dict[port_next] = port_this

                    ct_str += "\tassign {} = {};\n".format(
                        port_next,
                        port_this,
                    )
    ct_str += "endmodule\n"
    return ct_str, compressor_connect_dict, wire_connect_dict


def write_production_and(input_width):
    """
    input:
        * input_width:乘法器位宽
    return:
        * pp_str : and_production字符串
    """

    str_width = 2 * input_width - 1
    # 输入输出端口
    pp_str = "module production (a,b"
    for i in range(str_width):
        pp_str += ",out" + str(i)
    pp_str += ");\n"

    # 位宽
    pp_str += "\tinput[{}:0] a;\n".format(input_width - 1)
    pp_str += "\tinput[{}:0] b;\n".format(input_width - 1)
    for i in range(1, str_width + 1):
        len_i = input_width - abs(i - input_width)
        if len_i - 1 == 0:
            pp_str += "\toutput out{};\n".format(i - 1)
        else:
            pp_str += "\toutput[{}:0] out{};\n".format(len_i - 1, i - 1)

    # 赋值,out0代表高位
    for i in range(str_width):
        for j in range(input_width - abs(i - input_width + 1)):
            # i代表a，j代表b
            if i == 0 or i == str_width - 1:
                pp_str += "\tassign out{} = a[{}] & b[{}];\n".format(
                    i, int(input_width - i / 2 - 1), int(input_width - 1 - i / 2)
                )
            else:
                if i >= 0 and i <= input_width - 1:
                    pp_str += "\tassign out{}[{}] = a[{}] & b[{}];\n".format(
                        i, j, (input_width - i - 1 + j), (input_width - 1 - j)
                    )
                else:
                    pp_str += "\tassign out{}[{}] = a[{}] & b[{}];\n".format(
                        i, j, j, (2 * input_width - i - 2 - j)
                    )

    #
    pp_str += "endmodule\n"
    return pp_str


def write_HA():
    return HA_verilog_src


def write_FA():
    FA_str = FA_verilog_src + "\n"
    return FA_str


def __write_mul(
    mul_verilog_file,
    input_width,
    ct,
    pp_wiring,
    compressor_map,
    use_final_adder_optimize=False,
    cell_map=None,
):
    """
    input:
        * mul_verilog_file: 输出verilog路径
        *input_width: 输入电路的位宽
        *ct: 输入电路的压缩树 shape为2*stage_num*str_width
    """
    ct = ct.astype(int)[:, :, ::-1]
    if compressor_map is not None:
        compressor_map = np.asarray(compressor_map).astype(int)
        compressor_map = compressor_map[:, :, ::-1, :]
    if pp_wiring is not None:
        pp_wiring = np.asarray(pp_wiring).astype(int)
        pp_wiring = pp_wiring[:, ::-1, :]

    str_width = ct.shape[2]
    if str_width == input_width * 2 - 1:
        mult_type = "and"
    else:
        mult_type = "booth"
    # print(mult_type)
    with open(mul_verilog_file, "w") as f:
        f.write(write_FA())
        f.write(write_HA())
        CT_str, compressor_connect_dict, wire_connect_dict = write_CT(
            input_width, mult_type, ct, pp_wiring, compressor_map
        )
        if mult_type == "and":
            f.write(write_production_and(input_width))
            wire_constant_dict = {}
        else:
            booth_pp_str, wire_constant_dict = write_production_booth(input_width)
            f.write(write_booth_selector(input_width))
            f.write(booth_pp_str)
        f.write(CT_str)
        f.write("module MUL(a,b,clock,out);\n")
        f.write("\tinput clock;\n")
        f.write("\tinput[{}:0] a;\n".format(input_width - 1))
        f.write("\tinput[{}:0] b;\n".format(input_width - 1))
        f.write("\toutput[{}:0] out;\n".format(2 * input_width - 2))
        stage = ct.shape[1]
        final_pp = update_final_pp(ct, stage, mult_type)

        for i in range(len(final_pp)):
            f.write("\twire[{}:0] out{}_C;\n".format(final_pp[i] - 1, i))

        f.write("\tCompressor_Tree C0(.a(a),.b(b)")

        for i in range(len(final_pp)):
            f.write(",.data{}_s{}(out{}_C)".format(i, stage, i))
        f.write(");\n")

        if not use_final_adder_optimize:
            f.write("\twire[{}:0] addend;\n".format(str_width - 1))
            f.write("\twire[{}:0] augned;\n".format(str_width - 1))

            for i in range(len(final_pp)):
                if final_pp[len(final_pp) - i - 1] == 2:
                    f.write(
                        "\tassign addend[{}] = out{}_C[0];\n".format(
                            i, len(final_pp) - i - 1
                        )
                    )
                    f.write(
                        "\tassign augned[{}] = out{}_C[1];\n".format(
                            i, len(final_pp) - i - 1
                        )
                    )
                else:
                    f.write(
                        "\tassign addend[{}] = out{}_C[0];\n".format(
                            i, len(final_pp) - i - 1
                        )
                    )
                    f.write("\tassign augned[{}] = 1'b0;\n".format(i))
            if mult_type == "booth":
                f.write("\twire[{}:0] tmp = addend + augned;\n".format(2 * input_width - 1))
            else:
                f.write("\twire[{}:0] tmp = addend + augned;\n".format(2 * input_width - 2))
            f.write("\tassign out = tmp[{}:0];\n".format(2 * input_width - 2))
        else:
            f.write("\twire temp;\n")
            f.write("\tPrefixAdder prefix_adder_0 (")
            for i in range(len(final_pp) ):
                f.write(f".out{i}_C(out{i}_C), ")
            f.write(f".s(out), .cout(temp), .clock(clock)")
            f.write(");\n")
        f.write("endmodule\n")
        if use_final_adder_optimize:
            f.write(adder_output_verilog_from_ct_v1(cell_map, final_pp))
            # f.write(adder_output_verilog_all(cell_map, final_pp))


    return compressor_connect_dict, wire_connect_dict, wire_constant_dict


def write_mul(
    mul_verilog_file,
    input_width,
    ct,
    pp_wiring,
    compressor_map,
    use_final_adder_optimize=False,
    cell_map=None,
    is_adder_only=False,
):
    """
    input:
        * mul_verilog_file: 输出verilog路径
        *input_width: 输入电路的位宽
        *ct: 输入电路的压缩树 shape为2*stage_num*str_width
    """
    if not is_adder_only:
        return __write_mul(mul_verilog_file, input_width, ct, pp_wiring, compressor_map, use_final_adder_optimize, cell_map)
    else:
        with open(mul_verilog_file, "w") as f:
            final_pp = np.full(len(cell_map), 2)
            f.write(adder_output_verilog_from_ct_v1(cell_map, final_pp))
        return None, None, None
    

# fmt: on
def get_default_pp_wiring(max_stage_num, initial_pp, ct, init_type = "default"):
    """
    初始化部分积连线
    合法的位置是wire的map (0, 1, 2, ...)
    不合法的位置是 -1
    self.pp_wiring[stage_index][column_index] = [1, 2, 0, 3, ...]
    Parameters:
        init_type = "default" or "random"
    """
    pp_wiring = np.full(
        [max_stage_num, len(initial_pp), np.max(initial_pp).astype(int)], -1
    )

    remain_pp = copy.deepcopy(initial_pp)
    ct32_decomposed, ct22_decomposed, _, __ = decompose_compressor_tree(
        initial_pp, ct[0], ct[1]
    )
    stage_num = len(ct32_decomposed)
    for stage_index in range(stage_num):
        for column_index in range(len(initial_pp)):
            wire_num = int(remain_pp[column_index])
            if init_type == "random":
                random_index = [wire_index for wire_index in range(wire_num)]
                np.random.shuffle(random_index)
                for wire_index in range(wire_num):
                    pp_wiring[stage_index][column_index][wire_index] = random_index[wire_index]
            elif init_type == "default":
                for wire_index in range(wire_num):
                    pp_wiring[stage_index][column_index][wire_index] = wire_index

            # update remain pp
            remain_pp[column_index] += (
                -2 * ct32_decomposed[stage_index][column_index]
                - ct22_decomposed[stage_index][column_index]
            )
            if column_index > 0:
                remain_pp[column_index] += (
                    ct32_decomposed[stage_index][column_index - 1]
                    + ct22_decomposed[stage_index][column_index - 1]
                )
    return pp_wiring


def get_target_delay(bit_width):
    if bit_width == 8:
        return [50, 250, 400, 650]
    elif bit_width == 16:
        return [50, 200, 500, 1200]
    elif bit_width == 32:
        return [50, 300, 600, 2000]
    else:
        return [50, 600, 1500, 3000]

if __name__ == "__main__":
    # pass
    bit_width = 8
    encode_type = "and"
    compressor_type = "wallace"
    pp = get_initial_partial_product(bit_width, encode_type)
    ct32, ct22 = get_compressor_tree(pp, bit_width, compressor_type)
    ct32_decomposed, ct22_decomposed, partial_products, stage_num = (
        decompose_compressor_tree(pp, ct32, ct22)
    )

    # ct = np.stack(ct32_decomposed, ct22_decomposed)
    ct = np.zeros([2, len(ct32_decomposed), len(pp)])
    ct[0] = ct32_decomposed
    ct[1] = ct22_decomposed
    pp_wiring = get_default_pp_wiring(32, pp, [ct32, ct22])

    cell_map = get_init_cell_map(len(pp), "sklansky")
    write_mul("MUL.v", bit_width, ct, pp_wiring, None, use_final_adder_optimize=True, cell_map=cell_map)
    print(cell_map)
    # write_mul("MUL.v", bit_width, ct, pp_wiring, None, use_final_adder_optimize=False, cell_map=cell_map)
