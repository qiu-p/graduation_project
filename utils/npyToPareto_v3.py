# 主要是两部分，一个是部分积生成，一个是压缩树阶段
# 1. 确认输入，输入端口，以及阶段数
# 2. 根据产生部分积，赋予每列
import numpy as np
from pygmo import hypervolume
from multiprocessing import Pool
import os

lef = "/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary.lef"
lib = "/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary_typical.lib"
OpenRoadFlowPath = "/ai4multiplier/openroad_deb/OpenROAD-flow-scripts"
refer_point = {
    "booth": {
        8: [1100, 1.4],
        #
        16: [3200, 1.8],
        32: [12000, 3.0],
        64: [45000, 3.80],
    },
    "and": {
        8: [800, 1.2],
        16: [3300, 1.6],
        32: [15000, 2.7],
        64: [48000, 3.6],
    },
}


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
        booth_pp_str += f"\tassign out{2 * i + input_width + 1}[{len_output[2 * i + input_width + 1]} - 1] = sgn_{num_pp - i - 2};\n"
    booth_pp_str += "\n"
    # 收尾
    booth_pp_str += "endmodule\n"
    return booth_pp_str


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

    initial_state = get_initial_partial_product(mult_type, input_width)
    initial_state = initial_state[0][::-1]
    # print(len(initial_state))
    for i in range(str_width):
        if i == str_width - 1:
            initial_state[i] = initial_state[i] - 2 * ct32[i] - ct22[i]
        else:
            initial_state[i] = (
                initial_state[i] - 2 * ct32[i] - ct22[i] + ct32[i + 1] + ct22[i + 1]
            )
    initial_state = initial_state.astype(int)
    return initial_state


def instance_FA(num, port1, port2, port3, outport1, outport2):
    FA_str = "\tFA F{}(.a({}),.b({}),.cin({}),.sum({}),.cout({}));\n".format(
        num, port1, port2, port3, outport1, outport2
    )
    return FA_str


def instance_HA(num, port1, port2, outport1, outport2):
    HA_str = "\tHA H{}(.a({}),.cin({}),.sum({}),.cout({}));\n".format(
        num, port1, port2, outport1, outport2
    )
    return HA_str


def write_CT(input_width, mult_type, ct=[]):
    """
    input:
        *input_width:乘法器位宽
        *ct: 压缩器信息，shape为2*stage*str_width
    """
    stage, str_width = ct.shape[1], ct.shape[2]

    # 输入输出端口
    ct_str = "module Compressor_Tree(a,b"
    for i in range(str_width):
        ct_str += ",data{}_s{}".format(i, stage)
    ct_str += ");\n"

    # 位宽
    ct_str += "\tinput[{}:0] a;\n".format(input_width - 1)
    ct_str += "\tinput[{}:0] b;\n".format(input_width - 1)

    final_state = update_final_pp(ct, stage, mult_type)
    initial_state = get_initial_partial_product(mult_type, input_width)
    initial_state = initial_state[0][::-1]
    # print("final",final_state)

    # TODO: 根据每列最终的部分积确定最终的输出位宽
    for i in range(str_width):
        ct_str += "\toutput[{}:0] data{}_s{};\n".format(
            int(final_state[i]) - 1, i, stage
        )

    # 调用production模块，产生部分积
    ct_str += "\n\t//pre-processing block : production\n"
    for i in range(str_width):
        ct_str += "\twire[{}:0] out{};\n".format(int(initial_state[i]) - 1, i)

    if mult_type == "booth":
        ct_str += "\tproduction PD0(.x(a),.y(b)"
    else:
        ct_str += "\tproduction PD0(.a(a),.b(b)"
    for i in range(str_width):
        ct_str += ",.out{}(out{})".format(i, i)
    ct_str += ");"
    FA_num = 0
    HA_num = 0

    # 生成每个阶段的压缩树
    for stage_num in range(stage):
        ct_str += "\n\t//****The {}th stage****\n".format(stage_num + 1)
        final_stage_pp = update_final_pp(ct, stage_num, mult_type)

        remain_pp = update_remain_pp(ct, stage_num, final_stage_pp)

        for i in range(str_width):
            ct_str += "\twire[{}:0] data{}_s{};\n".format(
                final_stage_pp[i] - 1, i, stage_num + 1
            )

        num_tmp = 0
        for j in range(str_width):
            if stage_num == 0:
                for k in range(ct[0][stage_num][j]):
                    port1 = "out{}[{}]".format(j, 3 * k)
                    port2 = "out{}[{}]".format(j, 3 * k + 1)
                    port3 = "out{}[{}]".format(j, 3 * k + 2)
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
                    ct_str += instance_FA(
                        FA_num, port1, port2, port3, outport1, outport2
                    )
                    FA_num += 1
                for k in range(ct[1][stage_num][j]):
                    port1 = "out{}[{}]".format(j, 3 * ct[0][stage_num][j] + 2 * k)
                    port2 = "out{}[{}]".format(j, 3 * ct[0][stage_num][j] + 2 * k + 1)
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
                    ct_str += instance_HA(HA_num, port1, port2, outport1, outport2)
                    HA_num += 1
                # remain_ports
                for k in range(remain_pp[j]):
                    ct_str += "\tassign data{}_s{}[{}] = out{}[{}];\n".format(
                        j,
                        stage_num + 1,
                        ct[0][stage_num][j] + ct[1][stage_num][j] + k,
                        j,
                        3 * ct[0][stage_num][j] + 2 * ct[1][stage_num][j] + k,
                    )
            else:
                for k in range(ct[0][stage_num][j]):
                    port1 = "data{}_s{}[{}]".format(j, stage_num, 3 * k)
                    port2 = "data{}_s{}[{}]".format(j, stage_num, 3 * k + 1)
                    port3 = "data{}_s{}[{}]".format(j, stage_num, 3 * k + 2)
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
                    ct_str += instance_FA(
                        FA_num, port1, port2, port3, outport1, outport2
                    )
                    FA_num += 1
                for k in range(ct[1][stage_num][j]):
                    port1 = "data{}_s{}[{}]".format(
                        j, stage_num, 3 * ct[0][stage_num][j] + 2 * k
                    )
                    port2 = "data{}_s{}[{}]".format(
                        j, stage_num, 3 * ct[0][stage_num][j] + 2 * k + 1
                    )
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
                    ct_str += instance_HA(HA_num, port1, port2, outport1, outport2)
                    HA_num += 1
                # remain_ports
                for k in range(remain_pp[j]):
                    ct_str += "\tassign data{}_s{}[{}] = data{}_s{}[{}];\n".format(
                        j,
                        stage_num + 1,
                        ct[0][stage_num][j] + ct[1][stage_num][j] + k,
                        j,
                        stage_num,
                        3 * ct[0][stage_num][j] + 2 * ct[1][stage_num][j] + k,
                    )
    ct_str += "endmodule\n"
    return ct_str


def write_HA():
    HA_str = """module HA (a, cin, sum, cout);
\tinput a;
\tinput cin;
\toutput sum;
\toutput cout;
\tassign sum = a ^ cin; 
\tassign cout = a & cin; 
endmodule\n"""
    return HA_str


def write_FA():
    FA_str = """module FA (a, b, cin, sum, cout);
\tinput a;
\tinput b;
\tinput cin;
\toutput sum;
\toutput cout;
\twire  a_xor_b = a ^ b; 
\twire  a_and_b = a & b; 
\twire  a_and_cin = a & cin; 
\twire  b_and_cin = b & cin; 
\twire  _T_1 = a_and_b | b_and_cin;
\tassign sum = a_xor_b ^ cin;
\tassign cout = _T_1 | a_and_cin; 
endmodule\n"""
    return FA_str


def read_ct(ct_file, mult_type):
    with open(ct_file, "r") as f:
        lines = f.readlines()
    width = int(lines[0].strip().split(" ")[0])
    stage = 0
    pre_idx = 10000
    if mult_type == "and":
        ct = np.zeros((2, 1, 2 * width - 1))
    else:
        ct = np.zeros((2, 1, 2 * width))
    for i in range(2, len(lines)):
        line = lines[i].strip().split(" ")
        idx, kind = int(line[0]), int(line[1])
        if idx > pre_idx:
            stage += 1
            if mult_type == "and":
                news = np.zeros((2, 1, 2 * width - 1))
            else:
                news = np.zeros((2, 1, 2 * width))
            ct = np.concatenate((ct, news), axis=1)
            print(ct.shape)
        pre_idx = idx
        if kind == 1:
            ct[0][stage][idx] += 1
        else:
            ct[1][stage][idx] += 1
    return ct


def write_mul(mul_verilog_file, input_width, ct):
    """
    input:
        * mul_verilog_file: 输出verilog路径
        *input_width: 输入电路的位宽
        *ct: 输入电路的压缩树，shape为2*stage_num*str_width
    """
    ct = ct.astype(int)[:, :, ::-1]
    str_width = ct.shape[2]
    print(str_width)
    if str_width == input_width * 2 - 1:
        mult_type = "and"
    else:
        mult_type = "booth"
    print(mult_type)
    with open(mul_verilog_file, "w") as f:
        f.write(write_FA())
        f.write(write_HA())
        if mult_type == "and":
            f.write(write_production_and(input_width))
        else:
            f.write(write_booth_selector(input_width))
            f.write(write_production_booth(input_width))
        f.write(write_CT(input_width, mult_type, ct))
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
        f.write("\twire[{}:0] tmp = addend + augned;\n".format(2 * input_width - 2))
        f.write("\tassign out = tmp[{}:0];\n".format(2 * input_width - 2))
        f.write("endmodule\n")


def get_initial_partial_product(mult_type, input_width):
    if mult_type == "and":
        pp = np.zeros([1, input_width * 2 - 1])
        for i in range(0, input_width):
            pp[0][i] = i + 1
        for i in range(input_width, input_width * 2 - 1):
            pp[0][i] = input_width * 2 - 1 - i
    else:
        pp = np.zeros([1, input_width * 2])
        if input_width % 2 == 0:
            max = input_width / 2 + 1
        else:
            max = input_width / 2
        j = 3
        pos1, pos2 = {}, {}
        for i in range(0, input_width + 4):
            pos1[i] = 1
        pos1[input_width + 4] = 2
        for i in range(input_width + 5, input_width * 2, 2):
            pos1[i] = j
            pos1[i + 1] = j
            if j < max:
                j = j + 1
        k = 2
        for i in range(0, input_width * 2, 2):
            pos2[i] = k
            pos2[i + 1] = k
            if k < max:
                k = k + 1
        for i in range(0, input_width * 2):
            pp[0][i] = pos2[i] - pos1[i] + 1
    return pp


def decompose_compressor_tree(input_width, mult_type, state, file_name):
    next_state = np.zeros_like(state)
    next_state[0] = state[0]
    next_state[1] = state[1]
    stage_num = 0
    if mult_type == "and":
        str_bit_width = input_width
        int_bit_width = input_width - 0.5
    else:
        str_bit_width = input_width
        int_bit_width = input_width
    ct32 = np.zeros([1, int(int_bit_width * 2)])
    ct22 = np.zeros([1, int(int_bit_width * 2)])
    ct32[0] = next_state[0]
    ct22[0] = next_state[1]
    partial_products = np.zeros([1, int(int_bit_width * 2)])
    partial_products[0] = get_initial_partial_product(mult_type, input_width)
    for i in range(1, int(int_bit_width * 2)):
        j = 0  # j denotes the stage index, i denotes the column index
        while j <= stage_num:  # the condition is impossible to satisfy
            # j-th stage i-th column
            ct32[j][i] = next_state[0][i]
            ct22[j][i] = next_state[1][i]
            # initial j-th stage partial products
            if j == 0:  # 0th stage
                partial_products[j][i] = partial_products[j][i]
            else:
                partial_products[j][i] = (
                    partial_products[j - 1][i] + ct32[j - 1][i - 1] + ct22[j - 1][i - 1]
                )
            # when to break
            if (3 * ct32[j][i] + 2 * ct22[j][i]) <= partial_products[j][i]:
                # print(f"i: {ct22[j][i]}, i-1: {ct22[j][i-1]}")
                # update j-th stage partial products for the next stage
                partial_products[j][i] = (
                    partial_products[j][i] - ct32[j][i] * 2 - ct22[j][i]
                )
                # update the next state compressors
                next_state[0][i] -= ct32[j][i]
                next_state[1][i] -= ct22[j][i]
                break  # the only exit
            else:
                if j == stage_num:
                    # print(f"j {j} stage num: {stage_num}")
                    # add initial next stage partial products and cts
                    stage_num += 1
                    # np.r_将讲个矩阵上下拼接
                    ct32 = np.r_[ct32, np.zeros([1, int(int_bit_width * 2)])]
                    ct22 = np.r_[ct22, np.zeros([1, int(int_bit_width * 2)])]
                    partial_products = np.r_[
                        partial_products, np.zeros([1, int(int_bit_width * 2)])
                    ]
                # assign 3:2 first, then assign 2:2
                # only assign the j-th stage i-th column compressors
                if ct32[j][i] >= partial_products[j][i] // 3:
                    ct32[j][i] = partial_products[j][i] // 3
                    if partial_products[j][i] % 3 == 2:
                        if ct22[j][i] >= 1:
                            ct22[j][i] = 1
                    else:
                        ct22[j][i] = 0
                else:
                    ct32[j][i] = ct32[j][i]
                    if ct22[j][i] >= (partial_products[j][i] - ct32[j][i] * 3) // 2:
                        ct22[j][i] = (partial_products[j][i] - ct32[j][i] * 3) // 2
                    else:
                        ct22[j][i] = ct22[j][i]

                # update partial products
                partial_products[j][i] = (
                    partial_products[j][i] - ct32[j][i] * 2 - ct22[j][i]
                )
                next_state[0][i] = next_state[0][i] - ct32[j][i]
                next_state[1][i] = next_state[1][i] - ct22[j][i]
            j += 1

    # 2 write the compressors information into the text file
    sum = int(ct32.sum() + ct22.sum())
    with open(file_name, mode="w") as f:
        f.write(str(str_bit_width) + " " + str(str_bit_width))
        f.write("\n")
        f.write(str(sum))
        f.write("\n")
        for i in range(0, stage_num + 1):
            for j in range(0, int(int_bit_width * 2)):
                # write 3:2 compressors
                for k in range(0, int(ct32[i][int(int_bit_width * 2) - 1 - j])):
                    f.write(str(int(int_bit_width * 2) - 1 - j))
                    f.write(" 1")
                    f.write("\n")
                for k in range(0, int(ct22[i][int(int_bit_width * 2) - 1 - j])):
                    f.write(str(int(int_bit_width * 2) - 1 - j))
                    f.write(" 0")
                    f.write("\n")
    return ct32, ct22, partial_products, stage_num


def read_npy(npy_file, output_path, input_width, mult_type):
    """
    针对我们的方法是提取每个pool下的最后一state
    """
    loaded_data = np.load(npy_file, allow_pickle=True)
    loaded_data = loaded_data.reshape(1, -1)
    for j in range(4):
        state = loaded_data[0][0][str(j) + "-th env_initial_state_pool"][-1]["state"]
        file_name = (
            output_path
            + "/ct_"
            + str(input_width)
            + "_"
            + str(mult_type)
            + "_"
            + "pool"
            + str(j)
            + ".txt"
        )
        decompose_compressor_tree(input_width, mult_type, state, file_name)


def read_pareto(file_name):
    delay = []
    area = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        mark = 0
        for line in lines:
            line = line.strip()
            if line == "area":
                mark = 1
            if mark == 1:
                if line != "area":
                    area.append(line)
            else:
                if line != "delay":
                    delay.append(line)
    pareto_point = np.zeros([len(area), 2])
    for i in range(len(area)):
        pareto_point[i][0] = area[i]
        pareto_point[i][1] = delay[i]
    pareto_point = pareto_point[pareto_point[:, 0].argsort()]
    return pareto_point


def get_hypervolume(pareto_file, mult_type, input_width):
    pareto_point = read_pareto(pareto_file)
    hv = hypervolume(pareto_point)
    hv_value = hv.compute(refer_point[mult_type][input_width])
    return hv_value


def config_abc_sta(synthesis_path, target_delay):
    # generate a config dir for each target delay
    for i in range(len(target_delay)):
        ys_path = os.path.join(synthesis_path, f"ys{i}")
        print(ys_path)
        if not os.path.exists(ys_path):
            os.mkdir(ys_path)
        abc_constr_gen(ys_path)
        sta_scripts_gen(ys_path)


def abc_constr_gen(ys_path):
    abc_constr_file = os.path.join(ys_path, "abc_constr")
    with open(abc_constr_file, "w") as f:
        f.write("set_driving_cell BUF_X1")
        f.write("\n")
        f.write("set_load 10.0 [all_outputs]")
        f.write("\n")


def sta_scripts_gen(ys_path):
    sta_file = os.path.join(ys_path, "openroad_sta.tcl")
    with open(sta_file, "w") as f:
        f.write("read_lef " + str(lef))
        f.write("\n")
        f.write("read_lib " + str(lib))
        f.write("\n")
        f.write("read_verilog ./netlist.v")
        f.write("\n")
        f.write("link_design MUL")
        f.write("\n")
        f.write("set_max_delay -from [all_inputs] 0")
        f.write("\n")
        f.write("set critical_path [lindex [find_timing_paths -sort_by_slack] 0]")
        f.write("\n")
        f.write("set path_delay [sta::format_time [[$critical_path path] arrival] 4]")
        f.write("\n")
        f.write('puts "wns $path_delay"')
        # f.write('report_wns')
        f.write("\n")
        f.write("report_design_area")
        f.write("\n")
        f.write("exit")


def ys_scripts_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, "w") as f:
        f.write("read -sv " + f"{synthesis_path}rtl/MUL.v")
        f.write("\n")
        f.write("synth -top MUL")
        f.write("\n")
        f.write("dfflibmap -liberty " + str(lib))
        f.write("\n")
        f.write("abc -D ")
        f.write(str(target_delay))
        f.write(" -constr ./abc_constr -liberty " + str(lib))
        # f.write(' -liberty ' + str(lib))
        f.write("\n")
        f.write("write_verilog ./netlist.v")


def get_ppa(ys_path):
    synthesis_cmd = (
        "cd "
        + ys_path
        + " \n"
        + f"source {OpenRoadFlowPath}/env.sh\n"
        + "yosys ./syn_with_target_delay.ys"
    )  # open road synthesis
    sta_cmd = (
        "cd "
        + ys_path
        + " \n"
        + f"source {OpenRoadFlowPath}/env.sh\n"
        + "openroad openroad_sta.tcl | tee ./log"
    )  # openroad sta
    rm_log_cmd = "rm -f " + ys_path + "/log"
    rm_netlist_cmd = "rm -f " + ys_path + "/netlist.v"

    # execute synthesis cmd
    print(synthesis_cmd)
    print(sta_cmd)
    os.system(synthesis_cmd)
    os.system(sta_cmd)
    # get ppa
    with open(ys_path + "/log", "r") as f:
        rpt = f.read().splitlines()
        for line in rpt:
            if len(line.rstrip()) < 2:
                continue
            line = line.rstrip().split()
            if line[0] == "wns":
                delay = line[-1]
                # delay = delay[1:]
                continue
            if line[0] == "Design":
                area = line[2]
                break
    ppa_dict = {"area": float(area), "delay": float(delay)}
    # remove log
    os.system(rm_log_cmd)
    os.system(rm_netlist_cmd)

    return ppa_dict


def simulate_for_ppa(target_delay, ys_path, synthesis_path):
    ys_scripts_gen(target_delay, ys_path, synthesis_path)
    ppa_dict = get_ppa(ys_path)
    ppa_dict["target_delay"] = target_delay
    return ppa_dict


def get_reward(n_processing, synthesis_path, target_delay):
    ppas_dict = {
        "area": [],
        "delay": [],
        "power": [],
        "target_delay": [],
    }

    with Pool(processes=n_processing) as pool:

        def collect_ppa(ppa_dict):
            for k in ppa_dict.keys():
                ppas_dict[k].append(ppa_dict[k])

        for i, target in enumerate(target_delay):
            ys_path = os.path.join(synthesis_path, f"ys{i}")
            pool.apply_async(
                func=simulate_for_ppa,
                args=(target, ys_path, synthesis_path),
                callback=collect_ppa,
            )
        pool.close()
        pool.join()
    return ppas_dict


def get_pareto(synthesis_path, output_path, input_width):
    """
    input:
        *synthesis_path: 综合路径，下面rtl文件夹存放对应的MUL文件,综合因为要在yosy运行，所以要绝对路径
        *output_path : 输出文件路径，会在对应文件夹下输出得到的delay_area_target求解情况，以及对应的pareto解
    """
    target_delay = []
    if input_width == 8:
        for i in range(50, 1000, 10):
            target_delay.append(i)
    elif input_width == 16:
        for i in range(50, 2000, 10):
            target_delay.append(i)
    elif input_width == 32:
        for i in range(50, 3000, 10):
            target_delay.append(i)
    elif input_width == 64:
        for i in range(50, 4000, 10):
            target_delay.append(i)
    # for file in os.listdir(synthesis_path):
    n_processing = 12
    config_abc_sta(synthesis_path + "/", target_delay)
    ppas_dict = get_reward(n_processing, synthesis_path + "/", target_delay)
    pareto = []
    if os.path.exists(output_path):
        print("dir")
    else:
        os.makedirs(output_path)
    ppa_file = output_path + "/ppa.txt"
    print(ppas_dict)
    with open(ppa_file, "w") as f:
        f.write("delay" + "\n")
        for i in range(len(target_delay)):
            f.write(str(ppas_dict["delay"][i]) + "\n")
        f.write("area" + "\n")
        for i in range(len(target_delay)):
            f.write(str(ppas_dict["area"][i]) + "\n")
        f.write("target_delay" + "\n")
        for i in range(len(target_delay)):
            f.write(str(ppas_dict["target_delay"][i]) + "\n")
    for i in range(n_processing):
        mark = 0
        for j in range(n_processing):
            if (
                ppas_dict["area"][j] <= ppas_dict["area"][i]
                and ppas_dict["delay"][j] <= ppas_dict["delay"][i]
            ):
                if (
                    ppas_dict["area"][j] == ppas_dict["area"][i]
                    and ppas_dict["delay"][j] == ppas_dict["delay"][i]
                ):
                    continue
                else:
                    mark = 1
                    break
        if mark == 0:
            if (ppas_dict["area"][i], ppas_dict["delay"][i]) not in pareto:
                pareto.append((ppas_dict["area"][i], ppas_dict["delay"][i]))
    print(pareto)
    pareto_file = output_path + "/pareto.txt"
    with open(pareto_file, "w") as f:
        f.write("delay" + "\n")
        for i in range(len(pareto)):
            f.write(str(pareto[i][1]) + "\n")
        f.write("area" + "\n")
        for i in range(len(pareto)):
            f.write(str(pareto[i][0]) + "\n")


## 输入文件形式npz
# read_npy("./npy_file/ours/8_8_and/itr_5000.npy","./testdir/",8,'and')

ct = read_ct("./ct_8_booth.txt", "booth")
write_mul("./booth_MUL.v", 8, ct)

# get_pareto("/ai4multiplier/time_analysis/test","./test/build",8)
#
# print(get_hypervolume("./test/build/pareto.txt","and",8))
