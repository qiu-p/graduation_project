# 主要是两部分，一个是部分积生成，一个是压缩树阶段
# 1. 确认输入，输入端口，以及阶段数
# 2. 根据产生部分积，赋予每列
import numpy as np
import os

def write_production(width,num):
    """
    input:
        * width:乘法器位宽
    return:
        * pp_str : and_production字符串
    """

    width = width
    # 输入输出端口
    pp_str = "module production("
    for i in range(num):
        if i!=num-1:
            pp_str += "adder_{},".format(i)
        else:
            pp_str += "adder_{}".format(i)

    for i in range(width):
        pp_str += ",out" + str(i)
    pp_str += ");\n"

    # 位宽
    for i in range(num):
        pp_str += "\tinput[{}:0] adder_{};\n".format(width - 1,i)
    
    for i in range(width):
        pp_str += "\toutput[{}:0] out{};\n".format(num-1, i )

    # 赋值,out0代表高位
    for i in range(width):
        for j in range(num):
              pp_str += "\tassign out{}[{}] = adder_{}[{}];\n".format(i, j, j , width- i-1)

    pp_str += "endmodule\n"
    return pp_str


def update_remain_pp(ct, stage, final_stage_pp):
    ct32 = ct[0][stage][:]
    ct22 = ct[1][stage][:]

    width = ct.shape[2]
    initial_state = np.zeros((width))

    for i in range(width):
        if i == width - 1:
            initial_state[i] = final_stage_pp[i] - ct32[i] - ct22[i]
        else:
            initial_state[i] = (
                final_stage_pp[i] - ct32[i] - ct22[i] - ct32[i + 1] - ct22[i + 1]
            )
    initial_state = initial_state.astype(int)
    return initial_state


def update_final_pp(ct,stage,num,width):
    ct32 = np.sum(ct[0][: stage + 1][:], axis=0)
    ct22 = np.sum(ct[1][: stage + 1][:], axis=0)
    initial_state = np.full(width,num)
    for i in range(width):
        if i == width - 1:
            initial_state[i] = initial_state[i] - 2 * ct32[i] - ct22[i]
        else:
            initial_state[i] = (
                initial_state[i] - 2 * ct32[i] - ct22[i] + ct32[i + 1] + ct22[i + 1]
            )
    initial_state = initial_state.astype(int)
    #print(initial_state)
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


def write_CT(num,ct=[],):
    """
    input:
        *乘法器位宽
        *ct: 压缩器信息，shape为2*stage*width
    """
    stage, width = ct.shape[1], ct.shape[2]

    # 输入输出端口
    ct_str = "module Compressor_Tree("
    for i in range(num):
        if i!=num-1:
            ct_str += "adder_{},".format(i)
        else:
            ct_str += "adder_{}".format(i)

    for i in range(width):
        ct_str += ",data{}_s{}".format(i, stage)
    ct_str += ");\n"

    # 位宽
    for i in range(num):
        ct_str += "\tinput[{}:0] adder_{};\n".format(width - 1,i)

    final_state = update_final_pp(ct,stage,num,width)
    initial_state = np.full(width,num)
    # print("final",final_state)

    # TODO: 根据每列最终的部分积确定最终的输出位宽
    for i in range(width):
        ct_str += "\toutput[{}:0] data{}_s{};\n".format(
            int(final_state[i]) - 1, i, stage
        )

    # 调用production模块，产生部分积
    ct_str += "\n\t//pre-processing block : production\n"
    for i in range(width):
        ct_str += "\twire[{}:0] out{};\n".format(int(initial_state[i]) - 1, i)

    ct_str += "\tproduction PD0("
    for i in range(num):
        if i!=num-1:
            ct_str += ".adder_{}(adder_{}),".format(i,i)
        else:
            ct_str += ".adder_{}(adder_{})".format(i,i)
    
    for i in range(width):
        ct_str += ",.out{}(out{})".format(i, i)
    ct_str += ");"
    FA_num = 0
    HA_num = 0

    # 生成每个阶段的压缩树
    for stage_num in range(stage):
        ct_str += "\n\t//****The {}th stage****\n".format(stage_num + 1)
        
        final_stage_pp = update_final_pp(ct, stage_num,num,width)
        #print(final_stage_pp)
        #print("111:",final_stage_pp)
        remain_pp = update_remain_pp(ct, stage_num, final_stage_pp)

        for i in range(width):
            ct_str += "\twire[{}:0] data{}_s{};\n".format(
                final_stage_pp[i] - 1, i, stage_num + 1
            )

        num_tmp = 0
        for j in range(width):
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


def read_ct(ct_file):
    with open(ct_file, "r") as f:
        lines = f.readlines()
    width = int(lines[0].strip().split(" ")[1])
    num =  int(lines[0].strip().split(" ")[0])
    
    stage = 0
    pre_idx = 10000

    ct = np.zeros((2, 1, width) )
    for i in range(2, len(lines)):
        line = lines[i].strip().split(" ")
        idx, kind = int(line[0]), int(line[1])
        if idx > pre_idx:
            stage += 1
            news = np.zeros((2, 1, width) )
            ct = np.concatenate((ct, news), axis=1)
            #print(ct.shape)
        pre_idx = idx
        if kind == 1:
            ct[0][stage][idx] += 1
        else:
            ct[1][stage][idx] += 1
    return num,width,ct


def write_adder(mul_verilog_file, width,ct,num):
    """
    input:
        * mul_verilog_file: 输出verilog路径
        *width: 输入电路的位宽
        *ct: 输入电路的压缩树，shape为2*stage_num*width
    """
    ct = ct.astype(int)[:, :, ::-1]
    width = width
    with open(mul_verilog_file, "w") as f:
        f.write(write_FA())
        f.write(write_HA())
        f.write(write_production(width,num))

        f.write(write_CT(num,ct))
        f.write("module Adder(")
        for i in range(num):
            f.write("adder_{},".format(i))
        f.write("clock,out);\n")
        f.write("\tinput clock;\n")
        for i in range(num):
            f.write("\tinput[{}:0] adder_{};\n".format(width - 1,i))

        f.write("\toutput[{}:0] out;\n".format(width-1))
        stage = ct.shape[1]
        final_pp = update_final_pp(ct, stage,num,width)

        for i in range(len(final_pp)):
            f.write("\twire[{}:0] out{}_C;\n".format(final_pp[i] - 1, i))

        f.write("\tCompressor_Tree C0(")
        for i in range(num):
            if i!=num-1:
                f.write(".adder_{}(adder_{}),".format(i,i))
            else:
                f.write(".adder_{}(adder_{})".format(i,i))
        for i in range(len(final_pp)):
            f.write(",.data{}_s{}(out{}_C)".format(i, stage, i))
        f.write(");\n")

        f.write("\twire[{}:0] addend;\n".format(width - 1))
        f.write("\twire[{}:0] augned;\n".format(width - 1))

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
        f.write("\twire[{}:0] tmp = addend + augned;\n".format( width-1))
        f.write("\tassign out = tmp[{}:0];\n".format(width-1))
        f.write("endmodule\n")

# num,width,ct = read_ct("./test.txt")
# #print(ct)
# write_adder("./test.v",width, ct,num)

# get_pareto("/ai4multiplier/time_analysis/test","./test/build",8)
#
# print(get_hypervolume("./test/build/pareto.txt","and",8))
