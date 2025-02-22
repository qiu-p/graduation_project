import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from o0_mul_utils import (
    legalize_compressor_tree,
    decompose_compressor_tree,
    get_initial_partial_product,
    get_default_pp_wiring,
)
import queue

from scipy.optimize import curve_fit


class Cell:
    type_name = "Cell"

    def __init__(self, name, id):
        self.name = name
        self.input_wires_id_dict = {}
        self.output_wires_id_dict = {}
        """
        port_id: wire_id
        """
        self.id = id


class HalfAdder(Cell):
    """
    __________________
    a   b   |   s   c
    ------------------
    0   0   |   0   0
    0   1   |   1   0
    1   0   |   1   0
    1   1   |   0   1
    ------------------

    1.
    s(a, b) = ab' + a'b
    s(a = 1, b) = b'
    s(a = 0, b) = b

    diff (s, a)(b) = s(a=1, b) ^ s(a=0, b) = b' ^ b = b''b + b'b' = 1
    diff (s, b)(a) = 1

    P(s) = P(a)(1 - P(b)) + (1 - P(a))P(b)
    T(s) = P(diff (s, a)(b)) T(a) + P(diff (s, b)(a)) T(b) = T(a) + T(b)

    2.
    c = ab
    c(a = 1, b) = b
    c(a = 0, b) = 0

    diff (c, a) = c(a = 1, b) ^ c(a = 0, b) = b ^ 0 = b'0 + b0' = b
    diff (c, b) = a

    P(c) = P(a)P(b)
    T(c) = P(b)T(a) + P(a)T(b)
    """

    type_name = "HalfAdder"
    power_slew = [1.6, 1.5]

    def __init__(self, name, id):
        super().__init__(name, id)

    def get_output_activity(self, input_duty, input_freq):
        p_a, p_b = input_duty
        t_a, t_b = input_freq

        p_s = p_a * (1 - p_b) + (1 - p_a) * p_b
        t_s = t_a + t_b

        p_c = p_a * p_b
        t_c = p_b * t_a + p_a * t_b

        return [p_s, p_c], [t_s, t_c]

    def get_power(self, input_freq):
        p_a, p_b = input_freq
        return self.power_slew[0] * p_a + self.power_slew[1] * p_b


class FullAdder(Cell):
    """
    Full Adder

    ____________________
    a   b   c | s   cout
    --------------------
    0   0   0 | 0   0
    0   0   1 | 1   0
    0   1   0 | 1   0
    0   1   1 | 0   1
    1   0   0 | 1   0
    1   0   1 | 0   1
    1   1   0 | 0   1
    1   1   1 | 1   1
    --------------------
    1.
    s = a ^ b ^ c
    s(a = 1, b, c) = 1 ^ b ^ c = 1 (b ^ c)' + 0xx = (b ^ c)' = bc + b'c'
    s(a = 0, b, c) = 0 ^ b ^ c = 0xx + 1 (b ^ c) = b ^ c

    diff(s, a) = 1

    P(s) = (1 - P(a))(1 - P(b))P(c) + (1 - P(a))P(b)(1 - P(c)) + P(a)(1 - P(b))(1 - P(c)) + P(a)P(b)P(c)
    T(s) = T(a) + T(b) + T(c)

    2.
    cout = a'bc + ab'c + abc' + abc
    cout(a = 1, b, c) = b + c
    cout(a = 0, b, c) = bc

    diff(cout, a) = (b + c) ^ (bc)
        = (b + c)(bc)' + (b + c)'bc
        = (b + c)(b' + c') + b'c'bc
        = bc' + b'c

    P(cout) = (1 - P(a))P(b)P(c) + P(a)(1 - P(b))P(c) + P(a)P(b)(1 - P(c)) + P(a)P(b)P(c)
    T(cout) = T(a)(P(b)(1 - P(c)) + (1 - P(b))P(c)) + T(b)(P(a)(1 - P(c)) + (1 - P(a))P(c)) + T(c)(P(b)(1 - P(a)) + (1 - P(b))P(a))
    """

    type_name = "FullAdder"
    power_slew = [2.05, 3.59, 3.72]
    port_index_map = {
        "a": 0,
        "b": 1,
        "c": 2,
        "cin": 2,  # cin 和 c 是一个东西
        "s": 3,
        "cout": 4,
    }
    io_type = {
        0: "input",
        1: "input",
        2: "input",
        3: "output",
        4: "output",
    }

    def __init__(self, name, id):
        super().__init__(name, id)

    def get_output_activity(self, input_duty, input_freq):
        p_a, p_b, p_c = input_duty
        t_a, t_b, t_c = input_freq

        p_s = (
            (1 - p_a) * (1 - p_b) * p_c
            + (1 - p_a) * p_b * (1 - p_c)
            + p_a * (1 - p_b) * (1 - p_c)
            + p_a * p_b * p_c
        )
        t_s = t_a + t_b + t_c

        p_cout = (
            (1 - p_a) * p_b * p_c
            + p_a * (1 - p_b) * p_c
            + p_a * p_b * (1 - p_c)
            + p_a * p_b * p_c
        )
        t_cout = (
            t_a * (p_b * (1 - p_c) + (1 - p_b) * p_c)
            + t_b * (p_a * (1 - p_c) + (1 - p_a) * p_c)
            + t_c * (p_b * (1 - p_a) + (1 - p_b) * p_a)
        )

        return [p_s, p_cout], [t_s, t_cout]

    def get_power(self, input_freq):
        p_a, p_b, p_c = input_freq
        return (
            self.power_slew[0] * p_a
            + self.power_slew[1] * p_b
            + self.power_slew[2] * p_c
        )


class Wire:
    def __init__(self, name: str, wire_type: str, id: int):
        """
        name: 就是线的名字
        wire_type: input or output or inner or propagate
        propagate 表示这个线连到了别的线上，所以会有 connected_id
        """
        self.duty = None
        self.freq = None
        self.name = name
        self.wire_type = wire_type
        self.id = id

        self.connected_id = None


class NetList:
    def __init__(self):
        self.cell_id_dict = {}
        self.wire_id_dict = {}
        """
        xxx_id_dict = {
            id: 实例
        }
        """

        self.cell_name_dict = {}
        self.wire_name_dict = {}
        """
        xxx_name_dict = {
            name: id
        }
        """

        self.id_counter = 0
        self.input_wire_list = []

        self.ct32_power = 1.635e-05
        self.ct22_power = 9.21e-06
        self.other_power = -2.26e-05

    def register_id(self):
        id_num = self.id_counter
        self.id_counter += 1

        if id_num == 74:
            pass
        return id_num

    def load_netlist_from_ct(
        self, bit_width, encode_type, ct: np.ndarray, pp_wiring: np.ndarray = None
    ):
        """
        从 ct 中构建 网表
        需要注意的是

        对于stage s 上的 压缩器，输入的是 data_{s} 输出的是 data_{s+1}
        """
        # fmt: off
        max_stage_num = 2 * bit_width
        pp = get_initial_partial_product(bit_width, encode_type)
        ct32, ct22, _, __ = decompose_compressor_tree(pp, ct[0], ct[1])
        stage_num = len(ct32)
        if pp_wiring is None:
            pp_wiring = get_default_pp_wiring(max_stage_num, pp, ct)
        
        remain_pp = copy.deepcopy(pp)

        # 处理输入
        for column_index in range(len(pp)):
            for row_index in range(int(pp[column_index])):
                id_num = self.register_id()
                name = f"data{column_index}_s{0}[{row_index}]"
                wire = Wire(name, "input", id_num)
                self.input_wire_list.append(id_num)
                self.wire_id_dict[id_num] = wire
                self.wire_name_dict[name] = id_num

        # 处理中间的内容
        for stage_index in range(stage_num):
            for column_index in range(len(pp)):
                wire_num = int(remain_pp[column_index])
                out_wire_counter = 0
                wire_index = 0
                slice_pp_wiring = pp_wiring[stage_index][column_index]

                # offset 关系到 pp 的组织方式, 详情见 readme
                if column_index + 1 < len(pp):
                    cout_pp_index_offset = remain_pp[column_index + 1] - 2 * ct32[stage_index][column_index + 1] - ct22[stage_index][column_index + 1]
                else:
                    cout_pp_index_offset = 0
                cout_pp_index_offset = int(cout_pp_index_offset)
                # 3:2 压缩器
                for ct_index in range(int(ct32[stage_index][column_index])):
                    # 创建 FA 实例
                    fa_id = self.register_id()
                    fa_name = f"FA_s{stage_index}_c{column_index}_i{ct_index}"
                    fa = FullAdder(fa_name, fa_id)
                    # 连接输入的 wire
                    wire_1_name = f"data{column_index}_s{stage_index}[{slice_pp_wiring[wire_index]}]"
                    wire_2_name = f"data{column_index}_s{stage_index}[{slice_pp_wiring[wire_index + 1]}]"
                    wire_3_name = f"data{column_index}_s{stage_index}[{slice_pp_wiring[wire_index + 2]}]"

                    fa.input_wires_id_dict[0] = self.wire_name_dict[wire_1_name]
                    fa.input_wires_id_dict[1] = self.wire_name_dict[wire_2_name]
                    fa.input_wires_id_dict[2] = self.wire_name_dict[wire_3_name]

                    # 创建并且连接输出的 wire
                    ## s
                    wire_s_id = self.register_id()
                    wire_s_name = f"data{column_index}_s{stage_index+1}[{out_wire_counter}]"
                    wire_s = Wire(wire_s_name, "inner", wire_s_id)
                    self.wire_name_dict[wire_s_name] = wire_s_id
                    self.wire_id_dict[wire_s_id] = wire_s
                    fa.output_wires_id_dict[3] = wire_s_id
                    ## cout
                    wire_cout_id = self.register_id()

                    cout_pp_index = cout_pp_index_offset + out_wire_counter

                    wire_cout_name = f"data{column_index+1}_s{stage_index+1}[{cout_pp_index}]"
                    wire_cout = Wire(wire_cout_name, "inner", wire_cout_id)
                    self.wire_name_dict[wire_cout_name] = wire_cout_id
                    self.wire_id_dict[wire_cout_id] = wire_cout
                    fa.output_wires_id_dict[4] = wire_cout_id

                    # 注册 fa 到 netlist 中
                    self.cell_id_dict[fa_id] = fa
                    self.cell_name_dict[fa_name] = fa_id

                    wire_index += 3
                    out_wire_counter += 1

                # 2:2 压缩器
                for ct_index in range(int(ct22[stage_index][column_index])):
                    # 创建 HA 实例
                    ha_id = self.register_id()
                    ha_name = f"HA_s{stage_index}_c{column_index}_i{ct_index}"
                    ha = HalfAdder(ha_name, ha_id)
                    # 连接输入的 wire
                    wire_1_name = f"data{column_index}_s{stage_index}[{slice_pp_wiring[wire_index]}]"
                    wire_2_name = f"data{column_index}_s{stage_index}[{slice_pp_wiring[wire_index + 1]}]"

                    ha.input_wires_id_dict[0] = self.wire_name_dict[wire_1_name]
                    ha.input_wires_id_dict[1] = self.wire_name_dict[wire_2_name]

                    # 创建并且连接输出的 wire
                    ## s
                    wire_s_id = self.register_id()
                    wire_s_name = f"data{column_index}_s{stage_index+1}[{out_wire_counter}]"
                    wire_s = Wire(wire_s_name, "inner", wire_s_id)
                    self.wire_name_dict[wire_s_name] = wire_s_id
                    self.wire_id_dict[wire_s_id] = wire_s
                    ha.output_wires_id_dict[2] = wire_s_id
                    ## cout
                    wire_cout_id = self.register_id()

                    cout_pp_index = cout_pp_index_offset + out_wire_counter

                    wire_cout_name = f"data{column_index+1}_s{stage_index+1}[{cout_pp_index}]"
                    wire_cout = Wire(wire_cout_name, "inner", wire_cout_id)
                    self.wire_name_dict[wire_cout_name] = wire_cout_id
                    self.wire_id_dict[wire_cout_id] = wire_cout
                    ha.output_wires_id_dict[3] = wire_cout_id

                    # 注册 fa 到 netlist 中
                    self.cell_id_dict[ha_id] = ha
                    self.cell_name_dict[ha_name] = ha_id

                    wire_index += 2
                    out_wire_counter += 1

                # 顺延到下一个阶段
                for wire_index in range(int(3 * ct32[stage_index][column_index] + 2 * ct22[stage_index][column_index]), wire_num):
                    wire_id = self.register_id()
                    wire_name = f"data{column_index}_s{stage_index+1}[{out_wire_counter}]"
                    connected_name = f"data{column_index}_s{stage_index}[{wire_index}]"
                    connected_id = self.wire_name_dict[connected_name]

                    wire = Wire(wire_name, "propagate", wire_id)
                    wire.connected_id = connected_id

                    self.wire_name_dict[wire_name] = wire_id
                    self.wire_id_dict[wire_id] = wire

                    out_wire_counter += 1

                # update remain pp
                remain_pp[column_index] += (
                    -2 * ct32[stage_index][column_index]
                    - ct22[stage_index][column_index]
                )
                if column_index > 0:
                    remain_pp[column_index] += (
                        ct32[stage_index][column_index - 1]
                        + ct22[stage_index][column_index - 1]
                    )

    def get_activity(self, wire_id):
        src_id = wire_id
        while self.wire_id_dict[src_id].wire_type == "propagate":
            src_id = self.wire_id_dict[src_id].connected_id
        return self.wire_id_dict[src_id].duty, self.wire_id_dict[src_id].freq

    """
    假设
    - 输入的 a 和 b 是 p=0.5 的二项分布
    - 前后两个时刻 a/b 的取值无关

    则 P(a) = 0.5, T(a) = 0.5

    1. And 编码:
    pp(a, b) = ab
    pp(a=1, b) = b
    pp(a = 0, b) = 0
    diff(pp, a)(b) = b ^ 0 = b0' + b'0 = b

    P(pp) = 0.25
    T(pp) = P(b)T(a) + P(a)T(b) = 0.5

    2. Booth 编码

    x[0], x[1], x[2] => P = T = 0.5
    y => P = T = 0.5

    s = x0 x1
    d = x0 x1 x2' + x0' x1' x2
    n = x2
    z = s y[i] + d y[i-1]
    pp[i] = n ^ z

    1. 
    P(s) = 0.25
    T(s) = 0.5

    2. 
    d(x_0 = 0) = x1' x2
    d(x_0 = 1) = x1 x2'
    diff(d, x_0) = (x1' x2)'x_1 x_2' + x1' x2(x1 x2')'
        = (x1 + x2')x_1 x_2' + x1' x2(x1' + x2)
        = x1 x2' + x1'x2
    diff(d, x1) = x0 x2' + x0' x2

    d(x2 = 0) = x0x1
    d(x2 = 1) = x0'x1'
    diff(d, x2) = x0x1 + x0'x1'

    P(d) = 0.25
    T(d) = 0.5 * 0.25 + 0.5 * 0.25 + 0.5 * 0.5 = 0.5

    3.
    p := d y[i-1]
    z(s = 0) = p
    z(s = 1) = y[i] + p

    diff(z, s) = p' (y[i] + p) + p y[i]' p'
        = p' y[i] = (d' + y[i-1]')y[i]
    
    diff(z, s) = (d' + y[i-1]')y[i]
    diff(z, d) = (s' + y[i]')y[i-1]
    diff(z, y[i]) = (d' + y[i-1]')s
    diff(z, y[i-1]) = (s' + y[i]')d

    P(z) = 0.25 * 0.5 + 0.25 * 0.5 = 0.25
    T(z) = 0.5 * (0.75 + 0.5) * 0.5 * 2
        + 0.5 * (0.75 + 0.5) * 0.25 * 2
        = 0.9375
    4. 
    pp[i] = n ^ z = n'z + z'n
    P(pp[i]) = 0.5 * 0.25 + 0.5 * 0.75 = 0.5
    T(pp[i]) = T(n) + T(z) = 0.9375 + 0.5 = 1.4375

    """

    def report_power(self, bit_width, encode_type, ct, pp_wiring=None):
        self.load_netlist_from_ct(bit_width, encode_type, ct, pp_wiring)
        power = 0.0

        # 首先处理输入
        if encode_type == "and":
            for input_id in self.input_wire_list:
                self.wire_id_dict[input_id].duty = 0.25
                self.wire_id_dict[input_id].freq = 0.25
        elif encode_type == "booth":
            for input_id in self.input_wire_list:
                self.wire_id_dict[input_id].duty = 0.5
                self.wire_id_dict[input_id].freq = 1.4375

        pp = get_initial_partial_product(bit_width, encode_type)
        ct32, ct22, _, __ = decompose_compressor_tree(pp, ct[0], ct[1])
        stage_num = len(ct32)
        for stage_index in range(stage_num):
            for column_index in range(len(pp)):
                # 计算power
                for ct_index in range(int(ct32[stage_index][column_index])):
                    # FA power
                    cell_name = f"FA_s{stage_index}_c{column_index}_i{ct_index}"
                    fa_id = self.cell_name_dict[cell_name]
                    fa: FullAdder = self.cell_id_dict[fa_id]

                    input_duty = [None, None, None]
                    input_freq = [None, None, None]
                    for input_index, input_key in enumerate(
                        fa.input_wires_id_dict.keys()
                    ):
                        wire_id = fa.input_wires_id_dict[input_key]
                        input_duty[input_index], input_freq[input_index] = (
                            self.get_activity(wire_id)
                        )
                    power += fa.get_power(input_freq)

                    # 计算输出的 activity
                    output_duty, output_freq = fa.get_output_activity(
                        input_duty, input_freq
                    )

                    for out_port_index, out_port_key in enumerate(
                        fa.output_wires_id_dict.keys()
                    ):
                        out_port_id = fa.output_wires_id_dict[out_port_key]
                        self.wire_id_dict[out_port_id].duty = output_duty[
                            out_port_index
                        ]
                        self.wire_id_dict[out_port_id].freq = output_freq[
                            out_port_index
                        ]

                for ct_index in range(int(ct22[stage_index][column_index])):
                    # HA power
                    cell_name = f"HA_s{stage_index}_c{column_index}_i{ct_index}"
                    ha_id = self.cell_name_dict[cell_name]
                    ha: HalfAdder = self.cell_id_dict[ha_id]

                    input_duty = [None, None]
                    input_freq = [None, None]
                    for input_index, input_key in enumerate(
                        ha.input_wires_id_dict.keys()
                    ):
                        wire_id = ha.input_wires_id_dict[input_key]
                        input_duty[input_index], input_freq[input_index] = (
                            self.get_activity(wire_id)
                        )
                    power += ha.get_power(input_freq)

                    # 计算输出的 activity
                    output_duty, output_freq = ha.get_output_activity(
                        input_duty, input_freq
                    )

                    for out_port_index, out_port_key in enumerate(
                        ha.output_wires_id_dict.keys()
                    ):
                        out_port_id = ha.output_wires_id_dict[out_port_key]
                        self.wire_id_dict[out_port_id].duty = output_duty[
                            out_port_index
                        ]
                        self.wire_id_dict[out_port_id].freq = output_freq[
                            out_port_index
                        ]

        return power

    def set_paramter(self, ct32_power, ct22_power, other_power):
        self.ct32_power = ct32_power
        self.ct22_power = ct22_power
        self.other_power = other_power

    @staticmethod
    def __report_power_v2_kernal(x, a, b, c):
        num_ct32 = x[:, 0]
        num_ct22 = x[:, 1]

        return a * num_ct32 + b * num_ct22 + c

    def fit_report_power_v2(self, x, y):
        params, covariance = curve_fit(self.__report_power_v2_kernal, x, y, p0=[1, 1, 1])
        self.ct32_power, self.ct22_power, self.other_power = params
        return params

    def report_power_v2(self, ct):
        num_ct32 = np.sum(ct[0])
        num_ct22 = np.sum(ct[1])

        return num_ct32 * self.ct32_power + num_ct22 * self.ct22_power + self.other_power

    def emit_verilog(self):
        pass


if __name__ == "__main__":
    from o0_mul_utils import get_compressor_tree

    # np.random.seed(0)
    bit_width = 8
    encode_type = "and"
    pp = get_initial_partial_product(bit_width, encode_type)

    ct = get_compressor_tree(pp, bit_width, "dadda")

    netlist = NetList()
    pp_wiring = get_default_pp_wiring(2 * bit_width, pp, ct, "random")
    power = netlist.report_power(bit_width, encode_type, ct, pp_wiring)
    print(power)

    wire_duty = []
    for key in netlist.wire_id_dict.keys():
        wire: Wire = netlist.wire_id_dict[key]
        if wire.wire_type != "propagate":
            print(wire.name, wire.freq)
            wire_duty.append(wire.freq)

    plt.hist(wire_duty)
    plt.show()
