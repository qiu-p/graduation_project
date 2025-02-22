import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import json


def cell_map_legalize(cell_map):
    input_bit = len(cell_map)
    for x in range(input_bit):
        cell_map[x, x] = 1
        cell_map[x, 0] = 1
    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                cell_map[last_y - 1, y] = 1
                last_y = y
    return cell_map


def get_default_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit))
    for i in range(input_bit):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1
    return np.array(cell_map)


def get_brent_kung_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit))
    for i in range(input_bit):
        cell_map[i, i] = 1
        cell_map[i, 0] = 1
    t = 2
    while t < input_bit:
        for i in range(t - 1, input_bit, t):
            cell_map[i, i - t + 1] = 1
        t *= 2

    return np.array(cell_map)


def get_sklansky_init(input_bit):
    cell_map = np.zeros((input_bit, input_bit))
    for i in range(input_bit):
        cell_map[i, i] = 1
        t = i
        now = i
        x = 1
        level = 1
        while t > 0:
            if t % 2 == 1:
                last_now = now
                now -= x
                cell_map[i, now] = 1
                level += 1
            t = t // 2
            x *= 2
    cell_map = cell_map_legalize(cell_map)
    return np.array(cell_map)


def get_kogge_stone_init(input_bit):
    """
    生成 Kogge-Stone 加法器的进位树结构
    """

    cell_map = np.zeros((input_bit, input_bit))

    for i in range(input_bit):
        cell_map[i, i] = 1  # 自带输入
        cell_map[i, 0] = 1  # 自带输入
        j = 1
        while j < i:
            j *= 2
            cell_map[i, i - (j - 1)] = 1

    cell_map = cell_map_legalize(cell_map)
    return np.array(cell_map)


# GPT 生成的 正确性存疑
def get_han_carlson_init(input_bit):
    """
    生成 Han-Carlson 加法器的进位树结构
    """
    cell_map = np.zeros((input_bit, input_bit))
    for i in range(input_bit):
        cell_map[i, i] = 1  # 自带输入
        cell_map[i, 0] = 1  # 自带输入

    t = 1
    while t < input_bit:
        for i in range(t, input_bit, t * 2):
            cell_map[i, i - t] = 1  # 长跳跃连接
        t *= 2

    t = 2
    while t < input_bit:
        for i in range(t - 1, input_bit, t * 2):
            cell_map[i, i - t + 1] = 1  # 短跳跃连接
        t *= 2

    cell_map = cell_map_legalize(cell_map)
    return np.array(cell_map)


def get_init_cell_map(input_bit: int, init_type: str):
    if init_type == "default":
        return get_default_init(input_bit)
    elif init_type == "brent_kung":
        return get_brent_kung_init(input_bit)
    elif init_type == "sklansky":
        return get_sklansky_init(input_bit)
    elif init_type == "kogge_stone":
        return get_kogge_stone_init(input_bit)
    elif init_type == "han_carlson":
        return get_han_carlson_init(input_bit)
    else:
        raise NotImplementedError


BLACK_CELL = """module BLACK(gik, pik, gkj, pkj, gij, pij);
    input gik, pik, gkj, pkj;
    output gij, pij;
    assign pij = pik & pkj;
    assign gij = gik | (pik & gkj);
endmodule
"""

GREY_CELL = """module GREY(gik, pik, gkj, gij);
    input gik, pik, gkj;
    output gij;
    assign gij = gik | (pik & gkj);
endmodule
"""

# 改进后的 cell

BLACK_CELL_00 = """module BLACK_CELL_00(pik, pkj, pij);
    input pik, pkj;
    output pij;
    assign pij = pik & pkj;
endmodule
"""

BLACK_CELL_01 = """module BLACK_CELL_01(pik, pkj, gkj, gij, pij);
    input pik, pkj, gkj;
    output gij, pij;
    assign pij = pik & pkj;
    assign gij = pik & gkj;
endmodule
"""

BLACK_CELL_10 = """module BLACK_CELL_10(gik, pik, pkj, gij, pij);
    input gik, pik, pkj;
    output gij;
    output pij;
    assign gij = gik;
    assign pij = pik & pkj;
endmodule
"""

BLACK_CELL_11 = """module BLACK_CELL_11(gik, pik, gkj, pkj, gij, pij);
    input gik, pik, gkj, pkj;
    output gij, pij;
    assign pij = pik & pkj;
    assign gij = gik | (pik & gkj);
endmodule
"""

# grey cell 只会输出 gi0, 不输出 p
# 因为 s[i] = gi0 ^ pii
GREY_CELL_11 = """module GREY_CELL_11(gik, pik, gkj, gij);
    input gik, pik, gkj;
    output gij;
    assign gij = gik | (pik & gkj);
endmodule
"""

GREY_CELL_00 = """module GREY_CELL_00(gij);
    output gij;
    assign gij = 0;
endmodule
"""

GREY_CELL_01 = """module GREY_CELL_01(pik, gkj, gij);
    input pik, gkj;
    output gij;
    assign gij = pik & gkj;
endmodule
"""

GREY_CELL_10 = """module GREY_CELL_10(gik, gij);
    input gik;
    output gij;
    assign gij = gik;
endmodule
"""


def adder_output_verilog_top(cell_map: np.ndarray) -> str:
    input_bit = len(cell_map)
    content = ""

    content += f"module PrefixAdder(a,b,s,cout);\n"
    content += f"\tinput [{input_bit - 1}:0] a,b;\n"
    content += f"\toutput [{input_bit - 1}:0] s;\n"
    content += "\toutput cout;\n"
    wires = set()
    for i in range(input_bit):
        wires.add(f"c{i}")

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                else:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                    wires.add(f"p{last_y - 1}_{y}")
                    wires.add(f"g{x}_{y}")
                    wires.add(f"p{x}_{y}")
                last_y = y

    for i in range(input_bit):
        wires.add(f"p{i}_{i}")
        wires.add(f"g{i}_{i}")
        wires.add(f"g{i}_{0}")
        wires.add(f"c{x}")
    assert 0 not in wires
    assert "0" not in wires
    content += "\twire "

    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"\tassign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"\tassign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"\tassign g{i}_0 = c{i};\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:  # add grey module
                    content += f"\tGREY cell_{x}_{y}_grey(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, c{x});\n"
                else:
                    content += f"\tBLACK cell_{x}_{y}_black(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, p{last_y - 1}_{y}, g{x}_{y}, p{x}_{y});\n"
                last_y = y

    content += "\tassign s[0] = a[0] ^ b[0];\n"
    content += "\tassign c0 = g0_0;\n"
    content += f"\tassign cout = c{input_bit - 1};\n"
    for i in range(1, input_bit):
        content += f"\tassign s[{i}] = p{i}_{i} ^ c{i - 1};\n"
    content += "endmodule"
    content += "\n\n"

    return content


def adder_output_verilog_all(cell_map: np.ndarray, remain_pp: np.ndarray = None):
    if remain_pp is None:
        return BLACK_CELL + GREY_CELL + adder_output_verilog_top(cell_map)
    else:
        return (
            BLACK_CELL + GREY_CELL + adder_output_verilog_from_ct(cell_map, remain_pp)
        )


def adder_output_verilog_from_ct(cell_map: np.ndarray, final_pp: np.ndarray) -> str:
    input_bit = len(cell_map)
    content = ""

    # module head
    content += f"module PrefixAdder("
    for column_index in range(len(final_pp)):
        content += f"out{column_index}_C, "
    content += f"s,cout,clock);\n"
    # 声明
    for column_index in range(len(final_pp)):
        content += f"\t input[{final_pp[column_index] - 1}:0] out{column_index}_C;\n"
    content += "\toutput cout;\n"
    content += "\tinput clock;\n"
    content += f"\toutput[{len(final_pp) - 1}:0] s;\n\n"
    content += f"\twire[{len(final_pp) - 1}:0] a;\n"
    content += f"\twire[{len(final_pp) - 1}:0] b;\n"
    for column_index in range(len(final_pp)):
        content += f"\t assign a[{len(final_pp)- 1 - column_index}] = out{column_index}_C[0];\n"
        if final_pp[column_index] == 1:
            content += f"\t assign b[{len(final_pp)- 1 - column_index}] = 1'b0;\n"
        else:
            content += f"\t assign b[{len(final_pp)- 1 - column_index}] = out{column_index}_C[1];\n"

    wires = set()
    for i in range(input_bit):
        wires.add(f"c{i}")

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                else:
                    wires.add(f"g{x}_{last_y}")
                    wires.add(f"p{x}_{last_y}")
                    wires.add(f"g{last_y - 1}_{y}")
                    wires.add(f"p{last_y - 1}_{y}")
                    wires.add(f"g{x}_{y}")
                    wires.add(f"p{x}_{y}")
                last_y = y

    for i in range(input_bit):
        wires.add(f"p{i}_{i}")
        wires.add(f"g{i}_{i}")
        wires.add(f"g{i}_{0}")
        wires.add(f"c{x}")
    assert 0 not in wires
    assert "0" not in wires
    content += "\twire "

    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"\tassign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"\tassign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"\tassign g{i}_0 = c{i};\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:  # add grey module
                    content += f"\tGREY cell_{x}_{y}_grey(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, c{x});\n"
                else:
                    content += f"\tBLACK cell_{x}_{y}_black(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, p{last_y - 1}_{y}, g{x}_{y}, p{x}_{y});\n"
                last_y = y

    content += "\tassign s[0] = a[0] ^ b[0];\n"
    content += "\tassign c0 = g0_0;\n"
    content += f"\tassign cout = c{input_bit - 1};\n"
    for i in range(1, input_bit):
        content += f"\tassign s[{i}] = p{i}_{i} ^ c{i - 1};\n"
    content += "endmodule"
    content += "\n\n"

    return content


def get_cell_type_map(cell_map: np.ndarray, final_pp: np.ndarray) -> list:
    """
    根据输入的不同 有四种cell
    b1 \
        cell -- bo
    b2 /

    (b1 b2) = 00, 01, 10, 11

    00 的输出是 0
    其余的输出都是 1

    b 是针对组进位产生信号 g 而言的
    0 代表没有 g, 1代表 ...
    ##################################

    就输入而言, cell (ii) 其实是不存在的
    那怎么确定呢第一批 cell 的颜色呢

    单独考虑. 如果有父节点中是 cell (ii) 的, 就考察 final_pp

    ##################################

    因此引入两个概念
    cell_type_map: 这个 cell 的类型是什么 00 or 01, or ...
    cell_out_map: 这个 cell 的输出类型是什么, 0 or 1

    ##################################

    
    从功能上看 每个cell 的功能是

    ik \
        cell -- ij
    kj /

    x --> i     y --> j     k --> last_y - 1
    整棵树的输出是 i0
    整棵树的输入是 ii

    所以对于 cell[x, y], 左父节点 cell[x, last_y], 右父节点 cell[last_y - 1, y]
    """
    input_bit = len(cell_map)

    cell_type_map = np.full_like(cell_map, "", dtype=str).tolist()
    cell_out_map = np.full_like(cell_map, 0, dtype=int)

    for i in range(len(final_pp)):
        cell_out_map[len(final_pp) - 1 - i, len(final_pp) - 1 - i] = final_pp[i] - 1

    for x in range(input_bit):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                # 输入
                if cell_out_map[x, last_y] == 0 and cell_out_map[last_y - 1, y] == 0:
                    # 00 -> 0
                    cell_out_map[x, y] = 0
                else:
                    cell_out_map[x, y] = 1
                cell_type_map[x][
                    y
                ] = f"{cell_out_map[x, last_y]}{cell_out_map[last_y - 1, y]}"

                last_y = y

    return cell_type_map


def adder_output_verilog_from_ct_v1(cell_map: np.ndarray, final_pp: np.ndarray) -> str:

    input_bit = len(cell_map)
    content = BLACK_CELL_00 + BLACK_CELL_01 + BLACK_CELL_10 + BLACK_CELL_11 + "\n"
    content += GREY_CELL_00 + GREY_CELL_01 + GREY_CELL_10 + GREY_CELL_11 + "\n"

    cell_type_map = get_cell_type_map(cell_map, final_pp)

    # module head
    content += f"module PrefixAdder("
    for column_index in range(len(final_pp)):
        content += f"out{column_index}_C, "
    content += f"s,cout, clock);\n"
    # 声明
    for column_index in range(len(final_pp)):
        content += f"\t input[{final_pp[column_index] - 1}:0] out{column_index}_C;\n"
    content += "\tinput clock;\n"
    content += "\toutput cout;\n"
    content += f"\toutput[{len(final_pp) - 1}:0] s;\n\n"

    # 创建 wire
    content += f"\twire[{len(final_pp) - 1}:0] a;\n"
    content += f"\twire[{len(final_pp) - 1}:0] b;\n"
    for column_index in range(len(final_pp)):
        content += f"\t assign a[{len(final_pp)- 1 - column_index}] = out{column_index}_C[0];\n"
        if final_pp[column_index] == 1:
            content += f"\t assign b[{len(final_pp)- 1 - column_index}] = 1'b0;\n"
        else:
            content += f"\t assign b[{len(final_pp)- 1 - column_index}] = out{column_index}_C[1];\n"

    wires = set()
    for i in range(input_bit):
        wires.add(f"c{i}")

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:
                    # GREY CELL
                    if cell_type_map[x][y] == "11":
                        wires.add(f"g{x}_{last_y}")
                        wires.add(f"p{x}_{last_y}")
                        wires.add(f"g{last_y - 1}_{y}")
                    elif cell_type_map[x][y] == "00":
                        pass
                    elif cell_type_map[x][y] == "01":
                        wires.add(f"p{x}_{last_y}")
                        wires.add(f"g{last_y - 1}_{y}")
                    elif cell_type_map[x][y] == "10":
                        wires.add(f"g{x}_{last_y}")
                        wires.add(f"p{x}_{last_y}")
                    else:
                        raise NotImplementedError
                else:
                    if cell_type_map[x][y] == "11":
                        wires.add(f"g{x}_{last_y}")
                        wires.add(f"p{x}_{last_y}")
                        wires.add(f"g{last_y - 1}_{y}")
                        wires.add(f"p{last_y - 1}_{y}")

                        wires.add(f"g{x}_{y}")
                        wires.add(f"p{x}_{y}")
                    elif cell_type_map[x][y] == "00":
                        wires.add(f"p{x}_{last_y}")
                        wires.add(f"p{last_y - 1}_{y}")

                        wires.add(f"p{x}_{y}")
                    elif cell_type_map[x][y] == "01":
                        wires.add(f"p{x}_{last_y}")
                        wires.add(f"p{last_y - 1}_{y}")
                        wires.add(f"g{last_y - 1}_{y}")

                        wires.add(f"g{x}_{y}")
                        wires.add(f"p{x}_{y}")
                    elif cell_type_map[x][y] == "10":
                        wires.add(f"g{x}_{last_y}")
                        wires.add(f"p{last_y - 1}_{y}")

                        wires.add(f"g{x}_{y}")
                        wires.add(f"p{x}_{y}")
                last_y = y

    for i in range(input_bit):
        wires.add(f"p{i}_{i}")
        wires.add(f"g{i}_{i}")
        wires.add(f"g{i}_{0}")
        wires.add(f"c{x}")
    assert 0 not in wires
    assert "0" not in wires
    content += "\twire "

    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"\tassign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"\tassign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"\tassign c{i} = g{i}_0;\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                gij = f"g{x}_{y}"
                pij = f"p{x}_{y}"

                gik = f"g{x}_{last_y}"
                pik = f"p{x}_{last_y}"

                gkj = f"g{last_y - 1}_{y}"
                pkj = f"p{last_y - 1}_{y}"

                if y == 0:
                    # GREY CELL
                    if cell_type_map[x][y] == "11":
                        content += f"\tGREY_CELL_11 cell_{x}_{y}_grey({gik}, {pik}, {gkj}, {gij});\n"
                    elif cell_type_map[x][y] == "00":
                        content += f"\tGREY_CELL_00 cell_{x}_{y}_grey({gij});\n"
                    elif cell_type_map[x][y] == "01":
                        content += (
                            f"\tGREY_CELL_01 cell_{x}_{y}_grey({pik}, {gkj}, {gij});\n"
                        )
                    else:
                        content += f"\tGREY_CELL_10 cell_{x}_{y}_grey({gik}, {gij});\n"
                else:
                    # BLACK CELL
                    if cell_type_map[x][y] == "11":
                        content += f"\tBLACK_CELL_11 cell_{x}_{y}_black({gik}, {pik}, {gkj}, {pkj}, {gij}, {pij});\n"
                    elif cell_type_map[x][y] == "00":
                        content += f"\tBLACK_CELL_00 cell_{x}_{y}_black({pik}, {pkj}, {pij});\n"
                    elif cell_type_map[x][y] == "01":
                        content += f"\tBLACK_CELL_01 cell_{x}_{y}_black({pik}, {pkj}, {gkj}, {gij}, {pij});\n"
                    else:
                        content += f"\tBLACK_CELL_10 cell_{x}_{y}_black({gik}, {pik}, {pkj}, {gij}, {pij});\n"
                last_y = y

    content += "\tassign s[0] = a[0] ^ b[0];\n"
    content += "\tassign c0 = g0_0;\n"
    content += f"\tassign cout = c{input_bit - 1};\n"
    for i in range(1, input_bit):
        content += f"\tassign s[{i}] = p{i}_{i} ^ c{i - 1};\n"
    content += "endmodule"
    content += "\n\n"

    return content


def get_mask_map(cell_map: np.ndarray) -> np.ndarray:
    bit_width = len(cell_map)
    mask_map = np.full((2, bit_width, bit_width), False)
    for i in range(bit_width):
        for j in range(1, i):
            if cell_map[i, j] == 1:
                mask_map[0, i, j] = 0
                mask_map[1, i, j] = 1
            else:
                mask_map[0, i, j] = 1
                mask_map[1, i, j] = 0

    return mask_map


def get_level_map(cell_map: np.ndarray) -> np.ndarray:
    level_map = np.full_like(cell_map, -1)
    bit_width = len(cell_map)

    split_map = {}

    for x in range(bit_width - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                split_map[(x, y)] = last_y
                last_y = y

    def __get_level_value(x, y):
        if x == y:
            level_map[x, y] = 0
            return 0
        else:
            last_y = split_map[(x, y)]
            if level_map[x, y] >= 0:
                return level_map[x, y]
            else:
                left_level = __get_level_value(x, last_y)
                right_level = __get_level_value(last_y - 1, y)

                level = max(left_level, right_level) + 1
                level_map[x, y] = level

                return level

    for i in range(bit_width):
        level = __get_level_value(i, 0)
        level_map[i, 0] = level

    return level_map


def get_fanout_map(cell_map: np.ndarray) -> dict:
    bit_width = len(cell_map)
    fanout_map = {}

    for x in range(bit_width - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if (x, last_y) in fanout_map.keys():
                    fanout_map[(x, last_y)].append((x, y))
                else:
                    fanout_map[(x, last_y)] = [(x, y)]

                if (last_y - 1, y) in fanout_map.keys():
                    fanout_map[(last_y - 1, y)].append((x, y))
                else:
                    fanout_map[(last_y - 1, y)] = [(x, y)]

                last_y = y
    return fanout_map


def remove_tree_cell(cell_map: np.ndarray, target_x_list, target_y_list) -> np.ndarray:
    fanout_map = get_fanout_map(cell_map)
    new_cell_map = copy.deepcopy(cell_map)

    def __remove_cell(x, y):
        if x == y or y == 0:
            return
        else:
            new_cell_map[x, y] = 0
            if type(x) == torch.Tensor:
                x = int(x.cpu().flatten()[0])
                y = int(y.cpu().flatten()[0])
            for fanout_x, fanout_y in fanout_map[(x, y)]:
                __remove_cell(fanout_x, fanout_y)

    for x, y in zip(target_x_list, target_y_list):
        __remove_cell(x, y)
    new_cell_map = cell_map_legalize(new_cell_map)
    return new_cell_map


def draw_cell_map(cell_map: np.ndarray, power_mask: np.ndarray = None):
    plt.figure(figsize=[16, 10])

    bit_width = len(cell_map)
    level_map = get_level_map(cell_map)
    points = []
    points_color = []
    points_text = []
    lines = []
    max_level = np.max(level_map)

    for i in range(bit_width):
        last_j = i
        for j in range(i, -1, -1):
            if cell_map[i, j] == 1:
                points.append([bit_width - i, max_level - level_map[i, j]])
                points_text.append(f"({i}:{j})")
                if j == 0:
                    points_color.append("orange")
                else:
                    points_color.append("black")

                if j != i:
                    p_1 = [bit_width - i, max_level - level_map[i, j]]
                    p_2 = [bit_width - i, max_level - level_map[i, last_j]]
                    p_3 = [
                        bit_width - (last_j - 1),
                        max_level - level_map[last_j - 1, j],
                    ]
                    lines.append((p_1, p_2))
                    lines.append((p_1, p_3))

                last_j = j

    for line in lines:
        x, y = np.transpose(line)
        plt.plot(x, y, c="grey", alpha=0.5)

    x, y = np.transpose(points)
    if power_mask is None:
        plt.scatter(x, y, c=points_color)
    else:
        mask_color = []
        index = 0
        for i in range(bit_width):
            last_j = i
            for j in range(i, -1, -1):
                if cell_map[i, j] == 1:
                    mask_color.append(power_mask[i, j])
                    points_text[index] += f"\n{power_mask[i, j]:.2}"
                    last_j = j
                    index += 1
        plt.scatter(x, y, c=mask_color, s=100)

        i, j = np.unravel_index(np.argmax(power_mask), power_mask.shape)
        x, y = bit_width - i, max_level - level_map[i, j]
        plt.scatter(x, y, s=100, facecolors="none", edgecolors="red")

        plt.colorbar()
    for index, point in enumerate(points):
        x, y = np.transpose(point)
        plt.text(x, y - 0.5, points_text[index])

    plt.tight_layout()
    plt.show()


"""
UFO-MAC begin
"""


class FDC:
    """
    di = k0 x F_black + k1 x F_blue + k2 x N_black + k3 x N_blue + b
    """

    def __init__(self) -> None:
        self.k_0 = None
        self.k_1 = None
        self.k_2 = None
        self.k_3 = None
        self.b = None

        self.cell_map = None
        self.fanout_map = None
        self.arrival_time = None

    def load_params(self, db_path):
        with open(db_path, "r") as file:
            data = json.load(file)
        self.k_0 = data["k_0"]
        self.k_1 = data["k_1"]
        self.k_2 = data["k_2"]
        self.k_3 = data["k_3"]
        self.b = data["b"]

    def fit(self):
        pass

    def gen_train_data(self, save_data_path:str="./db/dfc_data.json"):
        for i in range(100):
            # random
            pass

    def set_arrival_time(self, arrival_time):
        self.arrival_time = arrival_time

    def set_cell_map(self, cell_map: np.ndarray):
        self.cell_map = copy.deepcopy(cell_map)
        self.fanout_map = get_fanout_map(cell_map)

    def __get_delay(self, x, y):
        if y == 0:
            return {
                "N_black": 0,
                "N_blue": 0,
                "F_black": 0,
                "F_blue": 0,
                "delay": 0,
                "path": [(x, y)],
            }
        fanouts = self.fanout_map[(x, y)]
        fanout_list = []
        for fanout_node in fanouts:
            fanout_list.append(self.__get_delay(fanout_node[0], fanout_node[1]))
        indices = list(range(len(fanout_list)))
        max_delay_index = max(indices, key=lambda x: fanout_list[x]["delay"])
        path = fanout_list[max_delay_index]["path"]
        path.append((x, y))

        N_black = fanout_list[max_delay_index]["N_black"]
        N_blue = fanout_list[max_delay_index]["N_blue"]
        F_black = fanout_list[max_delay_index]["F_black"]
        F_blue = fanout_list[max_delay_index]["F_blue"]

        # 跟新参数
        if y == 0:
            N_blue += 1
        else:
            N_black += 1
        
        for fanout_node in fanouts:
            if fanout_node[1] == 0:
                F_blue += 1
            else:
                F_black += 1

        # 更新完了 计算delay
        delay = (
            self.k_0 * F_black
            + self.k_1 * F_blue
            + self.k_2 * N_black
            + self.k_3 * N_blue
            + self.b
        )

        return {
            "N_black": N_black,
            "N_blue": N_blue,
            "F_black": F_black,
            "F_blue": F_blue,
            "delay": delay,
            "path": path,
        }

    def preidct(self, bit_position: int) -> float:

        return self.__get_delay(bit_position, bit_position)["delay"]


def get_parent(cell_map: np.ndarray, x: int, y: int):
    if y == x:
        return None, None
    last_y = y + 1
    while cell_map[x, last_y] != 1:
        last_y += 1

    return (x, last_y), (last_y - 1, y)


def graph_opt(cell_map: np.ndarray, x_p: int, y_p: int) -> np.ndarray:
    """
    Algorithm 2
    procedure GraphOpt(p)
        Create a new node s
        nt f (s) ← t f (nt f (p)), nt f (s) ← t f (nt f (p))
        t f (p) ← s, nt f (p) ← nt f (nt f (p))
    end procedure
    tf: 正上方那个
    nt f : 斜上方那个

    简单来说: 斜亲变兄弟
    方法: 添加一个节点，首先是在自己这一列，成为自己的直亲
        然后 split 是斜亲的直亲
    """

    if x_p == y_p:
        return

    tf, ntf = get_parent(cell_map, x_p, y_p)
    _, ntf_ntf = get_parent(cell_map, ntf[0], ntf[1])

    if ntf_ntf is not None:
        assert x_p >= ntf_ntf[0] + 1
        cell_map[x_p, ntf_ntf[0] + 1] = 1


def get_subtree_mask(cell_map: np.ndarray, x: int, y: int):
    assert x >= y
    subtree_mask = cell_map


def get_subtree_depth(cell_map: np.ndarray, x: int, y: int):
    pass


def get_subtree_max_depth_node(cell_map: np.ndarray, x: int, y: int):
    # TODO
    pass


def get_subtree_max_sibling_node(cell_map: np.ndarray, x: int, y: int):
    # TODO
    pass


def prefix_graph_optimization(
    cell_map: np.ndarray, arrival_times, timing_constraints, fdc: FDC, max_iter
):
    bit_width = len(arrival_times)
    fdc.set_arrival_time(arrival_times)
    depth_th = np.log2(bit_width)
    for iteration_index in range(max_iter):
        chage_flag = False
        for cur_bit in range(bit_width, -1, -1):
            fdc.set_cell_map()
            cur_time = fdc.preidct(cur_bit)
            if cur_time > timing_constraints[cur_bit]:
                chage_flag = True
                # violated 需要做优化
                depth = get_subtree_depth(cell_map, cur_bit, cur_bit)
                if depth > depth_th:
                    # 超过了 log2 N 需要深度优化
                    x_p, y_p = get_subtree_max_depth_node(cell_map, cur_bit, cur_bit)
                else:
                    # 针对 fanout 优化
                    x_p, y_p = get_subtree_max_sibling_node(cell_map, cur_bit, cur_bit)
                graph_opt(cell_map, x_p, y_p)
        if not chage_flag:
            # 没有新的更新了
            break


"""
UFO-MAC end
"""

if __name__ == "__main__":
    # cell_map = get_default_init(64)
    cell_map = get_sklansky_init(64)
    # cell_map = get_brent_kung_init(128)
    # cell_map = get_han_carlson_init(64)
    # cell_map = get_kogge_stone_init(64)
    print(cell_map)
    print(get_level_map(cell_map))
    fanout_map = get_fanout_map(cell_map)
    print(fanout_map[(31, 0)])
    draw_cell_map(cell_map)

    cell_map = remove_tree_cell(cell_map, [39], [32])
    fanout_map = get_fanout_map(cell_map)
    print(fanout_map[(31, 0)])
    draw_cell_map(cell_map)
