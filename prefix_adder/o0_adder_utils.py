import copy
import numpy as np


def legalize(cell_map, min_map, input_bit):
    min_map = copy.deepcopy(cell_map)
    for i in range(input_bit):
        min_map[i, 0] = 0
        min_map[i, i] = 0
    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                cell_map[last_y - 1, y] = 1
                min_map[last_y - 1, y] = 0
                last_y = y
    return cell_map, min_map


def cell_map_legalize(cell_map):
    input_bit = len(cell_map)
    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                cell_map[last_y - 1, y] = 1
                last_y = y
    return cell_map


def get_min_map_from_cell_map(cell_map, input_bit):
    min_map = copy.deepcopy(cell_map)
    for i in range(input_bit):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    return min_map


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


AdderInitialMethod = {
    "default": get_default_init,
    "normal": get_default_init,
    "brent_kung": get_brent_kung_init,
    "sklansky": get_sklansky_init,
}


def get_min_map_from_cell_map(cell_map):
    min_map = copy.deepcopy(cell_map)
    for i in range(len(cell_map)):
        min_map[i, i] = 0
        min_map[i, 0] = 0
    return min_map


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


def adder_output_verilog_main(cell_map: np.ndarray, input_bit: int) -> str:
    content = ""

    content += f"module main(a,b,s,cout);\n"
    content += f"input [{input_bit - 1}:0] a,b;\n"
    content += f"output [{input_bit - 1}:0] s;\n"
    content += "output cout;\n"
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
    content += "wire "

    for i, wire in enumerate(wires):
        if i < len(wires) - 1:
            content += f"{wire},"
        else:
            content += f"{wire};\n"
    content += "\n"

    for i in range(input_bit):
        content += f"assign p{i}_{i} = a[{i}] ^ b[{i}];\n"
        content += f"assign g{i}_{i} = a[{i}] & b[{i}];\n"

    for i in range(1, input_bit):
        content += f"assign g{i}_0 = c{i};\n"

    for x in range(input_bit - 1, 0, -1):
        last_y = x
        for y in range(x - 1, -1, -1):
            if cell_map[x, y] == 1:
                assert cell_map[last_y - 1, y] == 1
                if y == 0:  # add grey module
                    content += f"GREY grey{x}(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, c{x});\n"
                else:
                    content += f"BLACK black{x}_{y}(g{x}_{last_y}, p{x}_{last_y}, g{last_y - 1}_{y}, p{last_y - 1}_{y}, g{x}_{y}, p{x}_{y});\n"
                last_y = y

    content += "assign s[0] = a[0] ^ b[0];\n"
    content += "assign c0 = g0_0;\n"
    content += f"assign cout = c{input_bit - 1};\n"
    for i in range(1, input_bit):
        content += f"assign s[{i}] = p{i}_{i} ^ c{i - 1};\n"
    content += "endmodule"
    content += "\n\n"

    return content
