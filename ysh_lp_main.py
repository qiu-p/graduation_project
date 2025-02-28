import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from ysh_tools.utils import make_dir

from o0_state import State, legal_FA_list, legal_HA_list
from o0_mul_utils import (
    decompose_compressor_tree,
    get_compressor_tree,
    get_initial_partial_product,
    legal_FA_list,
    legal_HA_list,
    legalize_compressor_tree,
    write_mul,
)

class WireMap:
    def __init__(
        self,
        state: State,
        output_base_dir: str,
        task_index: int = 0,
        time_bound_s: int = 7200
        ):
        self.bit_width = state.bit_width
        self.pp_encode_type = state.encode_type
        self.cur_state = state
        
        self._get_coefficient()
        
        self.ct_decomposed = self.cur_state.ct_decomposed # (2, stage_num, column_num)
        self.stage_num = self.cur_state.stage_num
        self.column_num = len(self.cur_state.initial_pp)
        self.ct_decomposed = self.ct_decomposed.astype(int)
        self.pp = self.cur_state.sequence_pp
        
        self.output_dir = '{}/{}bits_{}_{}/'.format(output_base_dir, self.bit_width, self.pp_encode_type, task_index)
        self.wire_map_filename = self.output_dir + 'wire_map.txt'
        make_dir(self.output_dir, 'i')
        self.grb_log_filename = self.output_dir + 'grb.log'
        self.time_bound_s = time_bound_s

    def _get_coefficient(self):
        self.Z = 1e30
        if self.bit_width == 8 and self.pp_encode_type == 'and':
            self.beta = [0.16079747676849365, 0.4300612211227417, -0.2408769726753235, -0.2631942927837372]    # a b c _
            self.alpha = [-0.0007462067878805101, 0.32097557187080383, 0.06264670938253403]      # a b _
            self.v_s = [0.626052737236023, 0.1845366358757019, 0.3636322617530823]      # a b c
            self.v_c = [0.08953072130680084, 0.1415812224149704, -0.2348821461200714]      # a b c
            self.u_s = [-0.07915672659873962, 0.04327170178294182]      # a b
            self.u_c = [0.2704322934150696, 0.446780800819397]      # a b
        elif self.bit_width == 8 and self.pp_encode_type == 'booth':
            self.beta = [0.15726950764656067, 0.4243611693382263, -0.24238567054271698, -0.2550172209739685]    # a b c _
            self.alpha = [-0.021816110238432884, 0.30275338888168335, 0.06661204248666763]      # a b _
            self.v_s = [0.6199445128440857, 0.1789475828409195, 0.3569159209728241]      # a b c
            self.v_c = [0.08906574547290802, 0.14123566448688507, -0.23478761315345764]      # a b c
            self.u_s = [-0.08771635591983795, 0.03796663507819176]      # a b
            self.u_c = [0.2587961256504059, 0.4413468539714813]      # a b
        elif self.bit_width == 16 and self.pp_encode_type == 'and':
            self.beta = [0.18264345824718475, 0.4542485475540161, -0.22305601835250854, -0.27543628215789795]    # a b c _
            self.alpha = [0.05456593632698059, 0.38130441308021545, 0.07291869819164276]      # a b _
            self.v_s = [ 0.6638590097427368, 0.22307394444942474, 0.40062880516052246]      # a b c
            self.v_c = [0.11934997886419296, 0.1720152646303177, -0.20611044764518738]      # a b c
            self.u_s = [-0.037079814821481705, 0.08861196041107178]      # a b
            self.u_c = [0.31457823514938354, 0.49468693137168884]      # a b
        elif self.bit_width == 16 and self.pp_encode_type == 'booth':
            self.beta = [0.18308846652507782, 0.4548701345920563, -0.22277630865573883, -0.24895857274532318]    # a b c _
            self.alpha = [0.0433744452893734, 0.3665539622306824, 0.10456280410289764]      # a b _
            self.v_s = [0.65415358543396, 0.21383334696292877, 0.39061903953552246]      # a b c
            self.v_c = [0.10999104380607605, 0.16348949074745178, -0.21580485999584198]      # a b c
            self.u_s = [-0.04598590359091759, 0.07823961973190308]      # a b
            self.u_c = [0.3057803213596344, 0.4843278229236603]      # a b
        elif self.bit_width == 32 and self.pp_encode_type == 'and':
            self.beta = [0.20663635432720184, 0.47882795333862305, -0.20063605904579163, -0.2667306661605835]    # a b c _
            self.alpha = [0.0914212241768837, 0.41770926117897034, 0.09179569780826569]      # a b _
            self.v_s = [0.6896058917045593, 0.248875230550766, 0.42565295100212097]      # a b c
            self.v_c = [0.1424940973520279, 0.19539333879947662, -0.18394790589809418]      # a b c
            self.u_s = [-0.006348270922899246, 0.11787202209234238]      # a b
            self.u_c = [0.34477677941322327, 0.5234821438789368]      # a b
        elif self.bit_width == 32 and self.pp_encode_type == 'booth':
            self.beta = [0.21913373470306396, 0.49130979180336, -0.1882590800523758, -0.2195352166891098]    # a b c _
            self.alpha = [0.07835449278354645, 0.40483102202415466, 0.15057024359703064]      # a b _
            self.v_s = [0.6829172968864441, 0.24217435717582703, 0.41905805468559265]      # a b c
            self.v_c = [0.13641847670078278, 0.1893012672662735, -0.18995371460914612]      # a b c
            self.u_s = [-0.017787126824259758, 0.10710743069648743]      # a b
            self.u_c = [0.3328976035118103, 0.511999785900116]      # a b
    
    def _gen_variables(self):
        # power
        self.variables_power_f = []
        for i in range(self.stage_num):
            row_power_f = []
            for j in range(self.column_num):
                temp_name = 'power_f_{}_{}'.format(i,j)
                row_power_f.append(self.model.addVar(0, float('inf'), 0, vtype=GRB.CONTINUOUS, name=temp_name))
            self.variables_power_f.append(row_power_f)
        self.variables_power_h = []
        for i in range(self.stage_num):
            row_power_h = []
            for j in range(self.column_num):
                temp_name = 'power_h_{}_{}'.format(i,j)
                row_power_h.append(self.model.addVar(0, float('inf'), 0, vtype=GRB.CONTINUOUS, name=temp_name))
            self.variables_power_h.append(row_power_h)
        # frequency
        self.variables_frequency_in = []
        for i in range(self.stage_num):
            row_frequency_in = []
            for j in range(self.column_num):
                slice_frequency_in = []
                for k in range(self.pp[i][j]):
                    temp_name = 'frequency_in_{}_{}_{}'.format(i,j,k)
                    slice_frequency_in.append(self.model.addVar(0, float('inf'), 0, vtype=GRB.CONTINUOUS, name=temp_name))
                row_frequency_in.append(slice_frequency_in)
            self.variables_frequency_in.append(row_frequency_in)
        self.variables_frequency_out = []
        for i in range(self.stage_num):
            row_frequency_out = []
            for j in range(self.column_num):
                slice_frequency_out = []
                for k in range(self.pp[i][j]):
                    temp_name = 'frequency_out_{}_{}_{}'.format(i,j,k)
                    slice_frequency_out.append(self.model.addVar(0, float('inf'), 0, vtype=GRB.CONTINUOUS, name=temp_name))
                row_frequency_out.append(slice_frequency_out)
            self.variables_frequency_out.append(row_frequency_out)
        # Z
        self.variables_Z = []
        for i in range(self.stage_num):
            row_Z = []
            for j in range(self.column_num):
                slice_Z = []
                for k in range(self.pp[i][j]):
                    slice_Z_row = []
                    for k2 in range(self.pp[i][j]):
                        temp_name = 'Z_{}_{}_{}_{}'.format(i,j,k,k2)
                        slice_Z_row.append(self.model.addVar(vtype=GRB.BINARY, name=temp_name))
                    slice_Z.append(slice_Z_row)
                row_Z.append(slice_Z)
            self.variables_Z.append(row_Z)

    def _set_objective(self):
        obj = 0
        for i in range(self.stage_num):
            for j in range(self.column_num):
                obj += self.variables_power_f[i][j]
                obj += self.variables_power_h[i][j]
        self.model.setObjective(obj, GRB.MINIMIZE)

    def _add_constraint(self):
        # 2
        for i in range(self.stage_num):
            for j in range(self.column_num):
                temp_name = 'c2_{}_{}'.format(i, j)
                constrain = 0
                for k in range(self.ct_decomposed[0][i][j]):
                    print(i, j, k)
                    print(len(self.variables_frequency_in), len(self.variables_frequency_in[0]), len(self.variables_frequency_in[0][0]))
                    constrain += \
                        self.beta[0]*self.variables_frequency_in[i][j][3*k] + \
                        self.beta[1]*self.variables_frequency_in[i][j][3*k+1] + \
                        self.beta[2]*self.variables_frequency_in[i][j][3*k+2] + self.beta[3]
                self.model.addConstr(self.variables_power_f[i][j]==constrain, temp_name)
        # 3
        for i in range(self.stage_num):
            for j in range(self.column_num):
                temp_name = 'c3_{}_{}'.format(i, j)
                constrain = 0
                for k in range(self.ct_decomposed[1][i][j]):
                    constrain += \
                        self.alpha[0]*self.variables_frequency_in[i][j][3*self.ct_decomposed[0][i][j]+2*k] + \
                        self.alpha[1]*self.variables_frequency_in[i][j][3*self.ct_decomposed[0][i][j]+2*k+1] + self.alpha[2]
                self.model.addConstr(self.variables_power_h[i][j]==constrain, temp_name)
        # 4
        for i in range(self.stage_num-1):
            for j in range(self.column_num):
                for k in range(self.ct_decomposed[0][i][j]):
                    temp_name = 'c4_{}_{}_{}'.format(i, j, k)
                    constrain = \
                        self.v_s[0]*self.variables_frequency_in[i][j][3*k] + \
                        self.v_s[1]*self.variables_frequency_in[i][j][3*k+1] + \
                        self.v_s[2]*self.variables_frequency_in[i][j][3*k+2]
                    self.model.addConstr(self.variables_frequency_out[i+1][j][k]==constrain, temp_name)
        # 6
        for i in range(self.stage_num-1):
            for j in range(self.column_num-1):
                for k in range(self.ct_decomposed[0][i][j]):
                    temp_name = 'c6_{}_{}_{}'.format(i, j, k)
                    kappa = self.pp[i][j+1] - 2*self.ct_decomposed[0][i][j+1] - self.ct_decomposed[1][i][j+1]
                    constrain = \
                        self.v_c[0]*self.variables_frequency_in[i][j][3*k] + \
                        self.v_c[1]*self.variables_frequency_in[i][j][3*k+1] + \
                        self.v_c[2]*self.variables_frequency_in[i][j][3*k+2]
                    # print('c6 i={} j={} k={} kappa={} k+kappa={}'.format(i, j, k, kappa, k+kappa))
                    self.model.addConstr(self.variables_frequency_out[i+1][j+1][k+kappa]==constrain, temp_name)
        # 7
        for i in range(self.stage_num-1):
            for j in range(self.column_num):
                for k in range(self.ct_decomposed[1][i][j]):
                    temp_name = 'c7_{}_{}_{}'.format(i, j, k)
                    constrain = \
                        self.u_s[0]*self.variables_frequency_in[i][j][3*self.ct_decomposed[0][i][j]+2*k] + \
                        self.u_s[1]*self.variables_frequency_in[i][j][3*self.ct_decomposed[0][i][j]+2*k+1]
                    self.model.addConstr(self.variables_frequency_out[i+1][j][k+self.ct_decomposed[0][i][j]]==constrain, temp_name)
        # 8
        for i in range(self.stage_num-1):
            for j in range(self.column_num-1):
                for k in range(self.ct_decomposed[1][i][j]):
                    temp_name = 'c8_{}_{}_{}'.format(i, j, k)
                    kappa = self.pp[i][j+1] - 2*self.ct_decomposed[0][i][j+1] - self.ct_decomposed[1][i][j+1]
                    constrain = \
                        self.u_c[0]*self.variables_frequency_in[i][j][3*self.ct_decomposed[0][i][j]+2*k] + \
                        self.u_c[1]*self.variables_frequency_in[i][j][3*self.ct_decomposed[0][i][j]+2*k+1]
                    self.model.addConstr(self.variables_frequency_out[i+1][j+1][k+self.ct_decomposed[0][i][j]+kappa]==constrain, temp_name)
        # 9
        for i in range(self.stage_num-1):
            for j in range(self.column_num):
                for k in range(self.pp[i][j]-3*self.ct_decomposed[0][i][j]-2*self.ct_decomposed[1][i][j]):
                    temp_name = 'c9_{}_{}_{}'.format(i, j, k)
                    self.model.addConstr(
                        self.variables_frequency_out[i+1][j][k+self.ct_decomposed[0][i][j]+self.ct_decomposed[1][i][j]]==self.variables_frequency_in[i][j][k+3*self.ct_decomposed[0][i][j]+2*self.ct_decomposed[1][i][j]], 
                        temp_name)
        # 10
        for i in range(self.stage_num):
            for j in range(self.column_num):
                for k in range(self.pp[i][j]):
                    for l in range(self.pp[i][j]):
                        temp_name = 'c10_{}_{}_{}_{}'.format(i, j, k, l)
                        constrain = self.Z * (1-self.variables_Z[i][j][k][l])
                        self.model.addConstr(
                            self.variables_frequency_in[i][j][k]-self.variables_frequency_out[i][j][l]<=constrain, 
                            temp_name)
        # 11
        for i in range(self.stage_num):
            for j in range(self.column_num):
                for k in range(self.pp[i][j]):
                    for l in range(self.pp[i][j]):
                        temp_name = 'c11_{}_{}_{}_{}'.format(i, j, k, l)
                        constrain = self.Z * (1-self.variables_Z[i][j][k][l])
                        self.model.addConstr(
                            self.variables_frequency_out[i][j][k]-self.variables_frequency_in[i][j][l]<=constrain, 
                            temp_name)
        # 12
        for j in range(self.column_num-1):
            for k in range(self.pp[0][j]):
                temp_name = 'c12_{}_{}'.format(j, k)
                self.model.addConstr(self.variables_frequency_out[0][j][k]==1, temp_name)
        # 15
        for i in range(self.stage_num):
            for j in range(self.column_num):
                for l in range(self.pp[i][j]):
                    temp_name = 'c15_{}_{}_{}'.format(i, j, l)
                    constrain = 0
                    for k in range(self.pp[i][j]):
                        constrain += self.variables_Z[i][j][k][l]
                    self.model.addConstr(constrain==1, temp_name)
        # 16
        for i in range(self.stage_num):
            for j in range(self.column_num):
                for k in range(self.pp[i][j]):
                    temp_name = 'c16_{}_{}_{}'.format(i, j, k)
                    constrain = 0
                    for l in range(self.pp[i][j]):
                        constrain += self.variables_Z[i][j][k][l]
                    self.model.addConstr(constrain==1, temp_name)

    def run(self):
        with gp.Env() as env, gp.Model(env=env) as model:
            # 创建模型
            model.setParam(GRB.Param.LogFile, self.grb_log_filename)
            # model.setParam(GRB_INT_PAR_MIPFOCUS, "1");
            model.setParam(GRB.Param.Presolve, 2)
            model.setParam(GRB.Param.TimeLimit, self.time_bound_s)
            # model.setParam(GRB_DBL_PAR_HEURISTICS, "0.5");
            # model.setParam(GRB.Param.MIPGap, 0.005)
            model.setParam(GRB.Param.MIPGap, 0.1)
            self.model = model

            # 添加变量
            self._gen_variables()

            # 设置目标函数
            self._set_objective()

            # 添加约束
            self._add_constraint()

            print('===========================================================')
            print('start optimaizing...')
            # 求解模型
            self.model.optimize()

            # 输出结果
            if model.status == GRB.OPTIMAL:
                print(f"Optimal Value: {model.objVal}")
                with open(self.wire_map_filename, 'w') as f:
                    for i1, v1 in enumerate(self.variables_Z):
                        for i2, v2 in enumerate(v1):
                            str_m = ''
                            Z = []
                            for i3, v3 in enumerate(v2):
                                Z_row = []
                                for i4, v4 in enumerate(v3):
                                    Z_row.append(v4.X)
                                Z.append(Z_row)
                            print("row: {}, col: {}\n{}".format(i1, i2, Z))
                            for j1 in range(len(Z)):
                                for j2 in range(len(Z)):
                                    if Z[j2][j1] == 1:
                                       str_m += '{} '.format(j2) 
                                       break
                            f.write(str_m)
                            f.write('\n')
            # 如果模型不可行，计算 IIS
            elif model.status == GRB.INFEASIBLE:
                print("模型不可行，正在计算 IIS...")
                model.computeIIS()
                model.write("model.ilp")  # 将 IIS 保存到文件
                print("IIS 已保存到 model.ilp 文件，请检查冲突的约束。")

if __name__ == '__main__':
    task_index = 1 # 0: wallace  1: dadda

    bit_width = 32
    pp_encode_type = 'and'
    # pp_encode_type = 'booth'
    if task_index == 0:
        init_ct_type = 'wallace'
    elif task_index == 1:
        init_ct_type = 'dadda'
    MAX_STAGE_NUM = 2 * bit_width
    state = State(
        bit_width,
        pp_encode_type,
        MAX_STAGE_NUM,
        use_pp_wiring_optimize=True,
        pp_wiring_init_type='default',
        use_compressor_map_optimize=False,
        compressor_map_init_type='default',
        use_final_adder_optimize=False,
        final_adder_init_type='default',
        top_name='MUL',
    )
    state.init(init_ct_type)
    wire_map = WireMap(state, 'ysh_output', task_index)

    print('stage_num:', state.stage_num)
    print(state.ct)
    print(wire_map.ct_decomposed)
    print(wire_map.pp)

    wire_map.run()
    