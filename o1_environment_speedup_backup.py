import numpy as np
import torch
import os
import copy
import math
import random
from collections import deque
from multiprocessing import Pool
import torch.multiprocessing as mp
 
from o0_global_const import InitialState, PartialProduct, IntBitWidth, StrBitWidth, GOMILInitialState
from o5_utils import abc_constr_gen, sta_scripts_gen, ys_scripts_gen, ys_scripts_v2_gen, ys_scripts_v3_gen, ys_scripts_v5_gen, get_ppa, EasyMacPath, EasyMacTarPath, BenchmarkPath
from o1_environment import RefineEnv

class SpeedUpRefineEnv(RefineEnv):
    def instance_FA(self, num, port1, port2, port3, outport1, outport2):
        FA_str='\tFA F{}(.a({}),.b({}),.cin({}),.sum({}),.cout({}));\n'.format(num,port1,port2,port3,outport1,outport2)
        return FA_str
    
    def instance_HA(self, num, port1, port2, outport1, outport2):
        HA_str='\tHA H{}(.a({}),.cin({}),.sum({}),.cout({}));\n'.format(num,port1,port2,outport1,outport2)
        return HA_str
    
    def update_remain_pp(self, ct, stage, final_stage_pp):
        ct32=ct[0][stage][:]
        ct22=ct[1][stage][:]

        str_width = ct.shape[2]
        initial_state=np.zeros((str_width))

        for i in range(str_width):
            if i==str_width-1:
                initial_state[i] = final_stage_pp[i] - ct32[i] - ct22[i]
            else:
                initial_state[i] = final_stage_pp[i] - ct32[i] - ct22[i] - ct32[i+1] - ct22[i+1]
        initial_state = initial_state.astype(int)
        return initial_state
    
    def update_final_pp(self, ct, stage, mult_type):
        ct32=np.sum(ct[0][:stage+1][:],axis=0)
        ct22=np.sum(ct[1][:stage+1][:],axis=0)
        
        str_width = len(ct32)
        input_width = (str_width+1)//2

        initial_state=np.zeros((str_width))
        if mult_type=='and':
            for i in range(1,str_width+1):
                initial_state[i-1]=input_width-abs(i-input_width)
        
        initial_state=initial_state[::-1]

        for i in range(str_width):
            if i==str_width-1:
                initial_state[i] = initial_state[i] - 2*ct32[i] - ct22[i]
            else:
                initial_state[i] = initial_state[i] - 2*ct32[i] - ct22[i] + ct32[i+1] + ct22[i+1]
        initial_state = initial_state.astype(int)
        return initial_state
    
    def write_CT(self, input_width, ct=[]):
        """
        input:
            *input_width:乘法器位宽
            *ct: 压缩器信息，shape为2*stage*str_width
        """
        stage,str_width=ct.shape[1],ct.shape[2]
        mul_type = self.pp_encode_type
        # print(f"==========mul type: {mul_type}")
        
        # 输入输出端口 
        ct_str="module Compressor_Tree(a,b"
        for i in range(str_width):
            ct_str +=',data{}_s{}'.format(i,stage)
        ct_str+=');\n'

        # 位宽
        ct_str +='\tinput[{}:0] a;\n'.format(input_width-1)
        ct_str +='\tinput[{}:0] b;\n'.format(input_width-1)
        
        final_state = self.update_final_pp(ct,stage,mul_type)
        #print("final",final_state)

        # TODO: 根据每列最终的部分积确定最终的输出位宽
        for i in range(str_width):
            ct_str +='\toutput[{}:0] data{}_s{};\n'.format(int(final_state[i])-1,i,stage)
        
        # 调用production模块，产生部分积
        ct_str +='\n\t//pre-processing block : production\n'
        if mul_type=='and':
            for i in range(1,str_width+1):
                len_i=input_width-abs(i-input_width)
                ct_str +='\twire[{}:0] out{};\n'.format(len_i-1,i-1)
            
            ct_str+='\tproduction PD0(.a(a),.b(b)'
            for i in range(str_width):
                ct_str+=',.out{}(out{})'.format(i,i)
            ct_str+=');'
        
        FA_num = 0
        HA_num = 0

        # 生成每个阶段的压缩树
        for stage_num in range(stage):
            ct_str+='\n\t//****The {}th stage****\n'.format(stage_num+1)
            final_stage_pp = self.update_final_pp(ct,stage_num,mul_type)

            remain_pp = self.update_remain_pp(ct, stage_num, final_stage_pp)

            for i in range(str_width):
                ct_str += '\twire[{}:0] data{}_s{};\n'.format(final_stage_pp[i]-1,i,stage_num+1)

            num_tmp = 0
            for j in range(str_width):
                if stage_num==0:
                    for k in range(ct[0][stage_num][j]):
                        port1="out{}[{}]".format(j,3*k)
                        port2="out{}[{}]".format(j,3*k+1)
                        port3="out{}[{}]".format(j,3*k+2)
                        outport1="data{}_s{}[{}]".format(j,stage_num+1,k)
                        if j!=0:
                            outport2="data{}_s{}[{}]".format(j-1,stage_num+1,k+ct[0][stage_num][j-1]+ct[1][stage_num][j-1]+remain_pp[j-1])
                        else:
                            ct_str+="\twire[0:0] tmp{};\n".format(num_tmp)
                            outport2="tmp{}".format(num_tmp)
                            num_tmp+=1     
                        ct_str += self.instance_FA(FA_num,port1,port2,port3,outport1,outport2)
                        FA_num += 1
                    for k in range(ct[1][stage_num][j]):
                        port1="out{}[{}]".format(j,3*ct[0][stage_num][j]+2*k)
                        port2="out{}[{}]".format(j,3*ct[0][stage_num][j]+2*k+1)
                        outport1="data{}_s{}[{}]".format(j,stage_num+1,ct[0][stage_num][j]+k)
                        if j!=0:
                            outport2="data{}_s{}[{}]".format(j-1,stage_num+1,k+ct[0][stage_num][j-1]+ct[1][stage_num][j-1]+ct[0][stage_num][j]+remain_pp[j-1])
                        else:
                            ct_str+="\twire[0:0] tmp{};\n".format(num_tmp)
                            outport2="tmp{}".format(num_tmp)     
                            num_tmp+=1                        
                        ct_str += self.instance_HA(HA_num,port1,port2,outport1,outport2)  
                        HA_num += 1     
                    # remain_ports
                    for k in range(remain_pp[j]):
                        ct_str+='\tassign data{}_s{}[{}] = out{}[{}];\n'.format(j,stage_num+1,ct[0][stage_num][j]+ct[1][stage_num][j]+k,j,3*ct[0][stage_num][j]+2*ct[1][stage_num][j]+k)
                else:
                    for k in range(ct[0][stage_num][j]):
                        port1="data{}_s{}[{}]".format(j,stage_num,3*k)
                        port2="data{}_s{}[{}]".format(j,stage_num,3*k+1)
                        port3="data{}_s{}[{}]".format(j,stage_num,3*k+2)
                        outport1="data{}_s{}[{}]".format(j,stage_num+1,k)
                        if j!=0:
                            outport2="data{}_s{}[{}]".format(j-1,stage_num+1,k+ct[0][stage_num][j-1]+ct[1][stage_num][j-1]+remain_pp[j-1])
                        else:
                            ct_str+="\twire[0:0] tmp{};\n".format(num_tmp)
                            outport2="tmp{}".format(num_tmp)
                            num_tmp+=1                    
                        ct_str += self.instance_FA(FA_num,port1,port2,port3,outport1,outport2)
                        FA_num += 1 
                    for k in range(ct[1][stage_num][j]):
                        port1="data{}_s{}[{}]".format(j,stage_num,3*ct[0][stage_num][j]+2*k)
                        port2="data{}_s{}[{}]".format(j,stage_num,3*ct[0][stage_num][j]+2*k+1)
                        outport1="data{}_s{}[{}]".format(j,stage_num+1,ct[0][stage_num][j]+k)
                        if j!=0:
                            outport2="data{}_s{}[{}]".format(j-1,stage_num+1,k+ct[0][stage_num][j-1]+ct[1][stage_num][j-1]+ct[0][stage_num][j]+remain_pp[j-1])
                        else:
                            ct_str+="\twire[0:0] tmp{};\n".format(num_tmp)
                            outport2="tmp{}".format(num_tmp)  
                            num_tmp+=1             
                        ct_str += self.instance_HA(HA_num,port1,port2,outport1,outport2)  
                        HA_num += 1
                    # remain_ports
                    for k in range(remain_pp[j]):
                        ct_str+='\tassign data{}_s{}[{}] = data{}_s{}[{}];\n'.format(j,stage_num+1,ct[0][stage_num][j]+ct[1][stage_num][j]+k,j,stage_num,3*ct[0][stage_num][j]+2*ct[1][stage_num][j]+k)
        ct_str+='endmodule\n'
        return ct_str

    def write_production_and(self, input_width):
        """
        input:
            * input_width:乘法器位宽
        return:
            * pp_str : and_production字符串
        """ 

        str_width = 2*input_width-1
        # 输入输出端口
        pp_str="module production (a,b"
        for i in range(str_width):
            pp_str +=',out'+str(i)
        pp_str +=');\n'

        # 位宽
        pp_str +='\tinput[{}:0] a;\n'.format(input_width-1)
        pp_str +='\tinput[{}:0] b;\n'.format(input_width-1)
        for i in range(1,str_width+1):
            len_i=input_width-abs(i-input_width)
            if len_i-1==0:
                pp_str +='\toutput out{};\n'.format(i-1)
            else:
                pp_str +='\toutput[{}:0] out{};\n'.format(len_i-1,i-1)
        
        # 赋值,out0代表高位
        for i in range(str_width):
            for j in range(input_width-abs(i-input_width+1)):
                #i代表a，j代表b
                if i==0 or i==str_width-1:
                    pp_str +='\tassign out{} = a[{}] & b[{}];\n'.format(i,int(input_width-i/2-1),int(input_width-1-i/2))
                else:
                    if i>=0 and i<=input_width-1:
                        pp_str +='\tassign out{}[{}] = a[{}] & b[{}];\n'.format(i,j,(input_width-i-1+j),(input_width-1-j))
                    else:
                        pp_str +='\tassign out{}[{}] = a[{}] & b[{}];\n'.format(i,j,j,(2*input_width-i-2-j))

        #
        pp_str +='endmodule\n'
        return pp_str

    # def write_FA(self):
    #     FA_str="""module FA (a, b, cin, sum, cout);
    #         \tinput a;
    #         \tinput b;
    #         \tinput cin;
    #         \toutput sum;
    #         \toutput cout;
    #         \treg sum,cout;
    #         \talways@(*)
    #         \t\t{cout,sum} = a + b + cin;   
    #         endmodule\n"""
    #     return FA_str

    # def write_HA(self):
    #     HA_str="""module HA (a, cin, sum, cout);
    #         \tinput a;
    #         \tinput cin;
    #         \toutput sum;
    #         \toutput cout;
    #         \treg sum,cout;
    #         \talways@(*)
    #         \t\t{cout,sum} = a + cin;
    #         endmodule\n"""
    #     return HA_str

    def write_HA(self):
        HA_str="""module HA (a, cin, sum, cout);
            \tinput a;
            \tinput cin;
            \toutput sum;
            \toutput cout;
            \tassign sum = a ^ cin; 
            \tassign cout = a & cin; 
            endmodule\n"""
        return HA_str
    
    def write_FA(self):
        FA_str="""module FA (a, b, cin, sum, cout);
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
    
    def write_mul(self,mul_verilog_file,input_width,ct):
        """
        input:
            * mul_verilog_file: 输出verilog路径
            *input_width: 输入电路的位宽
            *ct: 输入电路的压缩树，shape为2*stage_num*str_width
        """
        ct=ct.astype(int)[:,:,::-1]
        with open(mul_verilog_file,'w') as f:
            f.write(self.write_FA()) 
            f.write(self.write_HA())
            f.write(self.write_production_and(input_width))
            f.write(self.write_CT(input_width,ct))
            f.write("module MUL(a,b,clock,out);\n")
            f.write("\tinput clock;\n")
            f.write("\tinput[{}:0] a;\n".format(input_width-1))
            f.write("\tinput[{}:0] b;\n".format(input_width-1))
            f.write("\toutput[{}:0] out;\n".format(2*input_width-2))
            stage=ct.shape[1]
            final_pp=self.update_final_pp(ct,stage,mult_type='and')

            for i in range(len(final_pp)):
                f.write("\twire[{}:0] out{}_C;\n".format(final_pp[i]-1,i))

            f.write("\tCompressor_Tree C0(.a(a),.b(b)")
            
            for i in range(len(final_pp)):
                f.write(",.data{}_s{}(out{}_C)".format(i,stage,i))
            
            f.write(");\n")
            
            f.write("\twire[{}:0] addend;\n".format(2*input_width-2))
            f.write("\twire[{}:0] augned;\n".format(2*input_width-2))

            for i in range(len(final_pp)):
                if final_pp[len(final_pp)-i-1]==2:
                    f.write("\tassign addend[{}] = out{}_C[0];\n".format(i,len(final_pp)-i-1))
                    f.write("\tassign augned[{}] = out{}_C[1];\n".format(i,len(final_pp)-i-1))
                else:
                    f.write("\tassign addend[{}] = out{}_C[0];\n".format(i,len(final_pp)-i-1))
                    f.write("\tassign augned[{}] = 1'b0;\n".format(i))
            f.write("\twire[{}:0] tmp = addend + augned;\n".format(2*input_width-2))
            f.write("\tassign out = tmp[{}:0];\n".format(2*input_width-2))
            f.write("endmodule\n")

    def read_ct(self, ct_file):
        with open(ct_file,'r') as f:
            lines=f.readlines()
        width=int(lines[0].strip().split(" ")[0])
        stage = 0
        pre_idx=10000
        ct=np.zeros((2,1,2*width-1))
        for i in range(2,len(lines)):
            line=lines[i].strip().split(" ")
            idx,kind=int(line[0]),int(line[1])
            if idx>pre_idx:
                stage+=1
                news = np.zeros((2,1,2*width-1))
                ct = np.concatenate((ct,news),axis=1)
                # print(ct.shape)
            pre_idx=idx
            if kind==1:
                ct[0][stage][idx] +=1
            else:
                ct[1][stage][idx] +=1 
        return ct    

    def decompose_compressor_tree(self, initial_partial_product, state):
        # 1. convert the current state to the EasyMac text file format, matrix to tensor
        next_state = np.zeros_like(state)
        next_state[0] = state[0]
        next_state[1] = state[1]
        stage_num = 0
        ct32 = np.zeros([1,int(self.int_bit_width*2)])
        ct22 = np.zeros([1,int(self.int_bit_width*2)])
        ct32[0] = next_state[0]
        ct22[0] = next_state[1]
        partial_products = np.zeros([1,int(self.int_bit_width*2)])
        partial_products[0] = initial_partial_product
        # decompose each column sequentially
        for i in range(1, int(self.int_bit_width*2)):
            j = 0 # j denotes the stage index, i denotes the column index
            while (j <= stage_num): # the condition is impossible to satisfy
                
                # j-th stage i-th column
                ct32[j][i] = next_state[0][i]
                ct22[j][i] = next_state[1][i]
                # initial j-th stage partial products
                if j == 0: # 0th stage
                    partial_products[j][i] = partial_products[j][i]
                else:
                    partial_products[j][i] = partial_products[j-1][i] + \
                        ct32[j-1][i-1] + ct22[j-1][i-1]

                # when to break 
                if (3*ct32[j][i] + 2*ct22[j][i]) <= partial_products[j][i]:
                    # print(f"i: {ct22[j][i]}, i-1: {ct22[j][i-1]}")
                    # update j-th stage partial products for the next stage
                    partial_products[j][i] = partial_products[j][i] - \
                        ct32[j][i]*2 - ct22[j][i]
                    # update the next state compressors
                    next_state[0][i] -= ct32[j][i]
                    next_state[1][i] -= ct22[j][i]
                    break # the only exit
                else:
                    if j == stage_num:
                        # print(f"j {j} stage num: {stage_num}")
                        # add initial next stage partial products and cts
                        stage_num += 1
                        ct32 = np.r_[ct32,np.zeros([1,int(self.int_bit_width*2)])]
                        ct22 = np.r_[ct22,np.zeros([1,int(self.int_bit_width*2)])]
                        partial_products = np.r_[partial_products,np.zeros([1,int(self.int_bit_width*2)])]
                    # assign 3:2 first, then assign 2:2
                    # only assign the j-th stage i-th column compressors
                    if (ct32[j][i] >= partial_products[j][i]//3):
                        ct32[j][i] = partial_products[j][i]//3
                        if (partial_products[j][i]%3 == 2):
                            if (ct22[j][i] >= 1):
                                ct22[j][i] = 1
                        else:
                            ct22[j][i] = 0
                    else:
                        ct32[j][i] = ct32[j][i]
                        if(ct22[j][i] >= (partial_products[j][i]-ct32[j][i]*3)//2):
                            ct22[j][i] = (partial_products[j][i]-ct32[j][i]*3)//2
                        else:
                            ct22[j][i] = ct22[j][i]
                    
                    # update partial products
                    partial_products[j][i] = partial_products[j][i] - ct32[j][i]*2 - ct22[j][i]
                    next_state[0][i] = next_state[0][i] - ct32[j][i]
                    next_state[1][i] = next_state[1][i] - ct22[j][i]
                j += 1
        
        # 2. write the compressors information into the text file
        sum = int(ct32.sum() + ct22.sum())
        file_name = os.path.join(self.build_path, f"compressor_tree_test_{self.task_index}.txt")
        with open(file_name, mode="w") as f:
            f.write(str(self.str_bit_width) + ' ' + str(self.str_bit_width))
            f.write('\n')
            f.write(str(sum))
            f.write('\n')
            for i in range(0, stage_num+1):
                for j in range(0, int(self.int_bit_width*2)):
                    # write 3:2 compressors
                    for k in range(0, int(ct32[i][int(self.int_bit_width*2)-1-j])):
                        f.write(str( int(self.int_bit_width*2)-1-j ))
                        f.write(' 1')
                        f.write('\n')
                    for k in range(0, int( ct22[i][int(self.int_bit_width*2)-1-j] )):
                        f.write(str( int(self.int_bit_width*2)-1-j ))
                        f.write(' 0')
                        f.write('\n')
        # print(f"stage num: {stage_num}")
                        
        # read ct and write verilog
        ct = self.read_ct(file_name)
        rtl_file = os.path.join(self.synthesis_path, 'rtl')
        if not os.path.exists(rtl_file):
            os.mkdir(rtl_file)
        rtl_file = os.path.join(rtl_file, "MUL.v")

        self.write_mul(
            rtl_file,
            math.ceil(self.int_bit_width),
            ct
        )

        return ct32, ct22, partial_products, stage_num

    def get_reward(self, n_processing=None, target_delays=None):
        # 1. Use the EasyMac to generate RTL files
        # compressor_file = os.path.join(self.build_path, f"compressor_tree_test_{self.task_index}.txt")
        # rtl_file = os.path.join(self.synthesis_path, 'rtl')
        # if not os.path.exists(rtl_file):
        #     os.mkdir(rtl_file)

        # 2. Use the RTL file to run openroad yosys
        ppas_dict = {
            "area": [],
            "delay": [],
            "power": []
        }
        if target_delays is None:
            n_processing = self.n_processing
            target_delays = self.target_delay

        with Pool(n_processing) as pool:
            def collect_ppa(ppa_dict):
                for k in ppa_dict.keys():
                    ppas_dict[k].append(ppa_dict[k])

            for i, target_delay in enumerate(target_delays):
                ys_path = os.path.join(self.synthesis_path, f"ys{i}")
                pool.apply_async(
                    func=RefineEnv.simulate_for_ppa,
                    args=(target_delay, ys_path, self.synthesis_path, self.synthesis_type),
                    callback=collect_ppa
                )

            pool.close()
            pool.join()
        
        return ppas_dict
    
    def step(self, action, is_model_evaluation=False, ppa_model=None):
        """
            action is a number, action coding:
                action=0: add a 2:2 compressor
                action=1: remove a 2:2 compressor
                action=2: replace a 3:2 compressor
                action=3: replace a 2:2 compressor
            Input: cur_state, action
            Output: next_state
        """

        # 1. given initial partial product and compressor tree state, can get the final partial product
            # 其实这个压缩的过程可以建模为两种情况：一种是并行压缩，就要分阶段；一种是从低位到高位的顺序压缩，就没有阶段而言，就是让每一列消消乐；能不能把这两种建模结合呢？为什么要结合这两种呢？优缺点在哪里？
        # 2. perform action，update the compressor tree state and update the final partial product
        # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        # 4. Evaluate the updated compressor tree state to get the reward
            # 上一个state的average ppa 和 当前state 的 average ppa 的差值

        action_column = int(action) // 4
        action_type = int(action) % 4
        initial_partial_product = PartialProduct[self.bit_width]
        state = self.cur_state
        # 1. compute final partial product from the lowest column to highest column
        final_partial_product = self.get_final_partial_product(initial_partial_product)

        # 2. perform action，update the compressor tree state and update the final partial product
        updated_partial_product = self.update_state(action_column, action_type, final_partial_product)
        # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        legalized_partial_product, legal_num_column = self.legalization(action_column, updated_partial_product)
        
        # legal_num_column = 0

        # 4. Decompose the compressor tree to multiple stages and write it to verilog
        next_state = copy.deepcopy(self.cur_state)
        ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product[:-1], next_state)
        next_state = copy.deepcopy(self.cur_state)
        # 5. Evaluate the updated compressor tree state to get the reward
        if self.is_debug:
            # do not go through openroad simulation
            reward = 0
            rewards_dict = {
                "area": 0,
                "delay": 0,
                "avg_ppa": 0,
                "last_state_ppa": 0,
                "legal_num_column": 0,
                "normalize_area": 0,
                "normalize_delay":0
            }
        elif self.reward_type == "simulate":
            rewards_dict = {}
            if is_model_evaluation:
                assert ppa_model is not None
                reward, avg_ppa, last_state_ppa, normalize_area, normalize_delay = self._model_evaluation(
                    ppa_model, ct32, ct22, stage_num
                )
                normalize_area_no_scale = 0
                normalize_delay_no_scale = 0
                area_reward = 0
                delay_reward = 0                
                rewards_dict['area'] = 0
                rewards_dict['delay'] = 0
            else:
                rewards_dict = self.get_reward()
                reward, avg_ppa, last_state_ppa, normalize_area_no_scale, normalize_delay_no_scale, area_reward, delay_reward, normalize_area, normalize_delay = self.process_reward(rewards_dict)
            rewards_dict['avg_ppa'] = avg_ppa
            rewards_dict['last_state_ppa'] = last_state_ppa
            rewards_dict['legal_num_column'] = legal_num_column
            rewards_dict['normalize_area_no_scale'] = normalize_area_no_scale
            rewards_dict['normalize_delay_no_scale'] = normalize_delay_no_scale
            rewards_dict['normalize_area'] = normalize_area
            rewards_dict['normalize_delay'] = normalize_delay
            rewards_dict['area_reward'] = area_reward
            rewards_dict['delay_reward'] = delay_reward
        elif self.reward_type == "node_num":
            ppa_estimation = next_state.sum()
            reward = self.last_ppa - ppa_estimation
            avg_ppa = ppa_estimation
            last_state_ppa = self.last_ppa
            rewards_dict = {
                "area": [0,0],
                "delay": [0,0],
                "avg_ppa": avg_ppa,
                "last_state_ppa": last_state_ppa,
                "legal_num_column": legal_num_column
            }
            self.last_ppa = ppa_estimation
        elif self.reward_type == "node_num_v2":
            ppa_estimation = 3 * ct32.sum() + 2 * ct22.sum()
            reward = self.last_ppa - ppa_estimation
            avg_ppa = ppa_estimation
            last_state_ppa = self.last_ppa
            rewards_dict = {
                "area": [0,0],
                "delay": [0,0],
                "avg_ppa": avg_ppa,
                "last_state_ppa": last_state_ppa,
                "legal_num_column": legal_num_column,
                "normalize_area": 0,
                "normalize_delay": 0
            }
            self.last_ppa = ppa_estimation
        elif self.reward_type == "ppa_model":
            ppa_estimation = self._predict_state_ppa(
                ct32, ct22, stage_num
            )
            reward = self.reward_scale * (self.last_ppa - ppa_estimation)
            avg_ppa = ppa_estimation
            last_state_ppa = self.last_ppa
            rewards_dict = {
                "area": [0,0],
                "delay": [0,0],
                "avg_ppa": avg_ppa,
                "last_state_ppa": last_state_ppa,
                "legal_num_column": legal_num_column
            }
            self.last_ppa = ppa_estimation
        # print(f"ct32: {ct32} shape: {ct32.shape}")
        # print(f"ct22: {ct22} shape: {ct22.shape}")

        return next_state, reward, rewards_dict

    def get_ppa_full_delay_cons(self, test_state):
        initial_partial_product = PartialProduct[self.bit_width]
        ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product[:-1], test_state)
        # generate target delay
        target_delay=[]
        input_width = math.ceil(self.int_bit_width)
        if input_width == 8:
            for i in range(50,1000,10):
                target_delay.append(i)
        elif input_width == 16:
            for i in range(50,2000,20):
                target_delay.append(i)
        elif input_width == 32: 
            for i in range(50,3000,20):
                target_delay.append(i)
        elif input_width == 64: 
            for i in range(50,4000,20):
                target_delay.append(i)
        #for file in os.listdir(synthesis_path): 
        n_processing = 12
        # config_abc_sta
        self.config_abc_sta(target_delay=target_delay)
        # get reward 并行 openroad
        ppas_dict = self.get_reward(n_processing=n_processing, target_delays=target_delay)

        return ppas_dict

    def legal_crossover_states(self, state, sel_column_index):
        # 1. get final partial product
        initial_partial_product = PartialProduct[self.bit_width]
        final_partial_product = np.zeros(initial_partial_product.shape[0]+1)
        if self.pp_encode_type == "booth":
            final_partial_product[0] = 2 # the first column must cotain two bits
        elif self.pp_encode_type == "and":
            final_partial_product[0] = 1 
        for i in range(1, int(self.int_bit_width*2)):
            final_partial_product[i] = initial_partial_product[i] + state[0][i-1] + \
                state[1][i-1] - 2 * state[0][i] - state[1][i]
        final_partial_product[int(self.int_bit_width*2)] = 0 # the last column 2*n+1 must contain 0 bits
        
        # 2. try to legalize if it exists
        legal_num_column = 0
        is_can_legal = True
        for i in range(sel_column_index, int(self.int_bit_width*2)):
            if final_partial_product[i] in [1, 2]:
                # it is legal, so break
                continue
            else:
                if final_partial_product[i] == 3:
                    # add a 3:2 compressor
                    state[0][i] += 1 
                    final_partial_product[i] = 1
                    final_partial_product[i+1] += 1
                elif final_partial_product[i] == 0:
                    # if 2:2 compressor exists, remove a 2:2
                    if state[1][i] >= 1:
                        state[1][i] -= 1
                        final_partial_product[i] += 1
                        final_partial_product[i+1] -= 1
                    # else: remove a 3:2
                    else:
                        state[0][i] -= 1
                        final_partial_product[i] += 2
                        final_partial_product[i+1] -= 1
                else:
                    is_can_legal = False
                    print(f"final partial product: {i} {final_partial_product[i]} num sel column {sel_column_index}")
                    break
            legal_num_column += 1
        print(f"legal num column: {legal_num_column}")
        print(f"legalized final partial product: {final_partial_product}")
        return state, is_can_legal

    def block_crossover(self, state1, state2):
        """
            input: two states
            output: two perturbed legalized states
        """
        num = 0
        while True:
            num += 1
            if num >= 20:
                print(f"warning!!! no valid block crossover in 20 steps")
                return None, None
            # 1. select a random column
            column_num = state1.shape[1]
            sel_column_index = np.random.choice(
                np.arange(column_num)
            )
            # 2. assert if equal before column index
            if np.array_equal(
                state1[:,:sel_column_index], state2[:,:sel_column_index]
            ):
                print(f"equal state before column index")
                continue 
            # 3. copy state
            cur_iteration_state1 = copy.deepcopy(state1)
            cur_iteration_state2 = copy.deepcopy(state2)

            # 3. crossover ct32 and ct22 in sel_column_index
            state1_block = cur_iteration_state1[:, sel_column_index:]
            state2_block = cur_iteration_state2[:, sel_column_index:]

            # print(f"state1 block: {state1_block}")
            # print(f"state2 block: {state2_block}")
            
            cur_iteration_state1[:, sel_column_index:] = state2_block
            cur_iteration_state2[:, sel_column_index:] = state1_block
                        
            # 3. legalize crossovered states
            legalized_state1, is_can_legal_state1 = self.legal_crossover_states(cur_iteration_state1, sel_column_index)
            legalized_state2, is_can_legal_state2 = self.legal_crossover_states(cur_iteration_state2, sel_column_index)
            if is_can_legal_state1 or is_can_legal_state2:
                print(f"sel column num: {sel_column_index}")
                break
            else:
                print(f"cannot change")

        if is_can_legal_state1 and is_can_legal_state2:
            return legalized_state1, legalized_state2
        elif is_can_legal_state1:
            return legalized_state1, None
        elif is_can_legal_state2:
            return None, legalized_state2
        else:
            return None, None

    def column_crossover(self, state1, state2):
        """
            input: two states
            output: two perturbed legalized states
        """
        num = 0
        while True:
            num += 1
            if num >= 20:
                print(f"warning!!! no valid column crossover in 20 steps")
                return None, None
            # 1. select a random column
            column_num = state1.shape[1]
            sel_column_index = np.random.choice(
                np.arange(column_num)
            )
            # 2. assert if equal before column index
            if np.array_equal(
                state1[:,:sel_column_index], state2[:,:sel_column_index]
            ):
                print(f"equal state before column index")
                continue 
            # 3. crossover ct32 and ct22 in sel_column_index
            ct32_state1 = int(state1[0, sel_column_index])
            ct22_state1 = int(state1[1, sel_column_index])
            ct32_state2 = int(state2[0, sel_column_index])
            ct22_state2 = int(state2[1, sel_column_index])
            # print(f"ct32 state1: {ct32_state1}")
            # print(f"ct22 state1: {ct22_state1}")
            # print(f"ct32 state2: {ct32_state2}")
            # print(f"ct22 state2: {ct22_state2}")
            if ct32_state1 == ct32_state2 and ct22_state1 == ct22_state2:
                print(f"equal state column")
                continue 
            
            # 4. copy state
            cur_iteration_state1 = copy.deepcopy(state1)
            cur_iteration_state2 = copy.deepcopy(state2)
            
            cur_iteration_state1[0, sel_column_index] = ct32_state2
            cur_iteration_state1[1, sel_column_index] = ct22_state2
            cur_iteration_state2[0, sel_column_index] = ct32_state1
            cur_iteration_state2[1, sel_column_index] = ct22_state1
            
            # 3. legalize crossovered states
            legalized_state1, is_can_legal_state1 = self.legal_crossover_states(cur_iteration_state1, sel_column_index)
            legalized_state2, is_can_legal_state2 = self.legal_crossover_states(cur_iteration_state2, sel_column_index)
            if is_can_legal_state1 or is_can_legal_state2:
                print(f"sel column num: {sel_column_index}")
                break
            else:
                print(f"cannot change")

        if is_can_legal_state1 and is_can_legal_state2:
            return legalized_state1, legalized_state2
        elif is_can_legal_state1:
            return legalized_state1, None
        elif is_can_legal_state2:
            return None, legalized_state2
        else:
            return None, None
    
class SpeedUpRefineEnvMultiObj(SpeedUpRefineEnv):
    def __init__(
            self, seed, q_policy,
            weight_list=[[4,1],[3,2],[2,3],[1,4]],
            gomil_area=1936,
            gomil_delay=1.35,
            load_gomil=True,
            **env_kwargs
    ):
        super(SpeedUpRefineEnvMultiObj, self).__init__(
            seed, q_policy, **env_kwargs
        )
        self.weight_list = weight_list
        # gomil kwargs
        self.gomil_area = gomil_area
        self.gomil_delay = gomil_delay
        self.load_gomil = load_gomil

        # reinitialize initial state pool
        if self.initial_state_pool_max_len > 0:
            self.initial_state_pool = [deque([],maxlen=self.initial_state_pool_max_len) for _ in range(len(self.weight_list))]
            self.imagined_initial_state_pool = [deque([],maxlen=self.initial_state_pool_max_len) for _ in range(len(self.weight_list))]

            # get wallace state information
            self.initial_wallace_state = copy.deepcopy(InitialState[self.bit_width])
            self.initial_gomil_state = copy.deepcopy(GOMILInitialState[self.bit_width])
            if self.q_policy is not None:
                initial_mask = self.get_state_mask_v2(self.q_policy, self.initial_wallace_state)
                initial_gomil_mask = self.get_state_mask_v2(self.q_policy, self.initial_gomil_state)
            
            for i, weights in enumerate(self.weight_list):
                wallace_ppa, wallace_normalize_area, wallace_normalize_delay = self._compute_ppa(
                    self.wallace_area, self.wallace_delay, weights=weights
                )
                gomil_ppa, gomil_normalize_area, gomil_normalize_delay = self._compute_ppa(gomil_area, gomil_delay, weights=weights)

                self.initial_state_pool[i].append(
                    {
                        "state": self.initial_wallace_state,
                        "area": self.wallace_area,
                        "delay": self.wallace_delay,
                        "state_mask": initial_mask,
                        "ppa": wallace_ppa,
                        "count": 1,
                        "state_type": "best_ppa",
                        "normalize_area": wallace_normalize_area,
                        "normalize_delay": wallace_normalize_delay
                    }
                )
                if self.load_gomil:
                    self.initial_state_pool[i].append(
                        {
                            "state": self.initial_gomil_state,
                            "area": self.gomil_area,
                            "delay": self.gomil_delay,
                            "state_mask": initial_gomil_mask,
                            "ppa": gomil_ppa,
                            "count": 1,
                            "state_type": "best_ppa",
                            "normalize_area": gomil_normalize_area,
                            "normalize_delay": gomil_normalize_delay
                        }
                    )
                self.imagined_initial_state_pool[i].append(
                    {
                        "state": self.initial_wallace_state,
                        "area": self.wallace_area,
                        "delay": self.wallace_delay,
                        "state_mask": initial_mask,
                        "ppa": wallace_ppa,
                        "count": 1,
                        "state_type": "best_ppa"
                    }
                )

    def get_state_mask_v2(self, policy, state):
        if self.is_policy_column:
            _, _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
        elif self.is_policy_seq:
            _, _, next_state_policy_info = policy.action(
                state
            )
            self.wallace_seq_state = next_state_policy_info['seq_state_pth']
            return next_state_policy_info['mask_pth']
        elif self.is_multi_obj:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 0,
                    deterministic=False,
                    is_softmax=False
                )
        elif self.is_multi_obj_condiiton:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 0,
                    [self.wallace_area, self.wallace_delay], self.target_delay[0] / 1500,
                    deterministic=False,
                    is_softmax=False
                )
        else:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
        return next_state_policy_info['mask']

    def _compute_ppa(self, area, delay, weights=[4,1]):
        normalize_area = self.ppa_scale * (area / self.wallace_area)
        normalize_delay = self.ppa_scale * (delay / self.wallace_delay)
        ppa = weights[0] * (area / self.wallace_area) + weights[1] * (delay / self.wallace_delay)
        ppa = self.ppa_scale * ppa

        return ppa, normalize_area, normalize_delay

    def select_state_from_pool(self, pool_index=0):
        sel_indexes = range(0, len(self.initial_state_pool[pool_index]))
        sel_index = random.sample(sel_indexes, 1)[0]
        initial_state = self.initial_state_pool[pool_index][sel_index]["state"]
        return initial_state, sel_index
    
    def reset(self, pool_index=0):
        initial_state, sel_index = self.select_state_from_pool(pool_index=pool_index)
        self.cur_state = copy.deepcopy(initial_state)
        self.last_area = self.initial_state_pool[pool_index][sel_index]["area"]
        self.last_delay = self.initial_state_pool[pool_index][sel_index]["delay"]
        self.last_ppa = self.initial_state_pool[pool_index][sel_index]["ppa"]
        self.last_normalize_area = self.initial_state_pool[pool_index][sel_index]["normalize_area"]
        self.last_normalize_delay = self.initial_state_pool[pool_index][sel_index]["normalize_delay"]
        return initial_state, sel_index

    def _model_evaluation(self, ppa_model, ct32, ct22, stage_num, pool_index=0):
        if self.is_sr_model:
            # call sr ppa model
            normalize_area, normalize_delay = self._call_sr_model(
                ppa_model, ct32, ct22, stage_num
            )
        else:
            # call nn ppa model
            normalize_area, normalize_delay = self._call_nn_model(
                ppa_model, ct32, ct22, stage_num
            )
        avg_ppa = self.weight_list[pool_index][0] * normalize_area + self.weight_list[pool_index][1] * normalize_delay
        
        # avg_ppa = avg_ppa * self.ppa_scale
        
        reward = self.last_ppa - avg_ppa
        area_reward = self.last_normalize_area - normalize_area
        delay_reward = self.last_normalize_delay - normalize_delay
        last_state_ppa = self.last_ppa
        # update last area delay
        self.last_ppa = avg_ppa
        self.last_normalize_area = normalize_area
        self.last_normalize_delay = normalize_delay
        return reward, avg_ppa, last_state_ppa, normalize_area, normalize_delay, area_reward, delay_reward

    def process_reward(self, rewards_dict, pool_index=0):
        avg_area = np.mean(rewards_dict['area'])
        avg_delay = np.mean(rewards_dict['delay'])
        # compute ppa
        avg_ppa, normalize_area, normalize_delay = self._compute_ppa(
            avg_area, avg_delay, weights=self.weight_list[pool_index]
        )
        # immediate reward
        reward = self.last_ppa - avg_ppa
        area_reward = self.last_normalize_area - normalize_area
        delay_reward = self.last_normalize_delay - normalize_delay
        # long-term reward
        long_term_reward = (self.weight_area + self.weight_delay) * self.ppa_scale - avg_ppa
        reward = reward + self.long_term_reward_scale * long_term_reward
        last_state_ppa = self.last_ppa
        # update last area delay
        self.last_area = avg_area
        self.last_delay = avg_delay
        self.last_ppa = avg_ppa
        self.last_normalize_area = normalize_area
        self.last_normalize_delay = normalize_delay
        # normalize_area delay
        normalize_area_no_scale, normalize_delay_no_scale = self._normalize_area_delay(
            avg_area, avg_delay
        )        
        return reward, avg_ppa, last_state_ppa, normalize_area_no_scale, normalize_delay_no_scale, area_reward, delay_reward, normalize_area, normalize_delay

    def step(self, action, is_model_evaluation=False, ppa_model=None, pool_index=0):
        """
            action is a number, action coding:
                action=0: add a 2:2 compressor
                action=1: remove a 2:2 compressor
                action=2: replace a 3:2 compressor
                action=3: replace a 2:2 compressor
            Input: cur_state, action
            Output: next_state
        """

        # 1. given initial partial product and compressor tree state, can get the final partial product
            # 其实这个压缩的过程可以建模为两种情况：一种是并行压缩，就要分阶段；一种是从低位到高位的顺序压缩，就没有阶段而言，就是让每一列消消乐；能不能把这两种建模结合呢？为什么要结合这两种呢？优缺点在哪里？
        # 2. perform action，update the compressor tree state and update the final partial product
        # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        # 4. Evaluate the updated compressor tree state to get the reward
            # 上一个state的average ppa 和 当前state 的 average ppa 的差值

        action_column = int(action) // 4
        action_type = int(action) % 4
        initial_partial_product = PartialProduct[self.bit_width]
        state = self.cur_state
        # 1. compute final partial product from the lowest column to highest column
        final_partial_product = self.get_final_partial_product(initial_partial_product)

        # 2. perform action，update the compressor tree state and update the final partial product
        updated_partial_product = self.update_state(action_column, action_type, final_partial_product)
        # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        legalized_partial_product, legal_num_column = self.legalization(action_column, updated_partial_product)
        
        # legal_num_column = 0

        # 4. Decompose the compressor tree to multiple stages and write it to verilog
        next_state = copy.deepcopy(self.cur_state)
        ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product[:-1], next_state)
        next_state = copy.deepcopy(self.cur_state)
        # 5. Evaluate the updated compressor tree state to get the reward
        if self.is_debug:
            # do not go through openroad simulation
            reward = 0
            rewards_dict = {
                "area": 0,
                "delay": 0,
                "avg_ppa": 0,
                "last_state_ppa": 0,
                "legal_num_column": 0,
                "normalize_area": 0,
                "normalize_delay":0
            }
        elif self.reward_type == "simulate":
            rewards_dict = {}
            if is_model_evaluation:
                assert ppa_model is not None
                reward, avg_ppa, last_state_ppa, normalize_area, normalize_delay, area_reward, delay_reward = self._model_evaluation(
                    ppa_model, ct32, ct22, stage_num, pool_index=pool_index
                )
                normalize_area_no_scale = 0
                normalize_delay_no_scale = 0
                area_reward = area_reward
                delay_reward = delay_reward                
                rewards_dict['area'] = 0
                rewards_dict['delay'] = 0
            else:
                rewards_dict = self.get_reward()
                reward, avg_ppa, last_state_ppa, normalize_area_no_scale, normalize_delay_no_scale, area_reward, delay_reward, normalize_area, normalize_delay = self.process_reward(rewards_dict, pool_index=pool_index)
            rewards_dict['avg_ppa'] = avg_ppa
            rewards_dict['last_state_ppa'] = last_state_ppa
            rewards_dict['legal_num_column'] = legal_num_column
            rewards_dict['normalize_area_no_scale'] = normalize_area_no_scale
            rewards_dict['normalize_delay_no_scale'] = normalize_delay_no_scale
            rewards_dict['normalize_area'] = normalize_area
            rewards_dict['normalize_delay'] = normalize_delay
            rewards_dict['area_reward'] = area_reward
            rewards_dict['delay_reward'] = delay_reward
        elif self.reward_type == "node_num":
            raise NotImplementedError
        elif self.reward_type == "node_num_v2":
            raise NotImplementedError
        elif self.reward_type == "ppa_model":
            raise NotImplementedError

        return next_state, reward, rewards_dict

class SerialSpeedUpRefineEnv(SpeedUpRefineEnv):
    def __init__(
            self, seed, q_policy,
            load_state_pool_path=None,
            pool_index=0,
            gomil_area=1936,
            gomil_delay=1.35,
            load_gomil=True,
            **env_kwargs
    ):
        super(SerialSpeedUpRefineEnv, self).__init__(
            seed, q_policy, **env_kwargs
        )
        self.found_best_info = {
            "found_best_ppa": mp.Manager().Value("d", 1000.0),
            "found_best_area": mp.Manager().Value("d", 1000.0),
            "found_best_delay": mp.Manager().Value("d", 1000.0)
        }
        self.gomil_area = gomil_area
        self.gomil_delay = gomil_delay
        self.load_gomil = load_gomil

        if self.initial_state_pool_max_len > 0:
            self.initial_wallace_state = copy.deepcopy(InitialState[self.bit_width])
            self.initial_gomil_state = copy.deepcopy(GOMILInitialState[self.bit_width])
            self.initial_state_pool = mp.Manager().list()
            if self.q_policy is not None:
                initial_mask = self.get_state_mask_v2(self.q_policy, self.initial_wallace_state)
                initial_gomil_mask = self.get_state_mask_v2(self.q_policy, self.initial_gomil_state)
            else:
                initial_mask = None
            
            gomil_ppa, gomil_normalize_area, gomil_normalize_delay = self._compute_ppa(gomil_area, gomil_delay)
            
            if load_state_pool_path is not None:
                self.load_state_pool_path = load_state_pool_path
                self.npy_pool = np.load(
                    self.load_state_pool_path, allow_pickle=True
                ).item()
                env_state_pool = self.npy_pool[f'{pool_index}-th env_initial_state_pool']
                for i in range(len(env_state_pool)):
                    self.initial_state_pool.append(env_state_pool[i])

            elif self.reward_type == "simulate":
                ppa, normalize_area, normalize_delay = self._compute_ppa(self.wallace_area, self.wallace_delay)
                self.initial_state_pool.append(
                    {
                        "state": self.initial_wallace_state,
                        "area": self.wallace_area,
                        "delay": self.wallace_delay,
                        "state_mask": initial_mask,
                        "ppa": ppa,
                        "count": 1,
                        "state_type": "best_ppa",
                        "normalize_area": normalize_area,
                        "normalize_delay": normalize_delay
                    }
                )
            if self.load_gomil:
                # append gomil state
                self.initial_state_pool.append(
                    {
                        "state": self.initial_gomil_state,
                        "area": self.gomil_area,
                        "delay": self.gomil_delay,
                        "state_mask": initial_gomil_mask,
                        "ppa": gomil_ppa,
                        "count": 1,
                        "state_type": "best_ppa",
                        "normalize_area": gomil_normalize_area,
                        "normalize_delay": gomil_normalize_delay
                    }
                )

    def get_state_mask_v2(self, policy, state):
        if self.is_policy_column:
            _, _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
        elif self.is_policy_seq:
            _, _, next_state_policy_info = policy.action(
                state
            )
            self.wallace_seq_state = next_state_policy_info['seq_state_pth']
            return next_state_policy_info['mask_pth']
        elif self.is_multi_obj:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 0,
                    deterministic=False,
                    is_softmax=False
                )
        elif self.is_multi_obj_condiiton:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 0,
                    [self.wallace_area, self.wallace_delay], self.target_delay[0] / 1500,
                    deterministic=False,
                    is_softmax=False
                )
        else:
            _, next_state_policy_info = policy.select_action(
                    torch.tensor(state), 0, 
                    deterministic=False,
                    is_softmax=False
                )
        return next_state_policy_info['mask']
    
    def update_env_initial_state_pool(self, state, rewards_dict, state_mask):
        if self.initial_state_pool_max_len > 0:
            found_best_ppa = self.ppa_scale * (self.weight_area * self.found_best_info['found_best_area'].value / self.wallace_area + self.weight_delay * self.found_best_info['found_best_delay'].value / self.wallace_delay)
            if found_best_ppa > rewards_dict['avg_ppa']:
            # if self.found_best_info['found_best_ppa'].value > rewards_dict['avg_ppa']:
                # push the best ppa state into the initial pool
                avg_area = np.mean(rewards_dict['area'])
                avg_delay = np.mean(rewards_dict['delay'])
                if len(self.initial_state_pool) >= self.initial_state_pool_max_len:
                    self.initial_state_pool.pop(0)
                self.initial_state_pool.append(
                    {
                        "state": copy.deepcopy(state),
                        "area": avg_area,
                        "delay": avg_delay,
                        "ppa": rewards_dict['avg_ppa'],
                        "count": 1,
                        "state_mask": state_mask,
                        "state_type": "best_ppa",
                        "normalize_area": rewards_dict["normalize_area"],
                        "normalize_delay": rewards_dict["normalize_delay"]
                    }
                )
        if self.found_best_info["found_best_ppa"].value > rewards_dict['avg_ppa']:
            self.found_best_info["found_best_ppa"].value = rewards_dict['avg_ppa']
            self.found_best_info["found_best_area"].value = np.mean(rewards_dict['area']) 
            self.found_best_info["found_best_delay"].value = np.mean(rewards_dict['delay'])

    def simulate_for_ppa_serial(self, target_delay, ys_path, synthesis_path, synthesis_type):
        if synthesis_type == "v1":
            ys_scripts_gen(target_delay, ys_path, synthesis_path)
        elif synthesis_type == "v2":
            ys_scripts_v2_gen(target_delay, ys_path, synthesis_path)
        elif synthesis_type == "v3":
            ys_scripts_v3_gen(target_delay, ys_path, synthesis_path)
        elif synthesis_type == "v5":
            ys_scripts_v5_gen(target_delay, ys_path, synthesis_path)
        ppa_dict = get_ppa(ys_path)

        return ppa_dict

    def get_reward(self, n_processing=None, target_delays=None):
        # 1. Use the EasyMac to generate RTL files
        """
            rtl generation commented
        """
        # compressor_file = os.path.join(self.build_path, f"compressor_tree_test_{self.task_index}.txt")
        # rtl_file = os.path.join(self.synthesis_path, 'rtl')
        # if not os.path.exists(rtl_file):
        #     os.mkdir(rtl_file)

        # rtl_generate_cmd = f'cd {EasyMacPath}' + ' \n'
        # if self.pp_encode_type == 'booth':
        #     rtl_generate_cmd = rtl_generate_cmd + f'sbt clean \'Test/runMain {self.mul_booth_file} ' + f'--compressor-file {compressor_file} ' + f'--rtl-path {rtl_file}' + '\'' + ' \n'
        # elif self.pp_encode_type == 'and':
        #     rtl_generate_cmd = rtl_generate_cmd + f'sbt clean \'Test/runMain {self.mul_booth_file} ' + f'--compressor-file {compressor_file} ' \
        #     + f'--prefix-adder-file benchmarks/16x16/ppa.txt ' + f'--rtl-path {rtl_file}' + '\'' + ' \n'
        # rtl_generate_cmd = rtl_generate_cmd + f'cd {self.initial_cwd_path}'

        # rtl_generate_cmd = f"java -cp {EasyMacPath} {self.mul_booth_file} " + f'--compressor-file {compressor_file} ' \
        #     + f'--prefix-adder-file {BenchmarkPath} ' + f'--rtl-path {rtl_file}'

        # rtl_generate_cmd = f"java -cp {EasyMacPath} {self.mul_booth_file} " + f'--compressor-file {compressor_file} ' \
        #     + f'--rtl-path {rtl_file}'
        # os.system(rtl_generate_cmd)
        # --batch -Dsbt.server.forcestart=true
        # 2. Use the RTL file to run openroad yosys
        ppas_dict = {
            "area": [],
            "delay": [],
            "power": []
        }
        if target_delays is None:
            target_delays = self.target_delay
            for i, target_delay in enumerate(target_delays):
                ys_path = os.path.join(self.synthesis_path, f"ys{i}")
                ppa_dict = self.simulate_for_ppa_serial(
                    target_delay, ys_path, self.synthesis_path, self.synthesis_type
                )
                for k in ppa_dict.keys():
                    ppas_dict[k].append(ppa_dict[k])
        else:
            with Pool(n_processing) as pool:
                def collect_ppa(ppa_dict):
                    for k in ppa_dict.keys():
                        ppas_dict[k].append(ppa_dict[k])

                for i, target_delay in enumerate(target_delays):
                    ys_path = os.path.join(self.synthesis_path, f"ys{i}")
                    pool.apply_async(
                        func=RefineEnv.simulate_for_ppa,
                        args=(target_delay, ys_path, self.synthesis_path, self.synthesis_type),
                        callback=collect_ppa
                    )

                pool.close()
                pool.join()
        return ppas_dict      
if __name__ == '__main__':
    env = SpeedUpRefineEnv(
        1, None, mul_booth_file="mul.test2", bit_width="16_bits_and",
        target_delay=[50,300,600,2000], initial_state_pool_max_len=20,
        wallace_area = ((517+551+703+595)/4), wallace_delay=((1.0827+1.019+0.9652+0.9668)/4),
        pp_encode_type='and', load_pool_index=3, reward_type="simulate",
        # load_initial_state_pool_npy_path='./outputs/2023-09-18/14-40-49/logger_log/test/dqn8bits_reset_v2_initialstate/dqn8bits_reset_v2_initialstate_2023_09_18_14_40_55_0000--s-0/itr_25.npy'
        load_initial_state_pool_npy_path='None', synthesis_type="v1", is_debug=False
    )
    state, _ = env.reset()
    print(f"before state: {state} shape: {state.shape}")
    next_state, reward, rewards_dict = env.step(torch.tensor([5]))
    print(f"next state: {next_state} shape: {next_state.shape}")
    # state = env.reset()
    print(f"rewards: {rewards_dict}")
    # print(reward)
    # print(rewards_dict)

    # 现在环境测试还差8位mul booth 的easymac 的Scala文件
