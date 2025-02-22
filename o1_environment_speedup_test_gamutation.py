import argparse
import numpy as np
import torch
import os
import copy
import math
from multiprocessing import Pool
from o0_global_const import PartialProduct 
from o1_environment import RefineEnv
import time

from ipdb import set_trace

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
            # set_trace()
            j = 0 # j denotes the stage index, i denotes the column index
            while (j <= stage_num): # the condition is impossible to satisfy
                # set_trace()
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
        wallace_node_num = 3 * int(ct32.sum()) + 2 * int(ct22.sum())
        print(f"wallace_node_num: {wallace_node_num}")
        print(f"stage_num: {stage_num}")
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
        time_dict = {}
        time_dict["start"] = time.time()
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
        
        time_dict["yosys_synthesis_and_openroad_sta"] = time.time()
        synthesis_sta_time = time_dict["yosys_synthesis_and_openroad_sta"] - time_dict["start"]
        
        print(f"time analysis: synthesis_sta_time: {synthesis_sta_time}")

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
        # # 1. compute final partial product from the lowest column to highest column
        # final_partial_product = self.get_final_partial_product(initial_partial_product)

        # # 2. perform action，update the compressor tree state and update the final partial product
        # updated_partial_product = self.update_state(action_column, action_type, final_partial_product)
        # # 3. The updated final partial product may be invalid, so perform legalization to update the partial product and compressor tree state
        # legalized_partial_product, legal_num_column = self.legalization(action_column, updated_partial_product)
        
        legal_num_column = 0

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
                "legal_num_column": legal_num_column
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
            for i in range(50,2000,10):
                target_delay.append(i)
        elif input_width == 32: 
            for i in range(50,3000,10):
                target_delay.append(i)
        elif input_width == 64: 
            for i in range(50,4000,10):
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
        set_trace()
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
        set_trace()
        return state, is_can_legal

    def block_crossover(self, state1, state2):
        """
            input: two states
            output: two perturbed legalized states
        """
        while True:
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
            state1_block = state1[:, sel_column_index:]
            state2_block = state2[:, sel_column_index:]

            # print(f"state1 block: {state1_block}")
            # print(f"state2 block: {state2_block}")
            
            state1[:, sel_column_index:] = state2_block
            state2[:, sel_column_index:] = state1_block
                        
            # 3. legalize crossovered states
            legalized_state1, is_can_legal_state1 = self.legal_crossover_states(state1, sel_column_index)
            legalized_state2, is_can_legal_state2 = self.legal_crossover_states(state2, sel_column_index)
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

    def column_crossover(self, state1, state2):
        """
            input: two states
            output: two perturbed legalized states
        """
        while True:
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
            
            state1[0, sel_column_index] = ct32_state2
            state1[1, sel_column_index] = ct22_state2
            state2[0, sel_column_index] = ct32_state1
            state2[1, sel_column_index] = ct22_state1
            
            # 3. legalize crossovered states
            legalized_state1, is_can_legal_state1 = self.legal_crossover_states(state1, sel_column_index)
            legalized_state2, is_can_legal_state2 = self.legal_crossover_states(state2, sel_column_index)
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

    def evaluate(self, test_state):
        # get avg ppa
        initial_partial_product = PartialProduct[self.bit_width]
        ct32, ct22, partial_products, stage_num = self.decompose_compressor_tree(initial_partial_product[:-1], test_state)
        rewards_dict = self.get_reward()
        avg_area = np.mean(rewards_dict['area'])
        avg_delay = np.mean(rewards_dict['delay'])
        avg_ppa = self.weight_area * avg_area / self.wallace_area + \
            self.weight_delay * avg_delay / self.wallace_delay
        avg_ppa = avg_ppa * self.ppa_scale
        normalize_area = self.ppa_scale * (avg_area / self.wallace_area)
        normalize_delay = self.ppa_scale * (avg_delay / self.wallace_delay)

        return avg_area, avg_delay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test GA Mutation')
    parser.add_argument(
        '--bit_width', type=str, default='16_bits_and') # image_state_reward

    args = parser.parse_args()
    if args.bit_width == "16_bits_and":
        # v1 脚本 加速Verilog
        npy_data_path = "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-03-12/14-08-16/logger_log/dqn_16bits_factor_action/dqn16bits_reset_rnd_factor_speedup_v1syn/dqn16bits_reset_rnd_factor_speedup_v1syn_2024_03_12_14_08_22_0000--s-1/itr_10000.npy"
        synthesis_type = "v1"
    elif args.bit_width == "32_bits_and":
        # v1 脚本 加速verilog
        npy_data_path = "/datasets/ai4multiplier/rl-mul-code/reconstruct_code/outputs/2024-04-08/10-14-43/logger_log/dqn_rnd_reset_factor_action/dqn32bits_reset_rnd_factor_speedup_v1syn/dqn32bits_reset_rnd_factor_speedup_v1syn_2024_04_08_10_14_49_0000--s-1/itr_10000.npy"
        synthesis_type = "v1"
    
    # dataset = np.load(npy_data_path, allow_pickle=True).item()
    # env_pool = dataset["env_initial_state_pool"]

    # area_list = []
    # delay_list = []
    # area_list.append(env_pool[-1]["area"])
    # delay_list.append(env_pool[-1]["delay"])
    # area_list.append(env_pool[0]["area"])
    # delay_list.append(env_pool[0]["delay"])
    
    env = SpeedUpRefineEnv(
        1, None, mul_booth_file="mul.test2", bit_width=args.bit_width,
        target_delay=[50,300,600,2000], initial_state_pool_max_len=20,
        wallace_area = ((517+551+703+595)/4), wallace_delay=((1.0827+1.019+0.9652+0.9668)/4),
        pp_encode_type='and', load_pool_index=3, reward_type="simulate",
        load_initial_state_pool_npy_path='None', synthesis_type="v1", is_debug=False
    )

    # for _ in range(100):
    #     state1 = copy.deepcopy(
    #         env_pool[-1]["state"]
    #     ) 
    #     state2 = copy.deepcopy(
    #         env_pool[0]["state"]
    #     )
    #     mutation_state1, mutation_state2 = env.block_crossover(
    #         state1, state2
    #     )
    #     if mutation_state1 is not None:
    #         area, delay = env.evaluate(mutation_state1)
    #         area_list.append(area)
    #         delay_list.append(delay)
    #     if mutation_state2 is not None:
    #         area, delay = env.evaluate(mutation_state2)
    #         area_list.append(area)
    #         delay_list.append(delay)
    
    # print(f"area mean: {np.mean(area_list)}, min: {np.min(area_list)} std: {np.std(area_list)}")
    # print(f"area list: {area_list}")

    # print(f"delay mean: {np.mean(delay_list)}, min: {np.min(delay_list)} std: {np.std(delay_list)}")
    # print(f"delay list: {delay_list}")
    
    #### test outlier state ####    
    # state = [
    #     [0,0,1,2,3,3,4,6,6,7,8,10,10,11,13,12,13,13,12,11,10,9,8,6,6,5,3,3,1,1,0],
    #     [0,1,1,1,0,3,3,2,3,3,3,1,2,2,0,3,2,1,1,1,1,1,1,3,2,2,4,3,5,4,4]
    #     ]

    state = [[0,0,1,1,2,3,6,5,6,8,8,10,10,10,13,12,13,13,12,11,10,9,8,6,6,5,3,3,1,1,0],
        [0,1,0,1,1,1,0,2,2,1,2,0,1,3,0,3,2,1,1,1,1,1,1,3,2,2,4,3,5,4,4]]
    
    state, is_can_legal = env.legal_crossover_states(state, 6)
    print(f"state: {state}")
    area, delay = env.evaluate(state)
    #### test outlier state ####

