import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dw',default = 16, type=int,help='multiplier data width')
#parser.add_argument('--easymac_path',default = "/home/dzuo/RL-MAC/wallace_gen/easymac", type=str,help='easymac root path (Absolute Path)')
#parser.add_argument('--ct_file',default = "/home/dzuo/RL-MAC/syn/ct/ct_rl_8bit_1104.txt", type=str,help='easymac format compressor tree description file (Absolute Path)')
#parser.add_argument('--rtl_path',default="/home/dzuo/RL-MAC/wallace_gen/rtl_test", type=str,help ='easymac RTL generation dir path (Absolute Path)')
args = parser.parse_args()

#easymac_path = args.easymac_path
#ct_file = args.ct_file
#rtl_path = args.rtl_path
DW = args.dw
pp = np.zeros([1,DW*2])
ct32 = np.zeros([1,DW*2])
ct22 = np.zeros([1,DW*2])
target = np.zeros(DW*2)
for i in range(0,DW*2):
    target[i] = 2
for i in range(0,DW):
    pp[0][i] = i+1
for i in range(DW,DW*2-1):
    pp[0][i] = DW*2-1-i
stage_num = 0
while(True):
    for i in range(2,DW*2):
        if(pp[stage_num][i]%3 == 0):
            ct32[stage_num][i] = pp[stage_num][i]//3
            ct22[stage_num][i] = 0
        elif(pp[stage_num][i]%3 == 1):
            ct32[stage_num][i] = pp[stage_num][i]//3
            ct22[stage_num][i] = 0
        elif(pp[stage_num][i]%3 == 2):
            ct32[stage_num][i] = pp[stage_num][i]//3
            if stage_num == 0:
                ct22[stage_num][i] = 0
            else:
                ct22[stage_num][i] = 1
    stage_num = stage_num + 1
    pp = np.r_[pp,np.zeros([1,DW*2])]
    pp[stage_num][0] = pp[stage_num-1][0] - ct32[stage_num-1][0]*2 - ct22[stage_num-1][0]
    for i in range(1,DW*2): 
        pp[stage_num][i] = pp[stage_num-1][i] + ct32[stage_num-1][i-1] + ct22[stage_num-1][i-1]  - ct32[stage_num-1][i]*2 - ct22[stage_num-1][i]
    if (pp[stage_num] <= target).all():
        break
    else:
        ct32 = np.r_[ct32,np.zeros([1,DW*2])]
        ct22 = np.r_[ct22,np.zeros([1,DW*2])]
sum = ct32.sum() + ct22.sum()
sum = int(sum)

compressed_ct32 = np.sum(ct32, axis=0)
compressed_ct22 = np.sum(ct22, axis=0)
print(f"stage num: {stage_num}")
print(f"ct32: {np.array(compressed_ct32)} len ct32: {len(compressed_ct32)}")
print(f"ct22: {np.array(compressed_ct22)} len ct22: {len(compressed_ct22)}")

# ct_file = '/home/dzuo/RL-MAC/wallace_gen/ct/ct_wallace.txt'
# f = open(ct_file, mode = 'w')
# f.write(str(DW) + ' ' + str(DW))
# f.write('\n')
# f.write(str(sum))
# f.write('\n')
# for i in range(0,stage_num):
#     for j in range(0,DW*2):
#         for k in range(0,int(ct32[i][DW*2-1-j])): 
#             f.write(str(DW*2-1-j))
#             f.write(' 1')
#             f.write('\n')
#         for k in range(0,int(ct22[i][DW*2-1-j])): 
#             f.write(str(DW*2-1-j))
#             f.write(' 0')
#             f.write('\n')
#cmd1 = 'cd ' + easymac_path + ' \n' + 'sbt \'Test/runMain mul.test2 --compressor-file  /home/dzuo/RL-MAC/wallace_gen/ct/ct_wallace_' + str(DW) + 'bit.txt --prefix-adder-file ./benchmarks/16x16/ppa.txt  --rtl-path ' + rtl_path + '\''
#os.system(cmd1)
