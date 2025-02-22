import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dw',default = 16, type=int,help='multiplier data width')
parser.add_argument('--type',default ='and', type=str,help='encode type')
#parser.add_argument('--easymac_path',default = "/home/dzuo/RL-MAC/wallace_gen/easymac", type=str,help='easymac root path (Absolute Path)')
#parser.add_argument('--ct_file',default = "/home/dzuo/RL-MAC/syn/ct/ct_rl_8bit_1104.txt", type=str,help='easymac format compressor tree description file (Absolute Path)')
#parser.add_argument('--rtl_path',default="/home/dzuo/RL-MAC/wallace_gen/rtl_test", type=str,help ='easymac RTL generation dir path (Absolute Path)')
args = parser.parse_args()


DW = args.dw
type=args.type
pp = np.zeros([1,DW*2])
ct32 = np.zeros([1,DW*2])
ct22 = np.zeros([1,DW*2])
target = np.zeros(DW*2)
for i in range(0,DW*2):
    target[i] = 2
## 基于and的部分积点阵
if type == 'and':
    for i in range(0,DW):
        pp[0][i] = i+1
    for i in range(DW,DW*2-1):
        pp[0][i] = DW*2-1-i
else:
    if (DW%2==0):
        max=DW/2+1
    else:
        max=DW/2
    j=3
    pos1,pos2={},{}
    for i in range(0,DW+4):
        pos1[i]=1
    pos1[DW+4]=2
    for i in range(DW+5,DW*2,2):
        pos1[i]=j
        pos1[i+1]=j
        if j<max:
            j=j+1
    k=2
    for i in range(0,DW*2,2):
        pos2[i]=k
        pos2[i+1]=k
        if k<max:
            k=k+1
    for i in range(0,DW*2):
        pp[0][i]=pos2[i]-pos1[i]+1
currentPath = os.getcwd().replace('\\','/')    # 获取当前路径_
ct_file = currentPath+'/build1/Wallace/ct_'+str(DW)+'_'+str(type)+'.txt'
stage_num = 0
while(True):
    if type=='booth':
        for i in range(0,DW*2):
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
    else:
        for i in range(0,DW*2-1):
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
    if type=='and':
        pp[stage_num][DW*2-1]=2
    if (pp[stage_num] <= target).all():
        break
    else:
        ct32 = np.r_[ct32,np.zeros([1,DW*2])]
        ct22 = np.r_[ct22,np.zeros([1,DW*2])]
print(stage_num)
np.savetxt('./build/ct32.txt', np.sum(ct32,axis=0).reshape(1,2*DW), fmt="%d", delimiter=",")
np.savetxt('./build/ct22.txt', np.sum(ct22,axis=0).reshape(1,2*DW), fmt="%d", delimiter=",")
sum = ct32.sum() + ct22.sum()
sum = int(sum)
f = open(ct_file, mode = 'w')
f.write(str(DW) + ' ' + str(DW))
f.write('\n')
f.write(str(sum))
f.write('\n')
for i in range(0,stage_num):
    for j in range(0,DW*2):
        for k in range(0,int(ct32[i][DW*2-1-j])): 
            f.write(str(DW*2-1-j))
            f.write(' 1')
            f.write('\n')
        for k in range(0,int(ct22[i][DW*2-1-j])): 
            f.write(str(DW*2-1-j))
            f.write(' 0')
            f.write('\n')
