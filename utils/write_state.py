import numpy as np
def get_pp(DW,type):
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
    return pp,ct32,ct22
PartialProduct={}
InitialState={}

for DW in [8,16,32,64]:
    for type in ['and','booth']:
        pp,ct32,ct22=get_pp(DW,type)
        pp=pp[0]
        ct32=np.sum(ct32,axis=0)
        ct22=np.sum(ct22,axis=0)
        ct=np.vstack((ct32,ct22))
        PartialProduct[str(DW)+'_bits_'+type]=pp
        InitialState[str(DW)+'_bits_'+type]=ct
with open("./global_const.txt",'w') as f:
    f.write('InitialState={ \n')
    for key,value in InitialState.items():
        f.write("\t")
        f.write('"'+key+'": np.array([ \n')
        f.write("\t\t[")
        if 'and' in key:
            for i in range(len(value[0])-1):
                if i==len(value[1])-2:
                    f.write(str(int(value[0][i])))
                else:
                    f.write(str(int(value[0][i]))+',')
        else:
            for i in range(len(value[0])):
                if i==len(value[1])-1:
                    f.write(str(int(value[0][i])))
                else:
                    f.write(str(int(value[0][i]))+',')
        f.write(']\n')
        f.write("\t\t[")
        if 'and' in key:
            for i in range(len(value[1])-1):
                if i==len(value[1])-2:
                    f.write(str(int(value[1][i])))
                else:
                    f.write(str(int(value[1][i]))+',')
        else:
            for i in range(len(value[1])):
                if i==len(value[1])-1:
                    f.write(str(int(value[1][i])))
                else:
                    f.write(str(int(value[1][i]))+',')
            
        f.write(']\n')
        f.write("\t")
        f.write(']),\n') 
    f.write('}\n')
    f.write('PartialProduct={ \n')
    for key,value in PartialProduct.items():
        f.write("\t")
        f.write('"'+key+'": np.array([ \n')
        f.write("\t\t")
        for i in range(len(value)):
            f.write(str(int(value[i]))+',')
            if i==len(value)/2-1:
                f.write("\n\t\t")
        if 'booth' in key:
            f.write('0,')
        f.write('\n')
        f.write("\t")
        f.write(']),\n') 
    f.write('}')