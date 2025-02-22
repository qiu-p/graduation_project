import os
from multiprocessing import Pool
lef = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary.lef'
lib = '/datasets/ai4multiplier/openroad_deb/leflib/NangateOpenCellLibrary_typical.lib'
OpenRoadFlowPath = '/datasets/ai4multiplier/openroad_deb/OpenROAD-flow-scripts'
def abc_constr_gen(ys_path):
    abc_constr_file = os.path.join(ys_path, 'abc_constr')
    with open(abc_constr_file, 'w') as f:
        f.write('set_driving_cell BUF_X1')
        f.write('\n')
        f.write('set_load 10.0 [all_outputs]')
        f.write('\n')
def sta_scripts_gen(ys_path):
    sta_file = os.path.join(ys_path, 'openroad_sta.tcl')
    with open(sta_file, 'w') as f:
        f.write('read_lef ' + str(lef))
        f.write('\n')
        f.write('read_lib ' + str(lib))
        f.write('\n')
        f.write('read_verilog ./netlist.v')
        f.write('\n')
        f.write('link_design MUL')
        f.write('\n')
        f.write('set_max_delay -from [all_inputs] 0')
        f.write('\n')
        f.write('set critical_path [lindex [find_timing_paths -sort_by_slack] 0]')
        f.write('\n')
        f.write('set path_delay [sta::format_time [[$critical_path path] arrival] 4]')
        f.write('\n')
        f.write('puts \"wns $path_delay\"')
        #f.write('report_wns')
        f.write('\n')
        f.write('report_design_area')
        f.write('\n')
        f.write('exit')
def ys_scripts_gen(target_delay, ys_path, synthesis_path):
    ys_file = os.path.join(ys_path, "syn_with_target_delay.ys")
    with open(ys_file, 'w') as f:
        f.write('read -sv ' + f'{synthesis_path}rtl/MUL.v')
        f.write('\n')
        f.write('synth -top MUL')
        f.write('\n')
        f.write('dfflibmap -liberty ' + str(lib))
        f.write('\n')
        f.write('abc -D ')
        f.write(str(target_delay))
        f.write(' -constr ./abc_constr -liberty ' + str(lib))
        #f.write(' -liberty ' + str(lib))
        f.write('\n')
        f.write('write_verilog ./netlist.v')
def get_ppa(ys_path):  
    synthesis_cmd = 'cd ' + ys_path + ' \n' + f'source {OpenRoadFlowPath}/env.sh\n' + 'yosys ./syn_with_target_delay.ys' # open road synthesis
    sta_cmd = 'cd ' + ys_path + ' \n' + f'source {OpenRoadFlowPath}/env.sh\n' + 'openroad openroad_sta.tcl | tee ./log' # openroad sta
    rm_log_cmd = 'rm -f ' + ys_path + '/log'
    rm_netlist_cmd = 'rm -f ' + ys_path + '/netlist.v'
    
    # execute synthesis cmd
    print(synthesis_cmd)
    print(sta_cmd)
    os.system(synthesis_cmd)  
    os.system(sta_cmd)
    # get ppa
    with open(ys_path + '/log', 'r') as f:
        rpt = f.read().splitlines()
        for line in rpt:
            if len(line.rstrip()) < 2:
                continue
            line = line.rstrip().split()
            if line[0] == 'wns':
                delay = line[-1]
                #delay = delay[1:]
                continue
            if line[0] == 'Design':
                area = line[2]
                break
    ppa_dict = {
        "area": float(area),
        "delay": float(delay)
    }
    # remove log
    #os.system(rm_log_cmd)
    #os.system(rm_netlist_cmd)
    
    return ppa_dict
def config_abc_sta(synthesis_path):
    # generate a config dir for each target delay
    for i in range(len(target_delay)):
        ys_path = os.path.join(synthesis_path, f"ys{i}")
        print(ys_path)
        if not os.path.exists(ys_path):
            os.mkdir(ys_path)
        abc_constr_gen(ys_path)
        sta_scripts_gen(ys_path)
def simulate_for_ppa(target_delay, ys_path, synthesis_path):
    ys_scripts_gen(target_delay, ys_path, synthesis_path)
    ppa_dict = get_ppa(ys_path)
    return ppa_dict
def get_reward(n_processing,synthesis_path,target_delay):
    ppas_dict = {
        "area": [],
        "delay": [],
        "power": []
    }
    '''
    def collect_ppa(ppa_dict):
        for k in ppa_dict.keys():
            ppas_dict[k].append(ppa_dict[k])
    for i in range(n_processing):
        ys_path = os.path.join(synthesis_path, f"ys{i}")
        collect_ppa(simulate_for_ppa(target_delay[i],ys_path,synthesis_path))
    return ppas_dict
    '''
    with Pool(n_processing) as pool:
        def collect_ppa(ppa_dict):
            for k in ppa_dict.keys():
                ppas_dict[k].append(ppa_dict[k])
        for i, target_delay in enumerate(target_delay):
            ys_path = os.path.join(synthesis_path, f"ys{i}")
            pool.apply_async(
                func=simulate_for_ppa,
                args=(target_delay, ys_path,synthesis_path),
                callback=collect_ppa
            )
        pool.close()
        pool.join()
    return ppas_dict
if __name__ == '__main__':
    MUL_SIZE=32

    type='booth'
    synthesis_path='/ai4multiplier/GOMIL/'+str(MUL_SIZE)+'_'+str(MUL_SIZE)+'_'+type+'/'
    #synthesis_path='/ai4multiplier/dqn_8_bits_and/'
    '''
    target_delay=[50,300,600,2000]
    n_processing=len(target_delay)
    config_abc_sta(synthesis_path)
    
    # {'area': [7429.0, 11074.0, 7429.0, 7429.0], 'delay': [1.4757, 1.3519, 1.4757, 1.4757], 'power': []}
    
    ppas_dict=get_reward(n_processing,synthesis_path,target_delay)
    print(ppas_dict)
    '''
    target_delay=[]
    for i in range(50,2000,10):
        target_delay.append(i)
    n_processing=len(target_delay)
    config_abc_sta(synthesis_path)
    ppas_dict=get_reward(n_processing,synthesis_path,target_delay)
    
    # [(7429.0, 1.4757), (10908.0, 1.3735), (11010.0, 1.3567), (10847.0, 1.3776), (10938.0, 1.3617), (7493.0, 1.4702), (10875.0, 1.3735), (11074.0, 1.3519)]
    pareto=[]
    with open("./GOMIL/build/ppa"+str(MUL_SIZE)+'_'+str(MUL_SIZE)+'_'+type+'.txt','w') as f:
        f.write('delay'+'\n')
        for i in range(len(target_delay)):
            f.write(str(ppas_dict['delay'][i])+'\n')
        f.write('area'+'\n')
        for i in  range(len(target_delay)):
            f.write(str(ppas_dict['area'][i])+'\n')

    for i in range(n_processing):
        mark=0
        for j in range(n_processing):
            if ppas_dict["area"][j]<=ppas_dict["area"][i] and ppas_dict["delay"][j]<=ppas_dict["delay"][i]:
                if ppas_dict["area"][j]==ppas_dict["area"][i] and ppas_dict["delay"][j]==ppas_dict["delay"][i]:
                    continue
                else:
                    mark=1
                    break
        if mark==0:
            if (ppas_dict['area'][i],ppas_dict['delay'][i]) not in pareto: 
                pareto.append((ppas_dict['area'][i],ppas_dict['delay'][i]))
    print(pareto)
    with open("./GOMIL/build/pareto"+str(MUL_SIZE)+'_'+str(MUL_SIZE)+'_'+type+'.txt','w') as f:
        f.write('delay'+'\n')
        for i in range(len(pareto)):
            f.write(str(pareto[i][1])+'\n')
        f.write('area'+'\n')
        for i in  range(len(pareto)):
            f.write(str(pareto[i][0])+'\n')