
read_lef /home/qiu/work/graduation_project/dataset/NangateOpenCellLibrary.lef
read_lib /home/qiu/work/graduation_project/dataset/NangateOpenCellLibrary_typical.lib
read_verilog ysh_output/32bits_and_0/32bits_and_1/worker_0/netlist.v
link_design MUL

set period 5
create_clock -period $period [get_ports clock]

set clk_period_factor .2

set clk [lindex [all_clocks] 0]
set period [get_property $clk period]
set delay [expr $period * $clk_period_factor]
set_input_delay $delay -clock $clk [delete_from_list [all_inputs] [all_clocks]]
set_output_delay $delay -clock $clk [delete_from_list [all_outputs] [all_clocks]]

set_max_delay -from [all_inputs] 0
set critical_path [lindex [find_timing_paths -sort_by_slack] 0]
set path_delay [sta::format_time [[$critical_path path] arrival] 4]
puts "wns $path_delay"
report_design_area

set_power_activity -input -activity 0.5
report_power

set nets [get_nets C0/data*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}

set nets [get_nets C0/out*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}

set nets [get_nets out*]
foreach net $nets {
    puts "net_name [get_property $net name]"
    set pins [get_pins -of_objects $net]
    foreach pin $pins {
        puts "pin [get_property $pin full_name] [get_property $pin direction] [get_property $pin activity]"
    }
}

foreach inst [get_cells C0/FA*] {
    report_power -instance $inst
}

foreach inst [get_cells C0/HA*] {
    report_power -instance $inst
}

exit
