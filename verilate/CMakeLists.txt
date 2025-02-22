# cmake_minimum_required(VERSION 3.15)
# cmake_policy(SET CMP0074 NEW)
# project(cmake_example)
find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
if (NOT verilator_FOUND)
  message(FATAL_ERROR "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable")
endif()

add_executable(VMAIN sim_main.cpp)
verilate(
    VMAIN
    # SOURCES ../temp_sta/MUL-adder-modified.v
    SOURCES ../pybuild/debug/multiple_routing/f1_h0/MUL.v
    TOP_MODULE MUL
)

add_executable(VMAIN_ADDER sim_main_adder.cpp)
verilate(
  VMAIN_ADDER
    # SOURCES ../temp_sta/MUL-adder-modified.v
    SOURCES /home/xiaxilin/MiraLab/NMI/ai4-multiplier-master/ks-adder-16.v
    TOP_MODULE PrefixAdder
)