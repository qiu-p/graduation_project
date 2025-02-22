cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0074 NEW)
project(benchmark)
find_package(verilator HINTS $ENV{VERILATOR_ROOT} ${VERILATOR_ROOT})
if (NOT verilator_FOUND)
  message(FATAL_ERROR "Verilator was not found. Either install it, or set the VERILATOR_ROOT environment variable")
endif()

# verilate文件夹 的也加进来
add_subdirectory(verilate)

# tpu like small
add_executable(
  tpu_small
  benchmarks/sim/tpu_like_small.cpp
  benchmarks/sim/tpu_driver.cpp
)
verilate(
  tpu_small COVERAGE TRACE
    SOURCES benchmarks/tpu_like/tpu_like.small.os.v benchmarks/MUL/MUL-8.v
    TOP_MODULE top
    VERILATOR_ARGS  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE
)

# tpu like large
add_executable(
  tpu_large
  benchmarks/sim/tpu_like_large.cpp
  benchmarks/sim/tpu_driver_large.cpp
)
verilate(
  tpu_large COVERAGE TRACE
    SOURCES benchmarks/tpu_like/tpu_like.large.os.v benchmarks/MUL/MUL-8.v
    TOP_MODULE top
    VERILATOR_ARGS  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE
)