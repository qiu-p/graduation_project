#pragma once
#include "Vtpu_like.h"
#include "verilated_vcd_c.h"
#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>

class TPUDriver
{
   public:
    std::shared_ptr<Vtpu_like> p_top;
    std::shared_ptr<VerilatedVcdC> p_tfp;

    int time_tick = 0;
    bool trace_on_flag = true;

    TPUDriver(std::shared_ptr<Vtpu_like> p_top_,
              std::shared_ptr<VerilatedVcdC> p_tfp_)
        : p_top(p_top_), p_tfp(p_tfp_){};

    int send_data(uint32_t *memory_a, int memory_size_a, uint32_t *memory_b,
                  int memory_size_b);

    void reset();
    void cycle();

    void enable_matmul();
    void start_tpu();
    void wait_to_execute();
    void clear_after_execution();
    void set_validity_mask();
    void set_data_address();
    void set_data_address_random();
    void set_data_address_stride();

    int pipeline();
};
