#include "Vtpu_like.h"
#include "Vtpu_like_seq_mac.h"
#include "tpu_driver.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <random>

int main(int argc, char **argv, char **env)
{
    Verilated::commandArgs(argc, argv);
    auto top = std::make_shared<Vtpu_like>();
    auto tfp = std::make_shared<VerilatedVcdC>();
    Verilated::traceEverOn(true);
    // top->trace(tfp.get(), 99);
    top->trace(tfp.get(), 1e8);
    tfp->open("./simulation.vcd");
    // tfp->open("./simulation-1000-steps.vcd");

    auto p_tpu = std::make_shared<TPUDriver>(top, tfp);

    // std::cout << "reset tpu" << std::endl;

    p_tpu->pipeline();

    std::cout << "jobs done" << std::endl;
    // 清理
    tfp->close();
    top.reset();

    return 0;
}
