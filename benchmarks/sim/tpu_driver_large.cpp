#include "tpu_driver_large.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

void TPUDriver::reset()
{
    this->p_top->reset = 1;
    this->p_top->resetn = 0;

    this->cycle();

    this->p_top->reset = 0;
    this->p_top->resetn = 1;

    this->cycle();
}

void TPUDriver::cycle()
{
    this->p_top->clk = !this->p_top->clk;
    this->p_top->clk_mem = !this->p_top->clk_mem;
    this->p_top->eval();

    this->p_top->clk = !this->p_top->clk;
    this->p_top->clk_mem = !this->p_top->clk_mem;
    this->p_top->eval();

    /* 记录到 vcd 中 */
    if (this->trace_on_flag)
    {
        this->p_tfp->dump(this->time_tick);
        this->time_tick += 1;
    }
}

int TPUDriver::send_data(uint32_t *memory_a, int memory_size_a,
                         uint32_t *memory_b, int memory_size_b)
{
    this->trace_on_flag = false;
    this->p_top->PENABLE = 0;

    this->p_top->bram_we_a_ext = 0xffff'ffff;
    this->p_top->bram_we_b_ext = 0xffff'ffff;

    this->cycle();
    this->cycle();

    // 设置内存 a 和 b 的数据
    int random_index = 0;
    for (int address = 0; address < 1024; ++address)
    {
        for (int i = 0; i < this->p_top->bram_wdata_a_ext.Words; ++i)
        {
            this->p_top->bram_wdata_a_ext[i] = memory_a[random_index];
        }
        random_index += 1;

        for (int i = 0; i < this->p_top->bram_wdata_a_ext.Words; ++i)
        {
            this->p_top->bram_wdata_b_ext[i] = memory_b[random_index];
        }
        this->p_top->bram_addr_a_ext = address;
        this->p_top->bram_addr_b_ext = address;
        this->cycle();
        this->cycle();
        random_index += 1;
        if (random_index >= memory_size_b || random_index >= memory_size_a)
        {
            random_index = 0;
        }
    }
    this->trace_on_flag = true;

    this->cycle();
    this->cycle();

    return 0;
}

void TPUDriver::enable_matmul()
{
    this->p_top->PADDR = 0x0000;
    this->p_top->PWDATA = 1;
    this->p_top->PWRITE = 1;

    this->cycle();
    this->cycle();
}

// void TPUDriver::start_tpu()
// {
//     this->p_top->PADDR = 0x0004;
//     this->p_top->PWDATA = 1;
//     this->p_top->PWRITE = 1;

//     this->cycle();
//     this->cycle();
// }

void TPUDriver::start_tpu()
{
    this->p_top->PADDR = 0x0004;
    this->p_top->PWRITE = 1;
    this->p_top->PWDATA = 1;
    this->cycle();
    this->cycle();
}

void TPUDriver::wait_to_execute()
{

    for (int i = 0; i < 200; ++i)
    {
        this->p_top->PADDR = 0x0004;
        this->p_top->PWRITE = 0;
        this->cycle();
        this->cycle();
        if (((this->p_top->PRDATA) & (((long)1) << 31)) != 0)
        {
            break;
        }
    }
    // this->set_data_address_random();
}

void TPUDriver::clear_after_execution()
{
    this->p_top->PADDR = 0x0004;
    this->p_top->PWDATA = 0;
    this->p_top->PWRITE = 1;
    this->cycle();
    this->cycle();
}

void TPUDriver::set_validity_mask()
{
    this->p_top->PWRITE = 1;
    this->p_top->PWDATA = 0xffff'ffff;

    this->p_top->PADDR = 0x0020;
    this->cycle();
    this->cycle();

    this->p_top->PADDR = 0x0054;
    this->cycle();
    this->cycle();

    this->p_top->PADDR = 0x005c;
    this->cycle();
    this->cycle();

    this->p_top->PADDR = 0x0058;
    this->cycle();
    this->cycle();
}

void TPUDriver::set_data_address()
{
    this->p_top->PWRITE = 1;

    /* address a */
    this->p_top->PADDR = 0x000e;
    this->p_top->PWDATA = 0x0000'0000;

    this->cycle();
    this->cycle();

    /* address b */
    this->p_top->PADDR = 0x0012;
    this->p_top->PWDATA = 0x0000'0000;

    this->cycle();
    this->cycle();

    /* address c */
    this->p_top->PADDR = 0x0016;
    this->p_top->PWDATA = 0x0000'0fff;

    this->cycle();
    this->cycle();
}

void TPUDriver::set_data_address_random()
{
    this->p_top->PWRITE = 1;

    /* address a */
    this->p_top->PADDR = 0x000e;
    this->p_top->PWDATA = rand();
    this->cycle();
    this->cycle();

    // /* address b */
    this->p_top->PADDR = 0x0012;
    this->p_top->PWDATA = rand();
    this->cycle();
    this->cycle();
}

void TPUDriver::set_data_address_stride()
{
    this->p_top->PWRITE = 1;

    /* stride a */
    this->p_top->PADDR = 0x0028;
    this->p_top->PWDATA = 0x0000'007f;  // 8 x 16
    this->cycle();
    this->cycle();

    /* stride b */
    this->p_top->PADDR = 0x0032;
    this->p_top->PWDATA = 0x0000'007f;  // 8 x 16
    this->cycle();
    this->cycle();

    /* stride c */
    this->p_top->PADDR = 0x0036;
    this->p_top->PWDATA = 0x0000'007f;  // 8 x 16
    this->cycle();
    this->cycle();
}

int TPUDriver::pipeline()
{
    std::cout << "reset tpu " << std::endl;
    this->reset();

    uint32_t memory_a[100];
    uint32_t memory_b[100];

    for (int i = 0; i < 100; ++i)
    {
        memory_a[i] = rand();
        memory_b[i] = rand();
    }
    std::cout << "sending data " << std::endl;

    this->send_data(memory_a, 100, memory_b, 100);

    this->p_top->PENABLE = 1;
    this->p_top->PSEL = 1;

    this->enable_matmul();

    this->set_validity_mask();
    this->set_data_address();
    this->set_data_address_stride();

    // for (int i = 0; i < 1000; ++i)
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "start term " << i << std::endl;
        this->start_tpu();
        this->wait_to_execute();
        this->clear_after_execution();
    }

    return 0;
}
