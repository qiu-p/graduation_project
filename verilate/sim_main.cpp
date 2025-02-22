// #include "VMUL_101.h"
// #include "VMUL_opt.h"
#include "VMUL.h"
// #include "VMUL_adder_modified.h"
#include "verilated.h"
#include <iostream>
#include <memory.h>
#include <memory>
#include <random>

// #define INPUT_WIDTH 8
// #define TEST_INPUT_WIDTH 8
// #define OUTPUT_WIDTH 15

#define INPUT_WIDTH 16
#define TEST_INPUT_WIDTH 16
#define OUTPUT_WIDTH 31

int main() {
  // auto top = std::make_shared<VMUL_LUT>();
  // auto top = std::make_shared<VMUL_opt>();
  auto top = std::make_shared<VMUL>();
  // auto top = std::make_shared<VMUL_adder_modified>();
  bool flag = true;
  int cnt = 0;
  std::random_device rd;  // 随机数种子
  std::mt19937 gen(rd()); // 使用 Mersenne Twister 19937 算法
  std::uniform_int_distribution<uint32_t> dis(
      0,
      (((long long)1) << TEST_INPUT_WIDTH) - 1); // 生成 0-255 之间的随机数
  for (long long i = 0; i < 1e6; i += 1) {
    cnt += 1;
    if (cnt % 100 == 0) {
      std::cout << "testing " << cnt << std::endl;
    }
    unsigned long long a = dis(gen);
    unsigned long long b = dis(gen);
    top->a = a;
    top->b = b;
    // 评估模块
    top->eval();

    // 打印输出信号的值
    // long long out = top->s.at(0);
    unsigned long long out = top->out;
    unsigned long long ground_truth = ((a * b) & (((long)1 << OUTPUT_WIDTH) - 1));
    if (out != ground_truth) {
      flag = false;
      printf("a = %lld, b = %lld, output out is %lld, true value is "
             "%lld\n",
             a, b, out, ground_truth);
    } else {
      // printf("ture for i = %ld, j = %ld\n", i, j);
    }
  }
  // }

  if (flag) {
    std::cout << "All " << cnt << " tests passed" << std::endl;
  }
  return 0;
}
