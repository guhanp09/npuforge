#include "simulator.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace npuforge;
namespace {
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void rejects(const std::function<void()> &fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "invalid workload was accepted");
}
Instruction inst(int id, std::string op, std::vector<int> deps = {}) {
  Instruction i;
  i.id = id;
  i.op = std::move(op);
  i.deps = std::move(deps);
  return i;
}
} // namespace
int main() {
  try {
    Hardware h;
    h.macs_per_cycle = 4;
    h.compute_setup_cycles = 2;
    h.dma_bytes_per_cycle = 8;
    h.dma_setup_cycles = 3;
    auto c = inst(0, "MATMUL");
    c.macs = 17; // 2 + ceil(17/4) = 7
    check(simulate({c}, h).cycles == 7, "single compute");
    auto d = inst(0, "DMA_LOAD");
    d.bytes = 17; // 3 + ceil(17/8) = 6
    check(simulate({d}, h).cycles == 6, "single DMA");
    c.id = 1;
    c.deps = {0};
    check(simulate({d, c}, h).cycles == 13, "dependency serialization");
    c.deps.clear();
    auto parallel = simulate({d, c}, h);
    check(parallel.cycles == 7, "independent resource overlap");
    auto c2 = c;
    c.id = 0;
    c2.id = 1;
    check(simulate({c, c2}, h).cycles == 14, "compute contention");
    auto a = inst(0, "ALLOC");
    a.buffer = 7;
    a.sram_bytes = 64;
    d.id = 1;
    d.deps = {0};
    d.buffer = 7;
    auto f = inst(2, "FREE", {1});
    f.buffer = 7;
    check(simulate({a, d, f}, h).peak_sram_bytes == 64, "SRAM lifetime");
    h.sram_bytes = 32;
    rejects([&] { simulate({a, d, f}, h); });
    h.sram_bytes = 64;
    auto a2 = a;
    a2.id = 1;
    a2.buffer = 8;
    auto f1 = inst(2, "FREE", {1});
    f1.buffer = 7;
    auto f2 = inst(3, "FREE", {2});
    f2.buffer = 8;
    rejects([&] { simulate({a, a2, f1, f2}, h); }); // Allocation deadlock
    c.id = 0;
    c.deps = {1};
    c2.deps = {0};
    rejects([&] { simulate({c, c2}, h); });
    c.deps = {9};
    rejects([&] { simulate({c}, h); });
    d.id = 0;
    d.deps.clear();
    d.buffer = -1;
    for (int j = 0; j < 20; ++j)
      check(simulate({d}, h).cycles == 6, "determinism");
    auto p = inst(1, "PIM_MATMUL");
    p.macs = 64;
    p.local_read_bytes = 128;
    // Default PIM: 2 setup + 1 MAC cycle + 1 local-read cycle = 4;
    // DMA and PIM share DRAM, hence 6 + 4 = 10.
    check(simulate({d, p}, h).cycles == 10, "PIM/DRAM contention");
    auto bad = inst(0, "UNKNOWN");
    rejects([&] { simulate({bad}, h); });
    h.clock_hz = 0;
    rejects([&] { simulate({}, h); });
    std::cout << "All hand-calculated simulator invariants passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
