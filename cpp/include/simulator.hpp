#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace npuforge {
using U64 = std::uint64_t;
struct Hardware {
  U64 sram_bytes = 65536;
  U64 macs_per_cycle = 64;
  U64 elements_per_cycle = 16;
  U64 dma_bytes_per_cycle = 32;
  U64 dma_setup_cycles = 8;
  U64 compute_setup_cycles = 4;
  U64 pim_macs_per_cycle = 64;
  U64 pim_bytes_per_cycle = 128;
  double clock_hz = 1e9;
};
struct Instruction {
  int id = 0;
  std::string op;
  std::vector<int> deps;
  int buffer = -1;
  U64 sram_bytes = 0;
  U64 working_set_bytes = 0;
  U64 bytes = 0;
  U64 macs = 0;
  U64 elements = 0;
  U64 local_read_bytes = 0;
};
struct Event {
  int id;
  std::string op;
  U64 start;
  U64 end;
};
struct Result {
  U64 cycles = 0;
  U64 dram_read_bytes = 0;
  U64 dram_write_bytes = 0;
  U64 offchip_read_bytes = 0;
  U64 pim_local_read_bytes = 0;
  U64 peak_sram_bytes = 0;
  U64 compute_busy = 0;
  U64 dma_busy = 0;
  U64 pim_busy = 0;
  U64 dram_busy = 0;
  double latency_ns = 0;
  std::string bottleneck;
  std::vector<Event> trace;
};
U64 duration(const Instruction &instruction, const Hardware &hardware);
Result simulate(const std::vector<Instruction> &instructions, const Hardware &hardware);
} // namespace npuforge
