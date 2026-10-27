#include "simulator.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace npuforge {
namespace {
U64 add(U64 a, U64 b) {
  if (b > std::numeric_limits<U64>::max() - a)
    throw std::invalid_argument("cycle or byte count overflow");
  return a + b;
}
U64 ceildiv(U64 a, U64 b) { return a / b + (a % b != 0); }
// A DMA and a PIM instruction both occupy the single DRAM port.
int resources(const std::string &op) {
  if (op == "DMA_LOAD" || op == "DMA_STORE")
    return 2 | 8;
  if (op == "PIM_MATMUL")
    return 4 | 8;
  if (op == "MATMUL" || op == "ADD" || op == "RELU" || op == "FUSED_MATMUL_ADD_RELU")
    return 1;
  if (op == "ALLOC" || op == "FREE")
    return 0;
  throw std::invalid_argument("unknown instruction: " + op);
}
void validate_hardware(const Hardware &h) {
  if (!h.sram_bytes || !h.macs_per_cycle || !h.elements_per_cycle || !h.dma_bytes_per_cycle ||
      !h.pim_macs_per_cycle || !h.pim_bytes_per_cycle || !std::isfinite(h.clock_hz) ||
      h.clock_hz <= 0)
    throw std::invalid_argument("hardware capacities, rates and clock must be positive");
}
} // namespace

U64 duration(const Instruction &i, const Hardware &h) {
  validate_hardware(h);
  resources(i.op);
  if (i.op == "ALLOC" || i.op == "FREE")
    return 0;
  if (i.op == "DMA_LOAD" || i.op == "DMA_STORE")
    return std::max<U64>(1, add(h.dma_setup_cycles, ceildiv(i.bytes, h.dma_bytes_per_cycle)));
  U64 work = 0;
  if (i.op == "MATMUL" || i.op == "FUSED_MATMUL_ADD_RELU")
    work = ceildiv(i.macs, h.macs_per_cycle);
  else if (i.op == "PIM_MATMUL")
    work = add(ceildiv(i.macs, h.pim_macs_per_cycle),
               ceildiv(i.local_read_bytes, h.pim_bytes_per_cycle));
  // Fused epilogues execute on the same engine; their scalar work is not free.
  work = add(work, ceildiv(i.elements, h.elements_per_cycle));
  return std::max<U64>(1, add(h.compute_setup_cycles, work));
}

Result simulate(const std::vector<Instruction> &instructions, const Hardware &h) {
  validate_hardware(h);
  const auto n = instructions.size();
  std::vector<std::vector<int>> children(n);
  std::vector<int> remaining(n);
  std::vector<int> masks(n);
  std::vector<U64> durations(n);
  // Validate the complete DAG before changing simulation state.
  for (std::size_t idx = 0; idx < n; ++idx) {
    const auto &i = instructions[idx];
    if (i.id != static_cast<int>(idx))
      throw std::invalid_argument("instruction IDs must be dense and in order");
    masks[idx] = resources(i.op);
    durations[idx] = duration(i, h);
    if ((i.op == "ALLOC" || i.op == "FREE") && i.buffer < 0)
      throw std::invalid_argument("allocation instructions require a buffer ID");
    if (i.op == "ALLOC" && (!i.sram_bytes || i.sram_bytes > h.sram_bytes))
      throw std::invalid_argument("allocation exceeds SRAM capacity or is empty");
    std::unordered_set<int> seen;
    for (int dep : i.deps) {
      if (dep < 0 || dep >= static_cast<int>(n) || dep == i.id || !seen.insert(dep).second)
        throw std::invalid_argument("invalid or duplicate dependency");
      children[dep].push_back(i.id);
      ++remaining[idx];
    }
  }
  auto counts = remaining;
  std::queue<int> roots;
  for (std::size_t j = 0; j < n; ++j)
    if (!counts[j])
      roots.push(static_cast<int>(j));
  std::size_t visited = 0;
  while (!roots.empty()) {
    int id = roots.front();
    roots.pop();
    ++visited;
    for (int child : children[id])
      if (--counts[child] == 0)
        roots.push(child);
  }
  if (visited != n)
    throw std::invalid_argument("cyclic instruction dependencies");

  struct Running {
    int id;
    U64 end;
  };
  std::vector<Running> running;
  std::vector<bool> started(n, false);
  std::map<int, U64> buffers;
  Result result;
  U64 now = 0, allocated = 0;
  int busy = 0;
  std::size_t completed = 0;
  auto finish = [&](int id) {
    ++completed;
    for (int child : children[id])
      --remaining[child];
  };
  while (completed < n) {
    for (auto it = running.begin(); it != running.end();) {
      if (it->end == now) {
        busy &= ~masks[it->id];
        finish(it->id);
        it = running.erase(it);
      } else {
        ++it;
      }
    }
    // Stable ID priority, repeating for zero-cycle ALLOC/FREE dependencies.
    bool progress;
    do {
      progress = false;
      for (std::size_t idx = 0; idx < n; ++idx) {
        if (started[idx] || remaining[idx] || (busy & masks[idx]))
          continue;
        const auto &i = instructions[idx];
        if (i.op == "ALLOC") {
          if (buffers.count(i.buffer))
            throw std::invalid_argument("buffer allocated twice");
          if (i.sram_bytes > h.sram_bytes - allocated)
            continue; // Wait for an eligible FREE or completion.
          buffers[i.buffer] = i.sram_bytes;
          allocated += i.sram_bytes;
          result.peak_sram_bytes = std::max(result.peak_sram_bytes, allocated);
        } else if (i.op == "FREE") {
          auto it = buffers.find(i.buffer);
          if (it == buffers.end())
            throw std::invalid_argument("free of unallocated buffer");
          for (const auto &r : running)
            if (instructions[r.id].buffer == i.buffer)
              throw std::invalid_argument("free while buffer is in use");
          allocated -= it->second;
          buffers.erase(it);
        } else if (i.buffer >= 0 && !buffers.count(i.buffer)) {
          throw std::invalid_argument("instruction uses unallocated buffer");
        }
        if (masks[idx] && i.buffer >= 0 &&
            (i.working_set_bytes > buffers.at(i.buffer) ||
             ((i.op == "DMA_LOAD" || i.op == "DMA_STORE") && i.bytes > buffers.at(i.buffer))))
          throw std::invalid_argument("instruction working set exceeds SRAM allocation");
        if (i.working_set_bytes && i.buffer < 0)
          throw std::invalid_argument("working set requires an SRAM allocation");
        started[idx] = true;
        progress = true;
        const U64 end = add(now, durations[idx]);
        result.trace.push_back({i.id, i.op, now, end});
        if (!durations[idx]) {
          finish(i.id);
        } else {
          running.push_back({i.id, end});
          busy |= masks[idx];
          if (masks[idx] & 1)
            result.compute_busy = add(result.compute_busy, durations[idx]);
          if (masks[idx] & 2)
            result.dma_busy = add(result.dma_busy, durations[idx]);
          if (masks[idx] & 4)
            result.pim_busy = add(result.pim_busy, durations[idx]);
          if (masks[idx] & 8)
            result.dram_busy = add(result.dram_busy, durations[idx]);
          if (i.op == "DMA_LOAD") {
            result.offchip_read_bytes = add(result.offchip_read_bytes, i.bytes);
            result.dram_read_bytes = add(result.dram_read_bytes, i.bytes);
          } else if (i.op == "DMA_STORE") {
            result.dram_write_bytes = add(result.dram_write_bytes, i.bytes);
          } else if (i.op == "PIM_MATMUL") {
            result.pim_local_read_bytes = add(result.pim_local_read_bytes, i.local_read_bytes);
            result.dram_read_bytes = add(result.dram_read_bytes, i.local_read_bytes);
          }
        }
      }
    } while (progress);
    if (completed == n)
      break;
    if (running.empty())
      throw std::invalid_argument("SRAM allocation deadlock: add release dependencies or capacity");
    now = std::min_element(running.begin(), running.end(), [](const auto &a, const auto &b) {
            return a.end < b.end;
          })->end;
  }
  if (!buffers.empty())
    throw std::invalid_argument("unreleased SRAM allocations at end of workload");
  result.cycles = now;
  result.latency_ns = static_cast<double>(now) * 1e9 / h.clock_hz;
  result.bottleneck = "no work";
  if (now) {
    auto compute = add(result.compute_busy, result.pim_busy);
    result.bottleneck = result.dram_busy > compute ? "memory port" : "compute engines";
    if (result.dram_busy == compute)
      result.bottleneck = "balanced compute / memory port";
  }
  return result;
}
} // namespace npuforge
