#include "simulator.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace npuforge;

PYBIND11_MODULE(_native, m) {
  m.doc() = "Deterministic C++17 abstract NPU resource scheduler";
  m.def("simulate", [](const py::list &program, const py::dict &config) {
    Hardware h;
#define FIELD(name) h.name = config[#name].cast<decltype(h.name)>()
    FIELD(sram_bytes);
    FIELD(macs_per_cycle);
    FIELD(elements_per_cycle);
    FIELD(dma_bytes_per_cycle);
    FIELD(dma_setup_cycles);
    FIELD(compute_setup_cycles);
    FIELD(pim_macs_per_cycle);
    FIELD(pim_bytes_per_cycle);
    FIELD(clock_hz);
#undef FIELD
    std::vector<Instruction> instructions;
    for (const auto &item : program) {
      const auto d = py::cast<py::dict>(item);
      Instruction i;
      i.id = d["id"].cast<int>();
      i.op = d["op"].cast<std::string>();
      i.deps = d["deps"].cast<std::vector<int>>();
#define OPTIONAL(name)                                                                             \
  if (d.contains(#name))                                                                           \
  i.name = d[#name].cast<decltype(i.name)>()
      OPTIONAL(buffer);
      OPTIONAL(sram_bytes);
      OPTIONAL(working_set_bytes);
      OPTIONAL(bytes);
      OPTIONAL(macs);
      OPTIONAL(elements);
      OPTIONAL(local_read_bytes);
#undef OPTIONAL
      instructions.push_back(std::move(i));
    }
    Result r;
    {
      py::gil_scoped_release release;
      r = simulate(instructions, h);
    }
    py::dict out;
#define OUTPUT(name) out[#name] = r.name
    OUTPUT(cycles);
    OUTPUT(latency_ns);
    OUTPUT(dram_read_bytes);
    OUTPUT(dram_write_bytes);
    OUTPUT(offchip_read_bytes);
    OUTPUT(pim_local_read_bytes);
    OUTPUT(peak_sram_bytes);
    OUTPUT(compute_busy);
    OUTPUT(dma_busy);
    OUTPUT(pim_busy);
    OUTPUT(dram_busy);
    OUTPUT(bottleneck);
#undef OUTPUT
    const double denom = r.cycles ? static_cast<double>(r.cycles) : 1.0;
    out["compute_utilization"] = r.compute_busy / denom;
    out["dma_utilization"] = r.dma_busy / denom;
    out["pim_utilization"] = r.pim_busy / denom;
    out["dram_utilization"] = r.dram_busy / denom;
    out["offchip_bytes"] = r.offchip_read_bytes + r.dram_write_bytes;
    py::list trace;
    for (const auto &e : r.trace) {
      py::dict d;
      d["id"] = e.id;
      d["op"] = e.op;
      d["start"] = e.start;
      d["end"] = e.end;
      trace.append(d);
    }
    out["trace"] = trace;
    return out;
  });
}
