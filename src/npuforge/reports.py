"""HTML reports and SVG graph/timeline plots."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from .ir import Graph
from .pipeline import Run

COLORS = {
    "DMA_LOAD": "#238da4",
    "DMA_STORE": "#238da4",
    "MATMUL": "#7756cb",
    "FUSED_MATMUL_ADD_RELU": "#7756cb",
    "ADD": "#7756cb",
    "RELU": "#7756cb",
    "PIM_MATMUL": "#c47a28",
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def graph_svg(graph: Graph) -> str:
    ordered = graph.ordered()
    height = max(180, 90 + 100 * len(ordered))
    positions = {o.output: (300, 80 + index * 100) for index, o in enumerate(ordered)}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 {height}" '
        'role="img" aria-label="Optimized compiler graph">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
        'refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#a5afc2"/>'
        "</marker></defs>",
        "<style>text{font-family:system-ui,sans-serif}</style>",
    ]
    for op in ordered:
        x, y = positions[op.output]
        sources = ", ".join(t for t in op.inputs if t not in positions)
        for tid in op.inputs:
            if tid in positions:
                _, py = positions[tid]
                parts.append(
                    f'<path d="M440 {py + 26} C550 {py + 26} 550 {y - 26} '
                    f'440 {y - 26}" fill="none" stroke="#a5afc2" '
                    'stroke-width="1.5" marker-end="url(#arrow)"/>'
                )
        fill = "#f1ecff" if op.op.startswith("Fused") else "#eff6f8"
        parts.extend(
            [
                f"<g><title>{esc(op.id)} {esc(json.dumps(op.attrs, sort_keys=True))}</title>",
                f'<rect x="{x - 140}" y="{y - 32}" width="280" height="64" rx="12" '
                f'fill="{fill}" stroke="#d1d9e6"/>',
                f'<text x="300" y="{y - 7}" text-anchor="middle" font-size="14" '
                f'font-weight="650" fill="#23314b">{esc(op.op)}</text>',
                f'<text x="300" y="{y + 16}" text-anchor="middle" font-size="12" '
                f'fill="#56647b">{esc(op.output)} · {esc(graph.tensors[op.output].shape)}</text>',
                f'<text x="145" y="{y}" text-anchor="end" font-size="10" '
                f'fill="#56647b">{esc(sources)}</text>',
                "</g>",
            ]
        )
    if not ordered:
        parts.append('<text x="30" y="80">Output aliases a graph input or initializer.</text>')
    parts.append("</svg>")
    return "".join(parts)


def timeline_svg(run: Run) -> str:
    metrics = run.metrics
    cycles = max(1, metrics["cycles"])
    trace = [e for e in metrics["trace"] if e["end"] > e["start"]]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 230" '
        'role="img" aria-label="Instruction resource timeline">',
        "<style>text{font-family:system-ui,sans-serif}</style>",
    ]
    for index, name in enumerate(["DMA", "Compute", "PIM"]):
        y = 40 + 48 * index
        parts.extend(
            [
                f'<text x="14" y="{y + 19}" font-size="13" fill="#56647b">{name}</text>',
                f'<rect x="105" y="{y}" width="865" height="30" rx="5" fill="#f0f3f8"/>',
            ]
        )
    for event in trace[:1000]:
        lane = 0 if event["op"].startswith("DMA") else (2 if event["op"] == "PIM_MATMUL" else 1)
        x = 105 + 865 * event["start"] / cycles
        width = max(0.2, 865 * (event["end"] - event["start"]) / cycles)
        parts.append(
            f'<rect x="{x:.3f}" y="{40 + 48 * lane}" width="{width:.3f}" '
            f'height="30" rx="2" fill="{COLORS[event["op"]]}"><title>'
            f"#{event['id']} {event['op']}: {event['start']}–{event['end']} cycles"
            "</title></rect>"
        )
    for tick in range(5):
        x = 105 + 865 * tick / 4
        parts.append(
            f'<text x="{x}" y="200" text-anchor="middle" font-size="12" '
            f'fill="#56647b">{cycles * tick / 4:,.0f}</text>'
        )
    note = "Cycles · first 1,000 active instructions shown" if len(trace) > 1000 else "Cycles"
    parts.append(
        f'<text x="540" y="225" text-anchor="middle" font-size="12" '
        f'fill="#56647b">{note}</text></svg>'
    )
    return "".join(parts)


def comparison_rows(runs: list[tuple[str, Run]]) -> str:
    rows = []
    for label, run in runs:
        m = run.metrics
        rows.append(
            f"<tr><td>{esc(label)}</td><td>{m['cycles']:,}</td>"
            f"<td>{m['latency_ns']:,.1f}</td><td>{m['offchip_bytes']:,}</td>"
            f"<td>{m['pim_local_read_bytes']:,}</td><td>{m['peak_sram_bytes']:,}</td>"
            f"<td>{m['compute_utilization']:.1%}</td>"
            f"<td>{m['dma_utilization']:.1%}</td><td>{m['pim_utilization']:.1%}</td></tr>"
        )
    return "".join(rows)


def render_report(runs: list[tuple[str, Run]], path: str | Path) -> Path:
    if not runs:
        raise ValueError("report requires at least one run")
    sections = []
    for label, run in runs:
        m = run.metrics
        changes = run.changes
        config = "".join(
            f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"
            for k, v in asdict(run.program.hardware).items()
        )
        timing = "".join(
            f"<tr><td>{esc(node)}</td><td>{t['start']:,}</td>"
            f"<td>{t['end']:,}</td><td>{t['busy_cycles']:,}</td></tr>"
            for node, t in m["per_operation"].items()
        )
        inputs = ", ".join(f"{t}: {run.graph.tensors[t].shape}" for t in run.graph.inputs)
        sections.append(f"""
<section><div class="section-label">COMPILATION / {esc(label)}</div>
<h2>{esc(run.model)} <span class="pill">{esc(run.program.hardware.architecture)}</span></h2>
<p class="muted">FP32 · Input {esc(inputs)} · {len(run.program.instructions):,} instructions ·
{len(run.graph.outputs)} output(s)</p>
<div class="cards">
<div><span>Simulated cycles</span><strong>{m["cycles"]:,}</strong></div>
<div><span>Estimated latency</span><strong>{m["latency_ns"]:,.0f}<small> ns</small></strong></div>
<div><span>Off-chip movement</span><strong>{m["offchip_bytes"]:,}<small> B</small></strong></div>
<div><span>Peak SRAM</span><strong>{m["peak_sram_bytes"]:,}<small> B</small></strong></div>
</div>
<div class="notice">Verified against ONNX Runtime · rtol {run.verification["rtol"]} ·
atol {run.verification["atol"]} · largest absolute error
{max(run.verification["max_absolute_error"].values()):.3g}</div>
<div class="two-col"><div><h3>Compiler graph</h3>
<p>{changes["original_operations"]} → {changes["final_operations"]} operations ·
{changes["dead_operations_removed"]} dead operations removed ·
{changes["fusion_patterns"]} fusion pattern(s)</p>{graph_svg(run.graph)}</div>
<div><h3>Modeled bottleneck</h3><p class="lead">{esc(m["bottleneck"])}</p>
<p>The busiest modeled resource determines this label. It is a pressure indicator,
not a causal speedup prediction. Dependencies and the compiler's single tile workspace
also limit overlap.</p>
<table><tr><td>Compute utilization</td><td>{m["compute_utilization"]:.1%}</td></tr>
<tr><td>DMA utilization</td><td>{m["dma_utilization"]:.1%}</td></tr>
<tr><td>PIM utilization</td><td>{m["pim_utilization"]:.1%}</td></tr>
<tr><td>DRAM port utilization</td><td>{m["dram_utilization"]:.1%}</td></tr>
<tr><td>Total DRAM reads</td><td>{m["dram_read_bytes"]:,} B</td></tr>
<tr><td>DRAM writes</td><td>{m["dram_write_bytes"]:,} B</td></tr>
<tr><td>Memory-local weight reads</td><td>{m["pim_local_read_bytes"]:,} B</td></tr></table>
<details><summary>Hardware configuration</summary><table>{config}</table></details></div></div>
<h3>Instruction timeline</h3>{timeline_svg(run)}
<details><summary>Per-operation timing (cycles)</summary><table>
<thead><tr><th>Operation</th><th>Start</th><th>End</th><th>Sum of busy durations</th></tr></thead>
<tbody>{timing}</tbody></table><p>Overlapping durations may sum above elapsed time.
Total latency is the native scheduler's makespan.</p></details></section>""")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NPUForge · Compiler &amp; architecture report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf1f7;color:#23314b;
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:auto;padding:48px 30px}}header{{padding:12px 0 32px}}
section{{background:white;border:1px solid #dce3ee;border-radius:18px;padding:30px;
margin:0 0 24px;box-shadow:0 4px 16px #23314b04;overflow:hidden}}
.section-label{{font-size:11px;letter-spacing:1.5px;font-weight:700;color:#6f7b91}}
.pill{{display:inline-block;font-size:12px;background:#eee8fa;color:#6850ac;
border-radius:20px;padding:4px 12px;vertical-align:middle;letter-spacing:0}}
.brand{{font-size:13px;letter-spacing:3px;color:#6850ac;font-weight:750}}
h1{{font-size:46px;line-height:1.15;letter-spacing:-1.8px;margin:14px 0}}h2{{font-size:26px;
letter-spacing:-.6px;margin:8px 0}}h3{{font-size:17px;margin:20px 0 8px}}
p{{margin:8px 0 16px}}.subtitle,.muted{{color:#62718a}}.subtitle{{font-size:18px;max-width:760px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}
.cards>div{{background:#f5f7fb;border:1px solid #e7ebf3;border-radius:12px;padding:18px}}
.cards span{{display:block;font-size:12px;color:#62718a}}strong{{display:block;font-size:28px;
font-weight:650;letter-spacing:-.6px}}small{{font-size:14px;font-weight:400;color:#62718a}}
.notice{{background:#eaf6f1;color:#21674f;padding:12px 16px;border-radius:8px;font-size:13px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:40px;margin-top:16px}}
svg{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}th,td{{padding:10px 12px;
text-align:right;border-bottom:1px solid #e8edf4;font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}}th{{color:#62718a;font-weight:600}}
.scroll{{overflow-x:auto}}.lead{{font-size:23px;font-weight:600;color:#6850ac}}
summary{{cursor:pointer;color:#6850ac;font-size:13px}}details{{margin:16px 0}}
footer{{font-size:12px;color:#62718a;padding:0 12px}}code{{font-size:12px}}
@media(max-width:800px){{main{{padding:24px 12px}}section{{padding:20px}}h1{{font-size:32px}}
.cards{{grid-template-columns:1fr 1fr}}.two-col{{grid-template-columns:1fr;gap:0}}}}
</style></head><body><main><header><div class="brand">NPUFORGE / ENGINEERING REPORT</div>
<h1>Compilation and simulation results.</h1>
<p class="subtitle">Timing, memory traffic and resource utilization from
the generated instruction streams.</p>
<span class="pill">FP32 · deterministic · abstract hardware model</span></header>
<section><div class="section-label">EXPERIMENT SUMMARY</div>
<h2>Architecture &amp; compiler comparisons</h2>
<p>Identical deterministic inputs and weights. Latency is simulated accelerator time,
not measured host runtime. All configurations and regressions are included.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Cycles</th><th>Latency (ns)</th>
<th>Off-chip (B)</th><th>PIM-local (B)</th><th>Peak SRAM (B)</th><th>Compute</th>
<th>DMA</th><th>PIM</th></tr></thead><tbody>{comparison_rows(runs)}</tbody></table></div>
<h3>What differs in PIM mode?</h3><p>Weights remain in a hypothetical memory-local compute
array. Activation tiles enter SRAM and output tiles leave through DMA. Each PIM matrix
instruction pays for both MAC work and local weight reads, and occupies the same DRAM port
as DMA. Bias epilogues also cost cycles. Conventional mode explicitly transfers both matrix
operands to SRAM. No cache reuse or persistent weight placement is assumed.</p>
<p>Off-chip bytes count DMA reads + writes; total DRAM reads additionally include PIM-local
weight reads. PIM performance depends on the stated rates and may be slower.</p></section>
{"".join(sections)}
<footer>Abstract hardware estimates. Timing formulas and data placement are documented in
docs/architecture.md. Full instruction and trace JSON are saved beside this report.
</footer></main></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
    return path
