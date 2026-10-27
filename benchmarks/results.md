| Case | Cycles | Latency (ns) | Off-chip bytes | PIM-local bytes | Peak SRAM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| matmul_conventional | 560 | 560 | 8704 | 0 | 3072 |
| matmul_pim | 456 | 456 | 5632 | 3072 | 2048 |
| matmul_tile8 | 1152 | 1152 | 13824 | 0 | 768 |
| mlp_unoptimized | 1169 | 1169 | 19984 | 0 | 4608 |
| mlp_optimized | 945 | 945 | 13840 | 0 | 4096 |
| mlp_pim | 798 | 798 | 9616 | 4224 | 3072 |
