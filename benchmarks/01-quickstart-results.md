# 01 — Quickstart Results

Settings: `n_threads=12`, `n_ctx=2048`, `n_batch=512`, `n_gpu_layers=99`.

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b-instruct-q4_k_m.gguf | 475 | 76 / 920 | 23.3 / 26.2 | 1542 / 2571 / 2794 | 42.9 |
| qwen2.5-0.5b-instruct-q2_k.gguf | 343 | 133 / 1733 | 22.7 / 22.9 | 1566 / 3156 / 4004 | 44.0 |

## Observations

- TTFT is the prefill cost. With short prompts this is small; with long prompts it dominates.
- TPOT is per-token decode latency. The decode rate is `1000 / TPOT_p50`.
- The bigger quantization (Q4_K_M) is usually only ~30–60% slower than Q2_K but produces noticeably better text. Q2_K is for *truly* tight RAM.
- `n_threads = physical_cores` is usually best on CPU. Hyperthreading (`logical_cores`) often hurts because the work is bandwidth-bound.

(Edit this file with your own observations before submitting.)
