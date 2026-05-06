# Bonus — GPU-offload sweep

Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`  ·  threads: `12`  ·  backend: Vulkan (Intel Iris Xe)

| -ngl | pp512 (t/s) | tg128 (t/s) |
|--:|--:|--:|
| 0 | 1012.0 | 35.7 |
| 8 | 1049.3 | 24.1 |
| 16 | 1140.6 | 30.7 |
| 24 | 1332.2 | 31.2 |
| 32 | 1140.6 | 38.5 |
| 99 | 403.8 | 33.6 |

## Observation

Optimal prompt processing at `-ngl 24` (1332 t/s, +32% vs CPU-only). Full GPU offload (`-ngl 99`) is actually **worse** than CPU-only for prompt processing (404 vs 1012 t/s) — this is because Intel Iris Xe uses unified memory (no dedicated VRAM). Full offload saturates the memory bus competing with CPU.

Best generation speed at `-ngl 32` (38.5 t/s), but the improvement over CPU-only (35.7 t/s) is marginal (+8%). Generation is memory-bandwidth-bound, not compute-bound — GPU shader cores can't help much.

Takeaway: On integrated GPUs with unified memory, partial offload (`-ngl 24`) is the sweet spot.
