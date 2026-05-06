# Bonus — Thread-count sweep

Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf`  ·  backend: Vulkan  ·  ngl: 24

| threads | tg128 (t/s) |
|--:|--:|
| 1 | 27.6 |
| 2 | 28.5 |
| 4 | 25.6 |
| 6 | 29.4 |
| 8 | 27.0 |
| 10 | 26.8 |
| 12 | 30.3 |

## Observation

Thread count has minimal impact on this Vulkan/iGPU setup — ranging from 25.6 to 30.3 t/s. The variance is high (±13–14 t/s) indicating the GPU is the bottleneck, not CPU thread count. On CPU-only builds, thread count typically shows a clear peak at physical core count. On Vulkan with GPU offload, the GPU compute dominates and CPU threads matter less.

Takeaway: For Vulkan/GPU inference, don't overthink thread count — 6–12 threads all work similarly.
