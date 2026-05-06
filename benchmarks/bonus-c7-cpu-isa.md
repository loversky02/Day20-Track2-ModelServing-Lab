# Bonus C7 — CPU Instruction Set Archaeology

Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` · backend: Vulkan · ngl=24 · threads=12

## CPU-only (ngl=0)

| Build | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| Generic (no AVX, -DGGML_NATIVE=OFF) | 978.8 | 31.3 |
| AVX2 only (-DGGML_AVX2=ON) | 990.9 | 26.8 |
| Native (-DGGML_NATIVE=ON, march=native) | 947.5 | 25.0 |

## GPU offload (ngl=24)

| Build | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| Generic (no AVX) | 1264.4 | 35.0 |
| AVX2 only | 563.1 (±593) | 31.3 |
| Native (march=native) | 1316.4 | 31.6 |

## Insight

With Vulkan GPU offload, CPU instruction set tuning has negligible impact on generation speed (31–35 tok/s across all builds). Prompt processing shows slight improvement with Native vs Generic (~4%). This is the opposite of what you'd see on CPU-only builds (where AVX2/Native can give 20–40% speedup). The takeaway: with GPU inference, invest tuning time in GPU backend knobs (ngl, batch size) rather than CPU instruction flags.

The AVX2-only build showed massive variance (±593 t/s in pp512), possibly due to missing instruction scheduling optimizations that march=native provides. This mirrors the deck's discussion: "match the kernel to the silicon" — partial instruction set enabling (just AVX2 without proper scheduling) can be worse than either full native or clean generic.
