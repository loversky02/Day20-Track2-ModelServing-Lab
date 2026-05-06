# Bonus C1 — Speculative Decoding

Model: Qwen2.5-1.5B-Instruct Q4_K_M (target) + Qwen2.5-0.5B-Instruct Q4_K_M (draft)
Backend: Vulkan, ngl=24 (target), ngld=0 (draft on CPU), threads=12, draft-threads=4

| Config | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| Target only (1.5B, ngl=24) | 127.4 | 9.8 |
| Target(1.5B,ngl=24) + Draft(0.5B,CPU) | 77.3 | 7.4 |

## Speedup

Speculative decoding is **0.76x** (i.e., 24% slower than target-only), and prompt processing drops 39% (127→77 t/s).

## Why it fails on this hardware

Speculative decoding normally helps when: (a) the draft model is _much_ faster than the target, and (b) draft acceptance rate is high (same tokenizer family). Both conditions fail here on Intel Iris Xe iGPU with unified memory:

1. **Memory bandwidth contention** — Unified memory means GPU and CPU share the same DRAM bus. Running target (GPU) and draft (CPU) simultaneously saturates memory bandwidth, hurting both models.
2. **Draft not fast enough** — The 0.5B draft on CPU (4 threads) isn't much faster than 1.5B target on Vulkan GPU. The speculation overhead (running draft forward passes, then verifying with target) is larger than the savings from accepted draft tokens.
3. **Low acceptance rate** — Qwen2.5-0.5B and Qwen2.5-1.5B are different model sizes with diverging token distributions. Even with `--spec-draft-n-max 4`, accepted tokens per draft cycle is likely 1–2, erasing the theoretical speedup.
4. **Prompt processing overhead** — The draft model must also process the prompt (77.3 vs 127.4 t/s), adding 39% overhead before generation begins.

## Conclusion

On integrated GPU (unified memory), speculative decoding is counterproductive. The compute saved by drafting tokens is less than the memory bandwidth cost of running two models. This technique is only worthwhile with:
- Dedicated GPU VRAM (target on GPU, draft on GPU with spare VRAM)
- Very fast draft model (e.g., 0.1B or n-gram predictor, not another NN)
- High-spec CPU with AVX2 that can run draft inference without touching GPU memory bus
