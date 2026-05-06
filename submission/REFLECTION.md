# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** Tran Dinh Minh Vuong
**MSSV:** 2A202600495
**Cohort:** A20-K1
**Ngày submit:** 2026-05-06

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

> Paste output của `python 00-setup/detect-hardware.py` vào đây, hoặc điền thủ công:

- **OS:** Ubuntu 7.0.0-15-generic (x86_64)
- **CPU:** 12th Gen Intel(R) Core(TM) i7-1255U
- **Cores:** 12 physical / 12 logical
- **CPU extensions:** AVX2
- **RAM:** 37.9 GB
- **Accelerator:** Intel Iris Xe Graphics (ADL GT2) — Vulkan
- **llama.cpp backend đã chọn:** Vulkan (via Mesa driver)
- **Recommended model tier:** Qwen2.5-0.5B-Instruct (thực tế dùng; recommended 7B không tải được do mirror HF lỗi)

**Setup story** (≤ 80 chữ): những gì cần thay đổi để lab chạy được trên máy bạn (vd: dùng WSL2, install CUDA Toolkit, fall back sang Vulkan vì ROCm phiên bản kén, tắt antivirus để pip install nhanh hơn, v.v.):

Cần cài `vulkan-tools libvulkan-dev glslc spirv-headers cmake` để build llama.cpp từ source với Vulkan. detect-hardware.py không nhận diện được Vulkan dù `vulkaninfo` có sẵn — phải sửa hardware.json thủ công. PyPI chậm từ VN — chuyển sang Alibaba mirror (`mirrors.aliyun.com/pypi/simple/`) mới tải được dependencies. Thiếu `python3.14-venv`, `uvicorn`, `starlette-context`, và `llama-cpp-python[server]` — phải cài bổ sung từng gói.

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

> Paste bảng từ `benchmarks/01-quickstart-results.md` xuống đây (auto-generated bởi `python 01-llama-cpp-quickstart/benchmark.py`).

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b-instruct-q4_k_m.gguf | 475 | 76 / 920 | 23.3 / 26.2 | 1542 / 2571 / 2794 | 42.9 |
| qwen2.5-0.5b-instruct-q2_k.gguf | 343 | 133 / 1733 | 22.7 / 22.9 | 1566 / 3156 / 4004 | 44.0 |

**Một quan sát** (≤ 50 chữ): Q4_K_M vs Q2_K trên máy bạn — số liệu nói gì? Quality đáng đánh đổi không?

Q4_K_M có TTFT P50 nhanh hơn (76ms vs 133ms) và P95 thấp hơn nhiều (920ms vs 1733ms), nhưng decode rate chậm hơn chút (42.9 vs 44.0 tok/s). Với 37.9GB RAM, không cần tiết kiệm 73MB — Q4_K_M là lựa chọn rõ ràng cho chất lượng tốt hơn.

---

## 3. Track 02 — llama-server load test

> Chạy 2 lần locust ở concurrency 10 và 50, paste tóm tắt bên dưới.

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.42 | 20000 | 25000 | 26000 | 0 |
| 50 | 0.39 | 16000 | 49000 | 50000 | 0 |

**KV-cache observation** (từ C++ `llama-server --metrics`): Sau 6 requests concurrent trên Qwen2.5-0.5B với `-c 2048`:

| Metric | Idle | Under load (5 concurrent) |
|---|---|---|
| `n_busy_slots_per_decode` | 1.0 | 2.48 |
| `requests_deferred` | 0 | 0 |
| `prompt_tokens_seconds` | 204 t/s | 90 t/s |
| `predicted_tokens_seconds` | 43 t/s | 22 t/s |
| `n_tokens_max` | 71 | 98 |

`requests_deferred = 0` ngay cả dưới 5 concurrent requests — KV cache (`-c 2048`) đủ lớn cho workload này. `n_busy_slots_per_decode` tăng từ 1 lên 2.48 cho thấy server đang xử lý song song 2–3 slots, xác nhận continuous batching hoạt động. Throughput giảm khi nhiều users chia sẻ GPU bandwidth.

Server xử lý 22 requests (10 users) và 23 requests (50 users) trong 1 phút. 0 failures. Bottleneck chính là small model (0.5B) + integrated GPU — requests queue lên dẫn đến P95 cao (25–49s).

**Prometheus /metrics (optional extra):** Chạy llama-server với `--metrics`, scrape `/metrics` endpoint trong 3 phase (idle → 4 concurrent requests → cool down):

| Metric | Idle | Under Load | Δ |
|---|---|---|---|
| `prompt_tokens_seconds` | 155 t/s | 86 t/s | -69 |
| `predicted_tokens_seconds` | 25.5 t/s | 29.3 t/s | +3.8 |
| `n_busy_slots_per_decode` | 3.75 | 1.91 | -1.84 |
| `requests_deferred` | 0 | 0 | 0 |
| `n_tokens_max` | 139 | 139 | 0 |

0 requests deferred, KV cache đủ. Throughput prompt giảm 44% dưới tải (155→86 t/s) do chia sẻ GPU bandwidth giữa các slots. Screenshot: `08-prometheus-metrics.png`.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub: localhost only
- **N17 (Data pipeline):** stub: in-memory dict
- **N18 (Lakehouse):** stub: TOY_DOCS keyword matching
- **N19 (Vector + Feature Store):** stub: TOY_DOCS keyword overlap retrieval

**Nơi tốn nhiều ms nhất** trong pipeline (đo bằng `time.perf_counter` trong `pipeline.py`):

- embed: N/A (keyword-based retrieval, không có embedder)
- retrieve: <1ms (keyword overlap trên 5 documents)
- llama-server: 543–5028ms (phụ thuộc query complexity)

**Reflection** (≤ 60 chữ): bottleneck nằm ở đâu? Có khớp với kỳ vọng không?

Bottleneck hoàn toàn ở LLM inference (99.9%+ thời gian). Đúng với kỳ vọng — retrieval trên toy dataset 5 docs gần như instant, trong khi llama-server trên 0.5B model + Vulkan iGPU mất 0.5–5s mỗi request. Trong production, embedding search sẽ tốn thêm 50–500ms.

---

## 5. Bonus — The single change that mattered most

> **Most important section.** Pick **một** thay đổi từ bonus track (build flag, thread sweep, quant pick, GPU offload, KV-cache quantization, speculative decoding, bất cứ challenge nào trong `BONUS-llama-cpp-optimization/CHALLENGES.md`) đã tạo ra speedup lớn nhất trên máy bạn.

**Change:** Build llama.cpp từ source với Vulkan backend + GPU offload tuning (`-ngl 24` thay vì `-ngl 0` CPU-only hoặc `-ngl 99` full offload)

**Before vs after** (paste 2-3 dòng từ sweep output):

```
ngl=0  (CPU-only):  pp512=1012 t/s,  tg128=35.7 t/s
ngl=24 (optimal):   pp512=1332 t/s,  tg128=31.2 t/s
ngl=99 (full GPU):  pp512= 404 t/s,  tg128=33.6 t/s
speedup (pp512): ~1.32× vs CPU-only, ~3.3× vs full offload
```

**Tại sao nó work** (1–2 đoạn ngắn — đây là phần grader đọc kỹ nhất):

Intel Iris Xe là integrated GPU chia sẻ unified memory với CPU. Với `-ngl 99` (full offload), tất cả layer được đẩy lên GPU nhưng GPU không có VRAM riêng — nó vẫn phải đọc/ghi qua system RAM. Điều này tạo ra bottleneck ở memory bus: GPU phải cạnh tranh băng thông với CPU, dẫn đến prompt processing giảm mạnh (404 t/s vs 1012 t/s CPU-only).

`-ngl 24` là sweet spot vì: (1) đủ layer trên GPU để tận dụng các Vulkan compute shader cho matrix multiply (tăng pp512 lên 1332 t/s), (2) không quá nhiều layer khiến memory bus bão hòa, (3) CPU vẫn xử lý các layer còn lại song song. Đây chính là hiện tượng "partial offload beats full offload" trong slide deck §3 — khi model fit trong RAM nhưng không có VRAM riêng, split inference tối ưu hơn full GPU offload.

Điều ngược với kỳ vọng: deck nói `-ngl 99` thường nhanh nhất. Trên iGPU unified memory, nó lại chậm nhất. Đây là insight quan trọng cho edge deployment — không phải cứ "more GPU layers" là tốt hơn.

### Bonus Challenge C7 — CPU Instruction Set Archaeology

**Setup:** Build 3 phiên bản llama.cpp từ source: Generic (`-DGGML_NATIVE=OFF`), AVX2 (`-DGGML_AVX2=ON`), Native (`-DGGML_NATIVE=ON`). Benchmark với Vulkan backend, ngl=24.

| Build flag | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| Generic (no AVX) | 1264.4 | 35.0 |
| AVX2 only | 563.1 (±593) | 31.3 |
| Native (march=native) | 1316.4 | 31.6 |

**Insight:** Với Vulkan GPU offload (ngl=24), CPU instruction set tuning có ảnh hưởng không đáng kể đến generation speed (31–35 tok/s). Prompt processing có chênh lệch nhỏ (~4% giữa Native và Generic). Lý do: khi compute chính nằm trên GPU (Vulkan shader cores), CPU chỉ làm data marshaling và layer không offload — những tác vụ này ít hưởng lợi từ AVX2. Điều này trái ngược với CPU-only build, nơi AVX2/Native có thể tăng tốc 20–40%. Kết luận: với GPU inference, thời gian nên đầu tư vào tuning GPU backend (ngl, batch size) hơn là CPU flags.

### Bonus Challenge C1 — Speculative Decoding

**Setup:** Target: Qwen2.5-1.5B Q4_K_M (ngl=24, Vulkan GPU). Draft: Qwen2.5-0.5B Q4_K_M (ngld=0, CPU 4 threads). Prompt: "Count from 1 to 10, one per line:" — 64 tokens.

| Config | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| Target only (1.5B, ngl=24) | 127.4 | **9.8** |
| Target(1.5B) + Draft(0.5B, CPU) | 77.3 | **7.4** |

**Speedup: 0.76x** — speculative decoding làm chậm 24% thay vì tăng tốc.

**Tại sao thất bại:** Trên iGPU unified memory, chạy đồng thời 2 model (target trên GPU + draft trên CPU) gây memory bandwidth contention — cả 2 model chia sẻ cùng DRAM bus. Draft model (0.5B) không đủ nhanh trên CPU 4 threads để bù đắp overhead của việc chạy target verify. Kết quả: prompt giảm 39% (127→77 t/s), generation giảm 24% (9.8→7.4 t/s).

**Insight:** Speculative decoding chỉ có lợi khi draft model *nhanh hơn rất nhiều* so với target (vd: 0.1B draft + 7B target) và có dedicated VRAM riêng. Trên edge device với unified memory, kỹ thuật này phản tác dụng — chi phí memory bandwidth > lợi ích từ token dự đoán trúng.

### Bonus Challenge C4 — Best-of-N Parallel Sampling + Reranker

**Setup:** Qwen2.5-0.5B Q4_K_M, temperature=0.8, max_tokens=64, heuristic reranker (length + diversity + repetition penalty + completeness), `multiprocessing.Pool` với mỗi worker `cores // N` threads.

| N | threads/worker | Avg wall (ms) | Avg quality score | Score gain vs N=1 |
|--:|---:|---:|---:|---:|
| 1 | 12 | 4455 | 51.0 | +0.0% |
| 2 | 6 | 5855 | 58.9 | +15.5% |
| 4 | 3 | 9981 | 61.9 | +21.4% |
| 8 | 1 | 13938 | 62.0 | +21.6% |

**Tại sao work:** Best-of-N sampling đánh đổi throughput lấy quality — sinh N responses song song rồi chọn response tốt nhất qua heuristic reranker. Trên 12-core CPU, N workers chạy song song nên wall-clock time tăng sub-linearly (N=4 chỉ ~2.2× chậm hơn N=1, không phải 4×). Heuristic scorer ưu tiên câu trả lời dài hơn, đa dạng hơn, ít lặp hơn — tương quan tốt với chất lượng thực tế.

**Diminishing returns:** Quality plateau ở N=4→8 (61.9→62.0, chỉ +0.1%) trong khi latency tăng 40% (9981→13938ms). Sweet spot cho model này là N=2 hoặc N=4.

**Production relevance:** Cùng ý tưởng với "reject sampling" / "best-of-N" trong RLHF pipelines (vd: InstructGPT paper). Trên GPU, parallel sampling dùng batch inference thay vì multiprocessing, nhưng nguyên lý giống hệt — dùng throughput dư thừa để tăng output quality.

### Bonus Challenge C3 — Multi-LoRA Serving

**Setup:** Base: Qwen2.5-0.5B Q4_K_M. 2 LoRA adapters: `pubmedqa` (r=16, 2 modules q_proj+v_proj, 2.1 MiB) + `chatbot` (r=4, 7 modules, 4.2 MiB). Convert từ PEFT → GGUF bằng `convert_lora_to_gguf.py`. Server: `--lora-init-without-apply`, adapter áp dụng per-request qua API `lora` field.

| Scenario | TTFT P50 (ms) | TTFT P95 (ms) | TPOT P50 (ms) | Errors |
|---|---|---|---|---|
| baseline (no LoRA) | 295 | 378 | 122.6 | 0 |
| lora-0 (pubmedqa, stable) | 119 | 311 | 91.5 | 0 |
| lora-1 (chatbot, stable) | 140 | 1386 | 108.3 | 0 |
| alternating (switching) | 125 | **1202** | 100.7 | 0 |

**Per-request switching cost:**

| Switch | First-request TTFT | Stable TTFT | Overhead |
|---|---|---|---|
| none → lora-0 (2 modules) | ~1082 ms | ~97 ms | **+985 ms** |
| lora-0 → lora-1 (7 modules) | ~1386 ms | ~110 ms | **+1275 ms** |
| lora-1 → none | ~328 ms | ~125 ms | **+195 ms** |

**Tại sao có switching cost:** Mỗi lần đổi adapter, llama.cpp phải: (1) unload adapter cũ, (2) load LoRA A/B matrices vào target layers, (3) recompute merged weights. Cost tỷ lệ với số module: chatbot (7 modules, 4.2 MiB) chậm hơn pubmedqa (2 modules, 2.1 MiB). Unload (lora→none) nhanh hơn vì chỉ cần gỡ adapter, không cần load lại.

**So sánh với Punica/S-LoRA (deck §4):** Hệ thống Punica/S-LoRA đạt switching cost gần 0 nhờ segmented matrix multiplication — chạy nhiều LoRA batch đồng thời qua CUDA kernel fusion. llama.cpp trên Vulkan không có fused kernel này, nên mỗi switch cần materialization step. Overhead ~1s phù hợp với dự đoán cho adapter lưu trên RAM.

**Insight:** Với single user, switching cost ~1s chấp nhận được. Với multi-tenant serving (mỗi request chọn adapter khác nhau), P95 TTFT tăng ~4× (311→1202ms) do switching tail. Đây chính là động lực cho fused multi-LoRA serving mà deck mô tả.

---

## 6. (Optional) Điều ngạc nhiên nhất

_(1–2 câu — không bắt buộc, nhưng người grader đọc tất cả)_

Bất ngờ nhất: Vulkan trên Intel Iris Xe iGPU cho prompt processing speedup 32% so với CPU-only, nhưng generation hầu như không cải thiện (thậm chí chậm hơn). Lý do: prefill là compute-bound (matrix multiply song song tốt trên GPU shader cores), còn decode là memory-bandwidth-bound (sequential token-by-token, GPU phải đợi RAM giống CPU). Điều này giải thích tại sao disaggregated prefill/decode serving (§3 deck) quan trọng — tách riêng 2 workload để tối ưu khác nhau.

---

## 7. Self-graded checklist

- [x] `hardware.json` đã commit
- [x] `models/active.json` đã commit (hoặc paste path snapshot vào section 1)
- [x] `benchmarks/01-quickstart-results.md` đã commit
- [x] `benchmarks/02-server-results.md` (hoặc CSV từ `record-metrics.py`) đã commit
- [x] `benchmarks/bonus-*.md` đã commit (ít nhất 1 sweep)
- [x] Ít nhất 6 screenshots trong `submission/screenshots/` (xem `submission/screenshots/README.md`)
- [x] `make verify` exit 0 (chạy ngay trước khi push)
- [ ] Repo trên GitHub ở chế độ **public**
- [ ] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.
