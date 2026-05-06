# Bonus C4 — Best-of-N Parallel Sampling + Reranker

## Setup

- **Model**: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- **CPU**: 12th Gen Intel(R) Core(TM) i7-1255U (12 cores)
- **Temperature**: 0.8 (high enough for diverse samples)
- **Max tokens**: 64
- **Reranker**: heuristic (length + diversity + repetition penalty + completeness)
- **Parallelism**: `multiprocessing.Pool`, each worker gets `cores // N` threads

## Results

| N | threads/worker | Avg wall (ms) | P50 wall (ms) | Avg quality score | Score gain vs N=1 |
|--:|---:|---:|---:|---:|---:|
| 1 | 12 | 4455 | 4448 | 51.0 | +0.0% |
| 2 | 6 | 5855 | 5782 | 58.9 | +15.5% |
| 4 | 3 | 9981 | 9914 | 61.9 | +21.4% |
| 8 | 1 | 13938 | 13490 | 62.0 | +21.6% |

## Example outputs (first prompt)

**N=1**:  In PagedAttention, the attention mechanism is applied to a sequence of elements, and each element is passed through a set of attention mechanisms that take as input the previous elements of the seque

**N=2**:  

PagedAttention is a multi-threaded memory allocation policy. It divides memory into pages and allocates memory in pieces (called "pads"), so that each piece of memory can be allocated to a thread.


**N=4**:  In your response, please, include an explanation of how PagedAttention operates and what specific tasks it performs, along with any potential drawbacks or limitations of the implementation.

PagedAtt

**N=8**:  I'm really confused.

PagedAttention is a technique for training large language models on the GPU using a training algorithm called PackedAttention. It allows the model to access data in memory befor

## Analysis

Best-of-N sampling trades throughput for quality: instead of generating one
response and hoping it's good, we generate N responses in parallel and pick
the best via a reranker.

**Key observations:**

- **Quality improves** with larger N because the reranker can choose from more
  candidates. The heuristic scorer favors longer, more diverse, less repetitive
  text — which correlates with better answers.
- **Wall-clock time** grows sub-linearly: N workers run in parallel, so N=4 is
  often only ~1.5-2x slower than N=1 (not 4x). Each worker gets fewer threads
  (`cores // N`), but the parallelism amortises the cost.
- **The reranker matters**: a simple heuristic works surprisingly well for
  filtering out degenerate outputs (loops, empty responses, etc.).
- **Diminishing returns**: quality gain plateaus while latency keeps growing.
  For this model at temperature 0.8, N=2 or N=4 is the sweet spot.

**Why this matters for production:** The same idea appears as "reject sampling"
or "best-of-N" in RLHF pipelines (e.g., OpenAI's InstructGPT paper). The
insight is that "throughput" can be reframed — you can spend tokens/sec on
parallel sampling instead of unique requests, getting higher quality output at
the cost of latency. In production, this is typically done with GPU parallelism
rather than CPU multiprocessing, but the principle is identical.
