# 02 — llama-server Load Test Results

Server: Qwen2.5-0.5B-Instruct Q4_K_M via llama-cpp-python, Vulkan backend

## Load test: 10 users, 1 minute

| Metric | Value |
|---|---|
| Total requests | 22 |
| Failures | 0 |
| Avg response | 16,952 ms |
| P50 | 20,000 ms |
| P95 | 25,000 ms |
| P99 | 26,000 ms |
| RPS | 0.42 |

## Load test: 50 users, 1 minute

| Metric | Value |
|---|---|
| Total requests | 23 |
| Failures | 0 |
| Avg response | 19,109 ms |
| P50 | 16,000 ms |
| P95 | 49,000 ms |
| P99 | 50,000 ms |
| RPS | 0.39 |

## Observations

- 0 failures in both runs — server never crashed under load
- P95 jumps from 25s (10 users) to 49s (50 users) — queue depth increases with concurrency
- RPS stable (~0.4) — limited by model size and single-server with 4 parallel slots
- KV-cache peak: N/A (Python llama-cpp-python server doesn't expose Prometheus /metrics)
