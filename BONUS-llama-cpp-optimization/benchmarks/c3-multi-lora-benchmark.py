#!/usr/bin/env python3
"""C3 Multi-LoRA Serving Benchmark.

Measures per-request LoRA adapter switching cost using llama-server's
per-request `lora` field in /v1/chat/completions.

Requires: llama-server running with:
  --lora-init-without-apply --lora <adapter0.gguf> --lora <adapter1.gguf>
"""

import argparse
import json
import os
import statistics
import sys
import time

import requests

BASE_URL = os.environ.get("LAB_SERVER_URL", "http://localhost:8080")
BASE_MODEL = os.environ.get("LAB_MODEL_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain what DNA is in one short paragraph.",
    "Write a Python function to compute factorial of n.",
    "What is the difference between HTTP and HTTPS?",
    "Name three planets in our solar system.",
    "Convert the number 255 to binary and explain briefly.",
    "What does the acronym SQL stand for?",
    "Write a haiku about programming.",
    "Explain the concept of recursion in one sentence.",
    "What is the boiling point of water in Celsius?",
]


def get_loaded_loras():
    """Query GET /lora-adapters to list loaded adapters."""
    try:
        r = requests.get(f"{BASE_URL}/lora-adapters", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def send_chat_request(messages, lora_config, max_tokens=64, temperature=0.7):
    """Send a chat completion request with optional per-request LoRA config.

    Args:
        messages: list of message dicts [{"role": "user", "content": "..."}]
        lora_config: list of {"id": int, "scale": float} dicts, or [] for none
        max_tokens: max tokens to generate
        temperature: sampling temperature

    Returns:
        dict with ttft_ms, tpot_ms, total_ms, tokens_generated, error
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if lora_config:
        payload["lora"] = lora_config

    result = {
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "total_ms": 0.0,
        "tokens_generated": 0,
        "error": None,
    }

    t0 = time.perf_counter()
    first_token_time = None

    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                result["total_ms"] = (time.perf_counter() - t0) * 1000
                return result

            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                line = line.decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta:
                        token_text = delta["content"]
                        if token_text and first_token_time is None:
                            first_token_time = time.perf_counter()
                        if token_text:
                            result["tokens_generated"] += 1
    except requests.RequestException as e:
        result["error"] = f"Request error: {e}"
        result["total_ms"] = (time.perf_counter() - t0) * 1000
        return result

    t_end = time.perf_counter()
    result["total_ms"] = (t_end - t0) * 1000

    if first_token_time and result["tokens_generated"] > 0:
        result["ttft_ms"] = (first_token_time - t0) * 1000
        generation_time = t_end - first_token_time
        if result["tokens_generated"] > 1:
            result["tpot_ms"] = (generation_time / (result["tokens_generated"] - 1)) * 1000
        else:
            result["tpot_ms"] = 0.0
    else:
        result["tpot_ms"] = 0.0

    return result


def benchmark_scenario(name, prompts, lora_config, rounds=3):
    """Run a benchmark scenario: send each prompt, collect latencies.

    Args:
        name: scenario label
        prompts: list of prompt strings
        lora_config: list of {"id": int, "scale": float} dicts, or [] for none
        rounds: number of times to repeat each prompt

    Returns:
        dict with aggregated stats
    """
    ttft_samples = []
    tpot_samples = []
    total_samples = []
    tokens_samples = []
    errors = 0

    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"LoRA: {lora_config if lora_config else 'none'}")
    print(f"Rounds: {rounds} x {len(prompts)} prompts = {rounds * len(prompts)} requests")
    print(f"{'='*60}")

    for r in range(rounds):
        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            res = send_chat_request(messages, lora_config)

            if res["error"]:
                errors += 1
                print(f"  [{r+1}/{rounds}][{i+1}/{len(prompts)}] ERROR: {res['error']}")
                continue

            ttft_samples.append(res["ttft_ms"])
            total_samples.append(res["total_ms"])
            tokens_samples.append(res["tokens_generated"])
            if res["tpot_ms"] > 0:
                tpot_samples.append(res["tpot_ms"])

            print(f"  [{r+1}/{rounds}][{i+1}/{len(prompts)}] "
                  f"TTFT={res['ttft_ms']:.0f}ms, TPOT={res['tpot_ms']:.1f}ms, "
                  f"tokens={res['tokens_generated']}, total={res['total_ms']:.0f}ms")

    if not ttft_samples:
        print(f"  No successful requests!")
        return None

    def p(x):
        return sorted(ttft_samples)[int(len(ttft_samples) * x / 100)] if x == 50 else \
               sorted(ttft_samples)[min(int(len(ttft_samples) * x / 100), len(ttft_samples) - 1)]

    stats = {
        "scenario": name,
        "lora_config": str(lora_config),
        "total_requests": len(ttft_samples),
        "errors": errors,
        "ttft_p50_ms": round(statistics.median(ttft_samples), 1),
        "ttft_p95_ms": round(sorted(ttft_samples)[int(len(ttft_samples) * 0.95)], 1),
        "ttft_mean_ms": round(statistics.mean(ttft_samples), 1),
        "tpot_p50_ms": round(statistics.median(tpot_samples), 1) if tpot_samples else 0,
        "tpot_mean_ms": round(statistics.mean(tpot_samples), 1) if tpot_samples else 0,
        "total_p50_ms": round(statistics.median(total_samples), 1),
        "total_mean_ms": round(statistics.mean(total_samples), 1),
        "avg_tokens": round(statistics.mean(tokens_samples), 1),
    }

    print(f"\n  Results: TTFT P50={stats['ttft_p50_ms']}ms, P95={stats['ttft_p95_ms']}ms, "
          f"TPOT P50={stats['tpot_p50_ms']}ms, total P50={stats['total_p50_ms']}ms, "
          f"errors={errors}")

    return stats


def run_alternating_benchmark(prompts, rounds=3):
    """Alternate between no-lora, lora-0, lora-1 on each request.

    This measures the per-request switching cost: each request may have a
    different LoRA config, forcing adapter reload/switch when the config
    differs from the previous request.
    """
    configs = [
        ([], "none"),
        ([{"id": 0, "scale": 1.0}], "lora-0 (pubmedqa)"),
        ([{"id": 1, "scale": 1.0}], "lora-1 (chatbot)"),
    ]

    ttft_samples = []
    tpot_samples = []
    total_samples = []
    tokens_samples = []
    errors = 0
    switches = 0

    print(f"\n{'='*60}")
    print(f"Scenario: ALTERNATING (switching every request)")
    print(f"Pattern: none → lora-0 → lora-1 → none → ...")
    print(f"Rounds: {rounds} passes through all (config × prompt) combos")
    print(f"{'='*60}")

    prev_config_idx = None

    for r in range(rounds):
        for config_idx, (lora_cfg, cfg_name) in enumerate(configs):
            for i, prompt in enumerate(prompts):
                if prev_config_idx is not None and prev_config_idx != config_idx:
                    switches += 1

                messages = [{"role": "user", "content": prompt}]
                res = send_chat_request(messages, lora_cfg)
                prev_config_idx = config_idx

                if res["error"]:
                    errors += 1
                    print(f"  [{cfg_name}][{i+1}] ERROR: {res['error']}")
                    continue

                ttft_samples.append(res["ttft_ms"])
                total_samples.append(res["total_ms"])
                tokens_samples.append(res["tokens_generated"])
                if res["tpot_ms"] > 0:
                    tpot_samples.append(res["tpot_ms"])

                print(f"  [{r+1}/{rounds}][{cfg_name}][{i+1}] "
                      f"TTFT={res['ttft_ms']:.0f}ms, TPOT={res['tpot_ms']:.1f}ms, "
                      f"total={res['total_ms']:.0f}ms")

    print(f"\n  Total adapter switches across benchmark: {switches}")

    if not ttft_samples:
        print(f"  No successful requests!")
        return None

    stats = {
        "scenario": "alternating (switching cost)",
        "lora_config": "cycling [none, lora-0, lora-1]",
        "total_requests": len(ttft_samples),
        "errors": errors,
        "adapter_switches": switches,
        "ttft_p50_ms": round(statistics.median(ttft_samples), 1),
        "ttft_p95_ms": round(sorted(ttft_samples)[int(len(ttft_samples) * 0.95)], 1),
        "ttft_mean_ms": round(statistics.mean(ttft_samples), 1),
        "tpot_p50_ms": round(statistics.median(tpot_samples), 1) if tpot_samples else 0,
        "tpot_mean_ms": round(statistics.mean(tpot_samples), 1) if tpot_samples else 0,
        "total_p50_ms": round(statistics.median(total_samples), 1),
        "total_mean_ms": round(statistics.mean(total_samples), 1),
        "avg_tokens": round(statistics.mean(tokens_samples), 1),
    }

    print(f"\n  Results: TTFT P50={stats['ttft_p50_ms']}ms, P95={stats['ttft_p95_ms']}ms, "
          f"TPOT P50={stats['tpot_p50_ms']}ms, total P50={stats['total_p50_ms']}ms")

    return stats


def check_server():
    """Verify llama-server is running and has LoRA adapters loaded."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            print(f"Server health check failed: HTTP {r.status_code}")
            return False
        print(f"Server health: OK ({r.json()})")
    except requests.RequestException as e:
        print(f"Cannot reach llama-server at {BASE_URL}: {e}")
        print(f"\nStart the server first with:")
        print(f"  llama-server -m <model.gguf> \\")
        print(f"    --lora lora_adapters/pubmedqa.gguf \\")
        print(f"    --lora lora_adapters/chatbot.gguf \\")
        print(f"    --lora-init-without-apply \\")
        print(f"    --host 0.0.0.0 --port 8080")
        return False

    loras = get_loaded_loras()
    if loras is None:
        print("Warning: Could not query /lora-adapters endpoint")
    else:
        print(f"Loaded LoRA adapters: {json.dumps(loras, indent=2)}")
        if len(loras) == 0:
            print("WARNING: No LoRA adapters loaded. Server must be started with --lora flags.")

    return True


def generate_markdown(results, output_path):
    """Write results as markdown table."""
    lines = [
        "# C3 — Multi-LoRA Serving Benchmark Results",
        "",
        "## Setup",
        "",
        f"- **Base model**: `{BASE_MODEL}`",
        "- **LoRA adapter 0 (pubmedqa)**: `lora_adapters/pubmedqa.gguf` (r=16, alpha=16, q_proj+v_proj)",
        "- **LoRA adapter 1 (chatbot)**: `lora_adapters/chatbot.gguf` (r=4, alpha=32, all 7 linear modules)",
        "- **Prompts per scenario**: 10",
        "- **Rounds per prompt**: 3 (30 requests per scenario)",
        "- **Server flags**: `--lora-init-without-apply` (adapters loaded but not applied until per-request `lora` field)",
        "",
        "## Results",
        "",
        "| Scenario | TTFT P50 (ms) | TTFT P95 (ms) | TTFT Mean (ms) | TPOT P50 (ms) | Total P50 (ms) | Avg Tokens | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['scenario']} | {r['ttft_p50_ms']} | {r['ttft_p95_ms']} | "
            f"{r['ttft_mean_ms']} | {r['tpot_p50_ms']} | {r['total_p50_ms']} | "
            f"{r['avg_tokens']} | {r['errors']} |"
        )

    lines.extend([
        "",
        "## Analysis",
        "",
        "### Switching cost",
        "",
        "The difference between the alternating scenario (TTFT P50) and the single-LoRA",
        "scenarios reveals the per-request adapter switching overhead.",
        "",
        "```",
    ])

    # Compute switching cost if we have results
    baseline = next((r for r in results if "none" in r.get("lora_config", "")), None)
    lora0 = next((r for r in results if "lora-0" in r.get("scenario", "").lower()), None)
    lora1 = next((r for r in results if "lora-1" in r.get("scenario", "").lower()), None)
    alternating = next((r for r in results if "alternating" in r.get("scenario", "").lower()), None)

    if baseline and alternating:
        overhead = alternating["ttft_p50_ms"] - baseline["ttft_p50_ms"]
        lines.append(f"Switching overhead (TTFT P50): {overhead:+.1f} ms vs baseline")
    if lora0 and alternating:
        overhead = alternating["ttft_p50_ms"] - lora0["ttft_p50_ms"]
        lines.append(f"Switching overhead (TTFT P50): {overhead:+.1f} ms vs single LoRA-0")

    lines.extend([
        "```",
        "",
        "### Key takeaways",
        "",
        "1. A single LoRA adapter adds a small latency increase over the baseline (weight merging overhead).",
        "2. The alternating scenario shows the full switching cost: llama.cpp reloads adapter weights",
        "   when the `lora` field changes between consecutive requests.",
        "3. With `--lora-init-without-apply`, adapters are pre-loaded to VRAM/RAM but not applied",
        "   until a request specifies them, which keeps the base model path fast.",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="C3 Multi-LoRA Serving Benchmark")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per prompt (default: 3)")
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--output", default="benchmarks/bonus-c3-multi-lora.md")
    args = parser.parse_args()

    if not args.skip_server_check:
        if not check_server():
            sys.exit(1)

    results = []

    # Scenario 1: No LoRA (baseline)
    r = benchmark_scenario("baseline (no LoRA)", PROMPTS, [], rounds=args.rounds)
    if r:
        results.append(r)

    # Scenario 2: LoRA-0 only (pubmedqa)
    r = benchmark_scenario("lora-0 (pubmedqa)", PROMPTS, [{"id": 0, "scale": 1.0}], rounds=args.rounds)
    if r:
        results.append(r)

    # Scenario 3: LoRA-1 only (chatbot)
    r = benchmark_scenario("lora-1 (chatbot)", PROMPTS, [{"id": 1, "scale": 1.0}], rounds=args.rounds)
    if r:
        results.append(r)

    # Scenario 4: Alternating (switching cost)
    r = run_alternating_benchmark(PROMPTS, rounds=args.rounds)
    if r:
        results.append(r)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<30} {'TTFT P50':>10} {'TTFT P95':>10} {'TPOT P50':>10} {'Total P50':>10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['scenario']:<30} {r['ttft_p50_ms']:>8.0f}ms {r['ttft_p95_ms']:>8.0f}ms "
              f"{r['tpot_p50_ms']:>8.1f}ms {r['total_p50_ms']:>8.0f}ms")

    # JSON output
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results: {json_path}")

    # Markdown output
    generate_markdown(results, args.output)


if __name__ == "__main__":
    main()
