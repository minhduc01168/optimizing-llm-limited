"""
Lesson 05: Continuous Batching & Scheduling
=============================================
So sánh Static Batching vs Continuous Batching

Hardware: NVIDIA T4 (16GB VRAM)
"""

import time
import json
import os
import asyncio
from typing import Dict, List
from dataclasses import dataclass, asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BatchingMetrics:
    """Metrics cho batching comparison"""
    method: str  # "static" hoặc "continuous"
    batch_size: int
    total_requests: int
    total_tokens: int
    elapsed_time_s: float
    throughput_tok_per_sec: float
    batch_utilization_pct: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    oom_errors: int


# ============================================================
# Static Batching
# ============================================================

def run_static_batch(model, tokenizer, prompts: List[str], max_tokens: int = 50) -> Dict:
    """
    Chạy static batching

    Args:
        model: Model đã load
        tokenizer: Tokenizer
        prompts: List of input prompts
        max_tokens: Max tokens to generate

    Returns: Dict chứa metrics
    """
    batch_size = len(prompts)
    latencies = []

    # 1. Tokenize ALL prompts cùng lúc (padding)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=3, do_sample=False)

    # 2. Generate cho toàn bộ batch
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # 3. Đếm tokens generated
    total_tokens = 0
    for i in range(len(prompts)):
        input_len = (inputs["attention_mask"][i] == 1).sum().item()
        output_len = outputs.shape[1]
        total_tokens += output_len - input_len

    avg_latency = (elapsed / len(prompts)) * 1000  # ms per request

    # 4. Tính utilization (trong static batching, batch luôn đầy)
    utilization = 100.0

    return {
        "batch_size": batch_size,
        "total_requests": len(prompts),
        "total_tokens": total_tokens,
        "elapsed_time_s": elapsed,
        "throughput_tok_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
        "batch_utilization_pct": utilization,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": avg_latency * 1.1,  # Ước tính
        "p99_latency_ms": avg_latency * 1.2,  # Ước tính
        "oom_errors": 0
    }


# ============================================================
# Continuous Batching (Simulated)
# ============================================================

def run_continuous_batch(model, tokenizer, prompts: List[str], max_tokens: int = 50, chunk_size: int = 4) -> Dict:
    """
    Chạy continuous batching (simulated trên single GPU)

    Args:
        model: Model đã load
        tokenizer: Tokenizer
        prompts: List of input prompts
        max_tokens: Max tokens to generate
        chunk_size: Số requests xử lý đồng thời

    Returns: Dict chứa metrics
    """
    total_start = time.perf_counter()
    all_latencies = []
    total_tokens = 0
    oom_errors = 0
    gpu_active_time = 0

    # Chia prompts thành chunks
    chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    for chunk_idx, chunk in enumerate(chunks):
        try:
            # 1. Tokenize chunk
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 2. Generate cho chunk
            chunk_start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

            torch.cuda.synchronize()
            chunk_elapsed = time.perf_counter() - chunk_start
            gpu_active_time += chunk_elapsed

            # 3. Đếm tokens
            for i in range(len(chunk)):
                input_len = (inputs["attention_mask"][i] == 1).sum().item()
                output_len = outputs.shape[1]
                num_tokens = output_len - input_len
                total_tokens += num_tokens
                all_latencies.append(chunk_elapsed / len(chunk) * 1000)  # ms per request

            # 4. Simulate continuous batching: giải phóng bộ nhớ ngay sau khi chunk xong
            del outputs, inputs
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            oom_errors += 1
            torch.cuda.empty_cache()
            print(f"OOM ở chunk {chunk_idx}, bỏ qua...")
        except Exception as e:
            oom_errors += 1
            print(f"Lỗi ở chunk {chunk_idx}: {e}")

    total_elapsed = time.perf_counter() - total_start

    # Tính utilization: thời gian GPU active / total time
    utilization = (gpu_active_time / total_elapsed * 100) if total_elapsed > 0 else 0

    all_latencies.sort()
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    p95_idx = int(len(all_latencies) * 0.95)
    p99_idx = int(len(all_latencies) * 0.99)

    return {
        "batch_size": chunk_size,
        "total_requests": len(prompts),
        "total_tokens": total_tokens,
        "elapsed_time_s": total_elapsed,
        "throughput_tok_per_sec": total_tokens / total_elapsed if total_elapsed > 0 else 0,
        "batch_utilization_pct": utilization,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": all_latencies[p95_idx] if all_latencies else 0,
        "p99_latency_ms": all_latencies[p99_idx] if all_latencies else 0,
        "oom_errors": oom_errors
    }


# ============================================================
# Benchmark
# ============================================================

def generate_test_prompts(n: int = 100) -> List[str]:
    """Tạo test prompts với độ dài khác nhau"""
    base_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Scientists have discovered that",
        "The most important lesson I learned is",
        "When we think about the universe,",
    ]
    
    prompts = []
    for i in range(n):
        prompts.append(base_prompts[i % len(base_prompts)])
    
    return prompts


def benchmark_batching(model_name: str = "facebook/opt-1.3b") -> List[BatchingMetrics]:
    """
    Benchmark so sánh static và continuous batching
    """
    results = []
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    prompts = generate_test_prompts(50)
    
    # Static batching
    print("\n" + "="*50)
    print("Static Batching")
    print("="*50)
    static_result = run_static_batch(model, tokenizer, prompts)
    if static_result:
        results.append(BatchingMetrics(method="static", **static_result))
    
    # Continuous batching
    print("\n" + "="*50)
    print("Continuous Batching")
    print("="*50)
    continuous_result = run_continuous_batch(model, tokenizer, prompts)
    if continuous_result:
        results.append(BatchingMetrics(method="continuous", **continuous_result))
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return results


def save_results(results: List[BatchingMetrics], output_path: str = "results/metrics.json"):
    """Lưu kết quả"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "lesson": "05-continuous-batching",
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_comparison(results: List[BatchingMetrics]):
    """In bảng so sánh"""
    if len(results) < 2:
        return
    
    static = results[0]
    continuous = results[1]
    
    print(f"\n{'='*60}")
    print("BATCHING COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Static':<15} {'Continuous':<15} {'Improvement':<15}")
    print("-"*60)
    
    throughput_imp = ((continuous.throughput_tok_per_sec / static.throughput_tok_per_sec) - 1) * 100
    utilization_imp = continuous.batch_utilization_pct - static.batch_utilization_pct
    
    print(f"{'Throughput (tok/s)':<25} {static.throughput_tok_per_sec:<15.1f} {continuous.throughput_tok_per_sec:<15.1f} {throughput_imp:+.1f}%")
    print(f"{'Utilization (%)':<25} {static.batch_utilization_pct:<15.1f} {continuous.batch_utilization_pct:<15.1f} {utilization_imp:+.1f}%")


def main():
    print("="*60)
    print("Lesson 05: Continuous Batching & Scheduling")
    print("="*60)
    
    MODEL = "facebook/opt-1.3b"  # T4-safe
    results = benchmark_batching(MODEL)
    save_results(results)
    print_comparison(results)


if __name__ == "__main__":
    main()
