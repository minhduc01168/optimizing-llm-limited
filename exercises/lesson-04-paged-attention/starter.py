"""
Lesson 04: PagedAttention & vLLM
==================================
Benchmark vLLM với PagedAttention trên T4

Hardware: NVIDIA T4 (16GB VRAM)
"""

import time
import json
import os
import asyncio
import torch
from typing import Dict, List
from dataclasses import dataclass, asdict
import subprocess


@dataclass
class vLLMMetrics:
    """Metrics từ vLLM benchmark"""
    model: str
    max_concurrent: int
    throughput_req_per_sec: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    latency_p50_ms: float
    latency_p99_ms: float
    cache_hit_rate: float
    vram_usage_gb: float
    total_requests: int
    successful_requests: int


# ============================================================
# vLLM Server Management
# ============================================================

def start_vllm_server(model_name: str, port: int = 8000, gpu_memory_utilization: float = 0.9) -> subprocess.Popen:
    """
    Khởi động vLLM server

    Args:
        model_name: Tên model từ HuggingFace
        port: Port để serve
        gpu_memory_utilization: Phần trăm GPU memory sử dụng

    Returns: Process object
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", "2048",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def wait_for_server(url: str = "http://localhost:8000/health", timeout: int = 120):
    """
    Đợi server sẵn sàng

    Args:
        url: Health check URL
        timeout: Timeout in seconds
    """
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=5)
            if req.status == 200:
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Server không khởi động trong {timeout}s")


def stop_vllm_server(process: subprocess.Popen):
    """Dừng vLLM server"""
    process.terminate()
    process.wait()


# ============================================================
# Benchmark Functions
# ============================================================

async def send_request(session, url: str, prompt: str, max_tokens: int = 50) -> Dict:
    """
    Gửi một request đến vLLM server

    Returns: Dict với latency, ttft, tokens_generated
    """
    import aiohttp

    payload = {
        "model": "default",  # vLLM sử dụng model đã load sẵn
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }

    ttft = None
    tokens_generated = 0
    start_time = time.perf_counter()

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return {
                    "latency_ms": 0,
                    "ttft_ms": 0,
                    "tokens_generated": 0,
                    "error": f"HTTP {response.status}"
                }

            # Đọc streaming response
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    import json
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        if text:
                            tokens_generated += 1
                            if ttft is None:
                                ttft = (time.perf_counter() - start_time) * 1000
                except Exception:
                    continue

        total_latency = (time.perf_counter() - start_time) * 1000

        return {
            "latency_ms": total_latency,
            "ttft_ms": ttft if ttft else total_latency,
            "tokens_generated": tokens_generated,
            "error": None
        }

    except Exception as e:
        return {
            "latency_ms": 0,
            "ttft_ms": 0,
            "tokens_generated": 0,
            "error": str(e)
        }


async def run_load_test(url: str, n_requests: int = 100, concurrency: int = 10) -> Dict:
    """
    Chạy load test với nhiều concurrent requests

    Returns: Dict chứa throughput, latency percentiles, etc.
    """
    import aiohttp
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "Scientists have discovered that",
        "The most important lesson I learned is",
        "When we think about the universe,",
    ]

    async def bounded_request(session, idx):
        async with semaphore:
            prompt = prompts[idx % len(prompts)]
            return await send_request(session, url, prompt)

    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session, i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # Tính toán metrics
    successful = [r for r in results if r["error"] is None]
    latencies = [r["latency_ms"] for r in successful]
    ttfts = [r["ttft_ms"] for r in successful]

    if not latencies:
        return {"throughput": 0, "ttft_p50": 0, "ttft_p99": 0, "latency_p50": 0, "latency_p99": 0, "total": n_requests, "successful": 0}

    latencies.sort()
    ttfts.sort()

    total_tokens = sum(r["tokens_generated"] for r in successful)

    return {
        "throughput": len(successful) / total_time if total_time > 0 else 0,
        "ttft_p50": ttfts[len(ttfts) // 2] if ttfts else 0,
        "ttft_p99": ttfts[int(len(ttfts) * 0.99)] if ttfts else 0,
        "latency_p50": latencies[len(latencies) // 2] if latencies else 0,
        "latency_p99": latencies[int(len(latencies) * 0.99)] if latencies else 0,
        "total": n_requests,
        "successful": len(successful),
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0
    }


def get_vllm_metrics(url: str) -> Dict:
    """
    Lấy metrics từ vLLM server

    Returns: Dict chứa cache hit rate, VRAM usage, etc.
    """
    import urllib.request

    try:
        req = urllib.request.urlopen(url, timeout=10)
        content = req.read().decode('utf-8')

        metrics = {}
        for line in content.split('\n'):
            if line.startswith('#'):
                continue
            if ' ' in line:
                name, value = line.rsplit(' ', 1)
                try:
                    metrics[name] = float(value)
                except Exception:
                    pass

        # Parse vLLM specific metrics
        cache_hit_rate = metrics.get("vllm:cache_hit_rate", 0) * 100
        vram_gb = metrics.get("gpu_memory_usage_bytes", 0) / 1e9 if "gpu_memory_usage_bytes" in metrics else 0

        # Fallback: ước tính VRAM từ torch
        if vram_gb == 0 and torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9

        return {
            "cache_hit_rate": cache_hit_rate,
            "vram_gb": vram_gb,
            "num_requests_running": metrics.get("vllm:num_requests_running", 0),
            "num_requests_waiting": metrics.get("vllm:num_requests_waiting", 0),
            "avg_generation_throughput": metrics.get("vllm:avg_generation_throughput_toks_per_s", 0),
        }

    except Exception as e:
        print(f"Không thể lấy vLLM metrics: {e}")
        # Fallback
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
        return {
            "cache_hit_rate": 0,
            "vram_gb": vram_gb,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
            "avg_generation_throughput": 0,
        }


# ============================================================
# Main Benchmark
# ============================================================

def benchmark_vllm(model_name: str = "facebook/opt-2.7b") -> vLLMMetrics:
    """
    Chạy benchmark vLLM hoàn chỉnh
    """
    print("="*60)
    print(f"Benchmarking vLLM with {model_name}")
    print("="*60)
    
    # 1. Start server
    print("\n[1/4] Starting vLLM server...")
    process = start_vllm_server(model_name)
    wait_for_server()
    print("Server ready!")
    
    try:
        # 2. Run load test
        print("\n[2/4] Running load test...")
        load_results = asyncio.run(run_load_test(
            url="http://localhost:8000/v1/completions",
            n_requests=100,
            concurrency=20
        ))
        
        # 3. Get vLLM metrics
        print("\n[3/4] Collecting metrics...")
        vllm_metrics = get_vllm_metrics("http://localhost:8000/metrics")
        
        # 4. Compile results
        print("\n[4/4] Compiling results...")
        metrics = vLLMMetrics(
            model=model_name,
            max_concurrent=20,
            throughput_req_per_sec=load_results.get("throughput", 0),
            ttft_p50_ms=load_results.get("ttft_p50", 0),
            ttft_p99_ms=load_results.get("ttft_p99", 0),
            latency_p50_ms=load_results.get("latency_p50", 0),
            latency_p99_ms=load_results.get("latency_p99", 0),
            cache_hit_rate=vllm_metrics.get("cache_hit_rate", 0),
            vram_usage_gb=vllm_metrics.get("vram_gb", 0),
            total_requests=load_results.get("total", 0),
            successful_requests=load_results.get("successful", 0)
        )
        
        return metrics
        
    finally:
        # 5. Stop server
        print("\nStopping server...")
        stop_vllm_server(process)


def save_results(metrics: vLLMMetrics, output_path: str = "results/metrics.json"):
    """Lưu kết quả"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "lesson": "04-paged-attention",
        "metrics": asdict(metrics),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    MODEL = "facebook/opt-2.7b"  # T4-safe model
    
    metrics = benchmark_vllm(MODEL)
    save_results(metrics)
    
    # In kết quả
    print(f"\n{'='*60}")
    print("vLLM BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model: {metrics.model}")
    print(f"Throughput: {metrics.throughput_req_per_sec:.1f} req/s")
    print(f"TTFT p50: {metrics.ttft_p50_ms:.0f} ms")
    print(f"TTFT p99: {metrics.ttft_p99_ms:.0f} ms")
    print(f"Latency p50: {metrics.latency_p50_ms:.0f} ms")
    print(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")
    print(f"VRAM usage: {metrics.vram_usage_gb:.1f} GB")
    print(f"Success rate: {metrics.successful_requests}/{metrics.total_requests}")


if __name__ == "__main__":
    main()
