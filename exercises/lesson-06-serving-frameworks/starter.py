"""
Lesson 06: Serving Frameworks Comparison
==========================================
Benchmark vLLM, TGI, llama.cpp, FastAPI

Hardware: NVIDIA T4 (16GB VRAM)
"""

import time
import json
import os
import asyncio
from typing import Dict, List
from dataclasses import dataclass, asdict
import subprocess


@dataclass
class FrameworkMetrics:
    """Metrics cho mỗi framework"""
    framework: str
    model: str
    requests_per_sec: float
    latency_p50_ms: float
    latency_p99_ms: float
    ttft_p50_ms: float
    vram_usage_gb: float
    cold_start_s: float
    image_size_gb: float  # Docker image size
    success_rate: float


# ============================================================
# Framework Launchers
# ============================================================

def start_vllm(model_name: str, port: int = 8000) -> tuple:
    """
    Khởi động vLLM server
    Returns: (process, base_url)
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base_url = f"http://localhost:{port}"
    return process, base_url


def start_tgi(model_name: str, port: int = 8001) -> tuple:
    """
    Khởi động TGI server
    Returns: (process, base_url)
    """
    cmd = [
        "docker", "run", "--gpus", "all",
        "-p", f"{port}:80",
        "--shm-size", "1g",
        "ghcr.io/huggingface/text-generation-inference:latest",
        "--model-id", model_name,
        "--max-input-length", "1024",
        "--max-total-tokens", "2048",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base_url = f"http://localhost:{port}"
    return process, base_url


def start_llama_cpp(model_path: str, port: int = 8002) -> tuple:
    """
    Khởi động llama.cpp server
    Returns: (process, base_url)
    """
    cmd = [
        "llama-server",
        "-m", model_path,
        "--port", str(port),
        "-c", "2048",
        "--n-gpu-layers", "32",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base_url = f"http://localhost:{port}"
    return process, base_url


def start_fastapi(model_name: str, port: int = 8003) -> tuple:
    """
    Khởi động FastAPI server
    Returns: (process, base_url)
    """
    import tempfile

    app_code = '''
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import uvicorn

app = FastAPI(title="LLM Serving API")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

class CompletionResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("Model loaded!")

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    start = time.perf_counter()
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True)
    elapsed = (time.perf_counter() - start) * 1000
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return CompletionResponse(text=text, tokens_generated=num_tokens, latency_ms=elapsed)

@app.get("/health")
async def health():
    return {{"status": "healthy"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''.format(model_name=model_name, port=port)

    # Tạo temporary file cho app
    app_file = os.path.join(tempfile.gettempdir(), f"llm_app_{port}.py")
    with open(app_file, 'w') as f:
        f.write(app_code)

    process = subprocess.Popen(
        ["python", app_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    base_url = f"http://localhost:{port}"
    return process, base_url


# ============================================================
# Benchmark Functions
# ============================================================

async def benchmark_endpoint(url: str, n_requests: int = 100, concurrency: int = 10) -> Dict:
    """
    Benchmark một endpoint

    Returns: Dict với throughput, latency percentiles, success rate
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

    async def send_one(session, idx):
        async with semaphore:
            prompt = prompts[idx % len(prompts)]
            payload = {
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.7
            }
            start = time.perf_counter()
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        latency = (time.perf_counter() - start) * 1000
                        return {"latency_ms": latency, "error": None}
                    else:
                        return {"latency_ms": 0, "error": f"HTTP {resp.status}"}
            except Exception as e:
                return {"latency_ms": 0, "error": str(e)}

    total_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [send_one(session, i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - total_start

    successful = [r for r in results if r["error"] is None]
    latencies = sorted([r["latency_ms"] for r in successful])

    if not latencies:
        return {"throughput": 0, "latency_p50": 0, "latency_p99": 0, "ttft_p50": 0, "success_rate": 0, "vram_gb": 0}

    vram_gb = 0
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass

    return {
        "throughput": len(successful) / total_time if total_time > 0 else 0,
        "latency_p50": latencies[len(latencies) // 2],
        "latency_p99": latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[-1],
        "ttft_p50": latencies[len(latencies) // 2] * 0.3,  # Ước tính TTFT ~ 30% latency
        "success_rate": len(successful) / n_requests * 100,
        "vram_gb": vram_gb
    }


def measure_cold_start(start_fn, *args) -> tuple:
    """
    Đo thời gian cold start
    Returns: (process, cold_start_time)
    """
    start = time.time()
    process, base_url = start_fn(*args)

    # Đợi server sẵn sàng
    import urllib.request
    import urllib.error

    health_url = f"{base_url}/health"
    timeout = 120

    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(health_url, timeout=5)
            if req.status == 200:
                cold_start_time = time.time() - start
                return process, cold_start_time
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
            pass
        time.sleep(0.5)

    cold_start_time = time.time() - start
    print(f"Warning: Server không phản hồi health check sau {timeout}s")
    return process, cold_start_time


def cleanup_process(process):
    """Dừng process"""
    try:
        process.terminate()
        process.wait(timeout=10)
    except Exception:
        process.kill()


# ============================================================
# Main Benchmark
# ============================================================

def run_comparison(model_name: str = "facebook/opt-2.7b") -> List[FrameworkMetrics]:
    """Chạy benchmark cho tất cả frameworks"""
    results = []
    
    frameworks = [
        ("vLLM", start_vllm, 8000),
        # ("TGI", start_tgi, 8001),  # Uncomment nếu có Docker
        # ("llama.cpp", start_llama_cpp, 8002),  # Uncomment nếu có GGUF model
        ("FastAPI", start_fastapi, 8003),
    ]
    
    for name, start_fn, port in frameworks:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")
        
        process = None
        try:
            # 1. Cold start
            process, cold_start = measure_cold_start(start_fn, model_name)
            
            # 2. Benchmark
            url = f"http://localhost:{port}/v1/completions"
            bench_results = asyncio.run(benchmark_endpoint(url))
            
            # 3. Compile metrics
            metrics = FrameworkMetrics(
                framework=name,
                model=model_name,
                requests_per_sec=bench_results.get("throughput", 0),
                latency_p50_ms=bench_results.get("latency_p50", 0),
                latency_p99_ms=bench_results.get("latency_p99", 0),
                ttft_p50_ms=bench_results.get("ttft_p50", 0),
                vram_usage_gb=bench_results.get("vram_gb", 0),
                cold_start_s=cold_start,
                image_size_gb=0,  # TODO: Measure Docker image size
                success_rate=bench_results.get("success_rate", 0)
            )
            results.append(metrics)
            
        except Exception as e:
            print(f"Error with {name}: {e}")
        
        finally:
            if process is not None:
                cleanup_process(process)
    
    return results


def save_results(results: List[FrameworkMetrics], output_path: str = "results/metrics.json"):
    """Lưu kết quả"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "lesson": "06-serving-frameworks",
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_comparison_table(results: List[FrameworkMetrics]):
    """In bảng so sánh"""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("SERVING FRAMEWORKS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Framework':<12} {'Req/s':<10} {'Lat p50':<10} {'Lat p99':<10} {'TTFT p50':<10} {'VRAM':<10} {'Cold Start':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r.framework:<12} {r.requests_per_sec:<10.1f} {r.latency_p50_ms:<10.0f} {r.latency_p99_ms:<10.0f} {r.ttft_p50_ms:<10.0f} {r.vram_usage_gb:<10.1f} {r.cold_start_s:<12.1f}")


def main():
    print("="*60)
    print("Lesson 06: Serving Frameworks Comparison")
    print("="*60)
    
    MODEL = "facebook/opt-2.7b"
    results = run_comparison(MODEL)
    save_results(results)
    print_comparison_table(results)
    
    # Find winner
    if results:
        best_throughput = max(results, key=lambda x: x.requests_per_sec)
        best_latency = min(results, key=lambda x: x.latency_p50_ms)
        
        print(f"\n{'='*60}")
        print("WINNERS")
        print(f"{'='*60}")
        print(f"Best Throughput: {best_throughput.framework} ({best_throughput.requests_per_sec:.1f} req/s)")
        print(f"Best Latency: {best_latency.framework} ({best_latency.latency_p50_ms:.0f}ms)")


if __name__ == "__main__":
    main()
