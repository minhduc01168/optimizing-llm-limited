"""
Lesson 01: Transformer Architecture & VRAM Analysis - SOLUTION
==============================================================
Reference implementation cho giảng viên và học sinh tham khảo
"""

import torch
import time
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml


@dataclass
class ModelMetrics:
    """Data class để lưu metrics của mỗi model"""
    model_name: str
    params_count: str
    vram_before_gb: float
    vram_after_gb: float
    vram_peak_gb: float
    load_time_s: float
    inference_latency_ms: float
    model_size_mb: float


def get_gpu_info() -> Dict:
    """Lấy thông tin GPU"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode('utf-8')
    
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    return {
        "name": gpu_name,
        "cuda_version": torch.version.cuda,
        "total_memory_gb": mem_info.total / 1e9,
        "torch_version": torch.__version__
    }


def get_vram_usage() -> float:
    """Đo VRAM hiện tại đang sử dụng (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_vram_peak() -> float:
    """Đo VRAM peak usage (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def load_model(model_name: str) -> tuple:
    """Load model và tokenizer từ HuggingFace"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def measure_inference_latency(model, tokenizer, prompt: str = "Hello, I am", max_tokens: int = 50) -> float:
    """Đo thời gian inference (ms/token)"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return (elapsed / num_tokens) * 1000  # ms per token


def get_model_size(model_name: str) -> float:
    """Lấy kích thước model trên disk (MB)"""
    try:
        from transformers.utils.hub import try_to_load_from_cache
        cache_path = try_to_load_from_cache(model_name, "pytorch_model.bin")
        if cache_path and os.path.exists(cache_path):
            return os.path.getsize(cache_path) / 1e6
    except:
        pass
    
    # Fallback: ước tính từ số parameters
    model = AutoModelForCausalLM.from_pretrained(model_name)
    num_params = sum(p.numel() for p in model.parameters())
    del model
    return (num_params * 2) / 1e6  # FP16 = 2 bytes per param


def count_parameters(model) -> str:
    """Đếm số parameters của model"""
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 1e9:
        return f"{total_params/1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params/1e6:.0f}M"
    else:
        return f"{total_params/1e3:.0f}K"


def benchmark_model(model_name: str) -> Optional[ModelMetrics]:
    """Benchmark một model hoàn chỉnh"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # 1. Đo VRAM trước khi load
        vram_before = get_vram_usage()
        print(f"VRAM before load: {vram_before:.2f} GB")
        
        # 2. Load model
        start_time = time.time()
        model, tokenizer = load_model(model_name)
        load_time = time.time() - start_time
        print(f"Load time: {load_time:.2f}s")
        
        # 3. Đo VRAM sau khi load
        vram_after = get_vram_usage()
        vram_peak = get_vram_peak()
        print(f"VRAM after load: {vram_after:.2f} GB")
        print(f"VRAM peak: {vram_peak:.2f} GB")
        
        # 4. Đếm parameters
        params_count = count_parameters(model)
        print(f"Parameters: {params_count}")
        
        # 5. Đo inference latency
        latency = measure_inference_latency(model, tokenizer)
        print(f"Inference latency: {latency:.2f} ms/token")
        
        # 6. Lấy model size
        model_size = get_model_size(model_name)
        print(f"Model size: {model_size:.2f} MB")
        
        # 7. Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return ModelMetrics(
            model_name=model_name,
            params_count=params_count,
            vram_before_gb=vram_before,
            vram_after_gb=vram_after,
            vram_peak_gb=vram_peak,
            load_time_s=load_time,
            inference_latency_ms=latency,
            model_size_mb=model_size
        )
        
    except Exception as e:
        print(f"Error benchmarking {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_benchmarks() -> List[ModelMetrics]:
    """Chạy benchmark cho tất cả models"""
    MODELS = [
        "distilgpt2",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "facebook/opt-1.3b",
    ]
    
    results = []
    for model_name in MODELS:
        metrics = benchmark_model(model_name)
        if metrics:
            results.append(metrics)
    
    return results


def save_results(results: List[ModelMetrics], output_path: str = "results/metrics.json"):
    """Lưu kết quả benchmark vào file JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "lesson": "01-transformer-vram",
        "gpu_info": get_gpu_info(),
        "benchmarks": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function"""
    print("="*60)
    print("Lesson 01: Transformer Architecture & VRAM Analysis")
    print("="*60)
    
    # 1. Hiển thị thông tin GPU
    gpu_info = get_gpu_info()
    print(f"\nGPU Info:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    # 2. Chạy benchmarks
    results = run_all_benchmarks()
    
    # 3. Lưu kết quả
    save_results(results)
    
    # 4. In bảng tóm tắt
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Params':<10} {'VRAM (GB)':<12} {'Latency (ms)':<15}")
    print("-"*60)
    for r in results:
        print(f"{r.model_name:<25} {r.params_count:<10} {r.vram_after_gb:<12.2f} {r.inference_latency_ms:<15.2f}")
    
    # 5. So sánh với baseline
    if results:
        baseline = results[0]  # distilgpt2
        print(f"\n{'='*60}")
        print("COMPARISON WITH BASELINE (distilgpt2)")
        print("="*60)
        for r in results[1:]:
            vram_ratio = r.vram_after_gb / baseline.vram_after_gb
            latency_ratio = r.inference_latency_ms / baseline.inference_latency_ms
            print(f"{r.model_name}: VRAM {vram_ratio:.1f}x, Latency {latency_ratio:.1f}x")


if __name__ == "__main__":
    main()
