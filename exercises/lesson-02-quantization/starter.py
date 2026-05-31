"""
Lesson 02: Quantization Techniques
====================================
So sánh các kỹ thuật lượng tử hóa: FP32, FP16, INT8, INT4

Hardware: NVIDIA T4 (16GB VRAM)
"""

import torch
import time
import json
import os
from typing import Dict, List
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class QuantizationMetrics:
    """Metrics cho mỗi quantization method"""
    method: str
    bits: int
    vram_gb: float
    model_size_mb: float
    load_time_s: float
    inference_latency_ms: float
    tokens_per_second: float
    perplexity: float = 0.0  # Optional


# ============================================================
# Quantization Configurations
# ============================================================

def get_fp32_config():
    """FP32 baseline"""
    return {"torch_dtype": torch.float32}


def get_fp16_config():
    """FP16 - half precision"""
    return {"torch_dtype": torch.float16}


def get_int8_config():
    """INT8 - 8-bit quantization"""
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True
        )
    }


def get_int4_config():
    """INT4-NF4 - 4-bit quantization"""
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    }


QUANT_CONFIGS = {
    "fp32": get_fp32_config,
    "fp16": get_fp16_config,
    "int8": get_int8_config,
    "int4": get_int4_config,
}


def load_model_with_quantization(model_name: str, quant_method: str) -> tuple:
    """
    Load model với quantization method chỉ định

    Args:
        model_name: Tên model từ HuggingFace
        quant_method: "fp32", "fp16", "int8", hoặc "int4"

    Returns: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = QUANT_CONFIGS[quant_method]()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        **config
    )
    model.eval()
    return model, tokenizer


def measure_vram() -> float:
    """Đo VRAM hiện tại (GB)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def measure_inference_speed(model, tokenizer, prompt: str = "The future of AI is", max_tokens: int = 100) -> Dict:
    """
    Đo tốc độ inference

    Returns: Dict với latency_ms và tokens_per_second
    """
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
    latency_ms = elapsed * 1000
    tokens_per_second = num_tokens / elapsed

    return {
        "latency_ms": latency_ms,
        "tokens_per_second": tokens_per_second
    }


def get_model_size_mb(model_name: str, quant_method: str) -> float:
    """
    Ước tính kích thước model sau quantization

    Returns: Size in MB
    """
    # Ước tính từ số parameters
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        num_params = sum(p.numel() for p in model.parameters())
        del model
        torch.cuda.empty_cache()
    except Exception:
        # Fallback: ước tính theo tên model
        return 0.0

    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }
    return (num_params * bytes_per_param.get(quant_method, 2)) / 1e6


def benchmark_quantization(model_name: str, quant_method: str) -> QuantizationMetrics:
    """
    Benchmark một quantization method
    """
    print(f"\n{'='*50}")
    print(f"Method: {quant_method.upper()}")
    print(f"{'='*50}")
    
    # 1. Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 2. Đo VRAM baseline
    vram_before = measure_vram()
    
    # 3. Load model
    start = time.time()
    model, tokenizer = load_model_with_quantization(model_name, quant_method)
    load_time = time.time() - start
    
    # 4. Đo VRAM sau load
    vram_after = measure_vram()
    vram_used = vram_after - vram_before
    
    # 5. Đo inference speed
    speed = measure_inference_speed(model, tokenizer)
    
    # 6. Lấy model size
    model_size = get_model_size_mb(model_name, quant_method)
    
    # 7. Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    metrics = QuantizationMetrics(
        method=quant_method,
        bits={"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}[quant_method],
        vram_gb=vram_used,
        model_size_mb=model_size,
        load_time_s=load_time,
        inference_latency_ms=speed.get("latency_ms", 0),
        tokens_per_second=speed.get("tokens_per_second", 0)
    )
    
    print(f"VRAM: {vram_used:.2f} GB")
    print(f"Model size: {model_size:.0f} MB")
    print(f"Speed: {speed.get('tokens_per_second', 0):.1f} tok/s")
    
    return metrics


def run_comparison(model_name: str = "facebook/opt-2.7b") -> List[QuantizationMetrics]:
    """So sánh tất cả quantization methods"""
    results = []
    
    for method in ["fp16", "int8", "int4"]:  # Skip fp32 để tiết kiệm VRAM
        try:
            metrics = benchmark_quantization(model_name, method)
            results.append(metrics)
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    return results


def save_results(results: List[QuantizationMetrics], output_path: str = "results/metrics.json"):
    """Lưu kết quả"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = {
        "lesson": "02-quantization",
        "model": "facebook/opt-2.7b",
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_comparison_table(results: List[QuantizationMetrics]):
    """In bảng so sánh"""
    if not results:
        return
    
    baseline = results[0]  # FP16
    
    print(f"\n{'='*70}")
    print("QUANTIZATION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<10} {'Bits':<6} {'VRAM (GB)':<12} {'Size (MB)':<12} {'Tok/s':<10} {'VRAM %':<10}")
    print("-"*70)
    
    for r in results:
        vram_pct = (r.vram_gb / baseline.vram_gb) * 100 if baseline.vram_gb > 0 else 0
        print(f"{r.method:<10} {r.bits:<6} {r.vram_gb:<12.2f} {r.model_size_mb:<12.0f} {r.tokens_per_second:<10.1f} {vram_pct:<10.1f}")


def main():
    print("="*60)
    print("Lesson 02: Quantization Techniques")
    print("="*60)
    
    MODEL = "facebook/opt-2.7b"
    
    results = run_comparison(MODEL)
    save_results(results)
    print_comparison_table(results)
    
    # Tính savings
    if len(results) >= 2:
        fp16 = results[0]
        int4 = results[-1]
        vram_savings = ((fp16.vram_gb - int4.vram_gb) / fp16.vram_gb) * 100
        size_savings = ((fp16.model_size_mb - int4.model_size_mb) / fp16.model_size_mb) * 100
        
        print(f"\n{'='*60}")
        print("KEY FINDINGS")
        print(f"{'='*60}")
        print(f"VRAM Savings (FP16 → INT4): {vram_savings:.1f}%")
        print(f"Model Size Savings: {size_savings:.1f}%")


if __name__ == "__main__":
    main()
