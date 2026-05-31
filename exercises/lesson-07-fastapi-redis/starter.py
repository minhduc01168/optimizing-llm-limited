"""
Lesson 07: FastAPI & Redis Microservices
==========================================
Xây dựng LLM API với caching

Hardware: NVIDIA T4 (16GB VRAM)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import time
import os
import hashlib
from typing import Optional
import uvicorn


app = FastAPI(title="LLM Serving API")

# ============================================================
# Redis connection
# ============================================================
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except Exception:
    redis_client = None
    print("Warning: Redis không khả dụng. Cache sẽ bị vô hiệu hóa.")


# ============================================================
# Request/Response Models
# ============================================================

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    model: str = "facebook/opt-2.7b"


class CompletionResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float
    cached: bool


class CacheStats(BaseModel):
    hits: int
    misses: int
    hit_rate: float
    keys: int


# Global model cache
_model = None
_tokenizer = None


def load_model(model_name: str):
    """Load LLM model"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    _model.eval()
    print("Model loaded!")
    return _model, _tokenizer


def generate_text(prompt: str, max_tokens: int, temperature: float) -> dict:
    """Generate text từ model"""
    import torch

    model, tokenizer = load_model("facebook/opt-2.7b")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    return {"text": text, "tokens": num_tokens}


# ============================================================
# Cache Functions
# ============================================================

def get_cache_key(request: CompletionRequest) -> str:
    """
    Tạo cache key từ request parameters
    """
    key_data = f"{request.prompt}:{request.max_tokens}:{request.temperature}:{request.model}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_from_cache(key: str) -> Optional[dict]:
    """
    Lấy kết quả từ Redis cache
    Returns: Cached result hoặc None
    """
    if redis_client is None:
        return None
    try:
        data = redis_client.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"Cache read error: {e}")
    return None


def set_to_cache(key: str, value: dict, ttl: int = 3600):
    """
    Lưu kết quả vào Redis cache
    Args:
        key: Cache key
        value: Result to cache
        ttl: Time to live in seconds
    """
    if redis_client is None:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        print(f"Cache write error: {e}")


# ============================================================
# API Endpoints
# ============================================================

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Generate text completion với caching
    """
    start_time = time.time()
    
    # 1. Check cache
    cache_key = get_cache_key(request)
    cached_result = get_from_cache(cache_key)
    
    if cached_result:
        # Cache hit
        latency = (time.time() - start_time) * 1000
        return CompletionResponse(
            text=cached_result["text"],
            tokens_generated=cached_result["tokens"],
            latency_ms=latency,
            cached=True
        )
    
    # 2. Cache miss - generate
    try:
        result = generate_text(request.prompt, request.max_tokens, request.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # 3. Save to cache
    set_to_cache(cache_key, result)
    
    # 4. Return response
    latency = (time.time() - start_time) * 1000
    return CompletionResponse(
        text=result["text"],
        tokens_generated=result["tokens"],
        latency_ms=latency,
        cached=False
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Lấy cache statistics"""
    if redis_client is None:
        return CacheStats(hits=0, misses=0, hit_rate=0.0, keys=0)

    try:
        info = redis_client.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0

        # Đếm số keys
        dbsize = redis_client.dbsize()

        return CacheStats(
            hits=hits,
            misses=misses,
            hit_rate=round(hit_rate, 2),
            keys=dbsize
        )
    except Exception as e:
        print(f"Cache stats error: {e}")
        return CacheStats(hits=0, misses=0, hit_rate=0.0, keys=0)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        import torch

        # GPU metrics
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            gpu_memory_total = torch.cuda.get_device_properties(0).total_mem / 1e9

        # Cache metrics
        cache_info = {}
        if redis_client:
            try:
                info = redis_client.info("stats")
                cache_info = {
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                }
            except Exception:
                pass

        # Format as Prometheus metrics
        lines = [
            "# HELP llm_gpu_memory_used_gb GPU memory used in GB",
            "# TYPE llm_gpu_memory_used_gb gauge",
            f"llm_gpu_memory_used_gb {gpu_memory_used}",
            "# HELP llm_gpu_memory_total_gb GPU memory total in GB",
            "# TYPE llm_gpu_memory_total_gb gauge",
            f"llm_gpu_memory_total_gb {gpu_memory_total}",
            "# HELP llm_cache_hits Total cache hits",
            "# TYPE llm_cache_hits counter",
            f"llm_cache_hits {cache_info.get('hits', 0)}",
            "# HELP llm_cache_misses Total cache misses",
            "# TYPE llm_cache_misses counter",
            f"llm_cache_misses {cache_info.get('misses', 0)}",
        ]

        return {"content_type": "text/plain", "body": "\n".join(lines)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Main
# ============================================================

def save_results(output_path: str = "results/metrics.json"):
    """Lưu kết quả test"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "lesson": "07-fastapi-redis",
        "endpoints": ["/v1/completions", "/health", "/cache/stats", "/metrics"],
        "features": ["redis_caching", "prometheus_metrics", "gpu_monitoring"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


def print_summary():
    """In tóm tắt service"""
    print(f"\n{'='*60}")
    print("LLM SERVING API SUMMARY")
    print(f"{'='*60}")
    print("Endpoints:")
    print("  POST /v1/completions  - Text generation with caching")
    print("  GET  /health          - Health check")
    print("  GET  /cache/stats     - Redis cache statistics")
    print("  GET  /metrics         - Prometheus metrics")
    print(f"Redis: {'connected' if redis_client else 'disabled'}")
    print(f"URL: http://localhost:8000")


def main():
    print("="*60)
    print("Lesson 07: FastAPI & Redis Microservices")
    print("="*60)

    save_results()
    print_summary()

    print("\nStarting server on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
