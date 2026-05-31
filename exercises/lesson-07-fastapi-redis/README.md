# Lesson 07: FastAPI & Redis Microservices

## Mục tiêu

Xây dựng LLM serving API với FastAPI và Redis caching

## Acceptance Criteria

- [ ] AC-7.1: Tạo FastAPI endpoint phục vụ LLM inference
- [ ] AC-7.2: Implement Redis caching cho responses
- [ ] AC-7.3: Đo cache hit rate và latency improvement

## Architecture

```
Client Request
     │
     ▼
┌─────────────┐
│  FastAPI     │ ← Rate limiting, validation
│  Gateway     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Redis     │ ←──→ │  LLM Model  │
│   Cache     │     │  (vLLM)     │
└─────────────┘     └─────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Generate text |
| `/v1/chat` | POST | Chat completion |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/cache/stats` | GET | Cache statistics |

## Metrics cần thu thập

| Metric | Without Cache | With Cache | Improvement |
|--------|---------------|------------|-------------|
| Latency p50 (ms) | TBD | TBD | Target: -50% |
| Throughput (req/s) | TBD | TBD | Target: +100% |
| Cache hit rate | 0% | >40% | - |
| Error rate (%) | TBD | TBD | Target: <1% |

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-07-fastapi-redis
pip install fastapi uvicorn redis torch transformers pydantic
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `load_model()` - Load LLM model (global cache, AutoTokenizer, AutoModelForCausalLM)
2. Hàm `generate_text()` - Generate text từ model (tokenize, generate, decode)
3. Hàm `get_cache_key()` - Tạo cache key từ request parameters (MD5 hash)
4. Hàm `get_from_cache()` - Lấy kết quả từ Redis cache (redis_client.get, json.loads)
5. Hàm `set_to_cache()` - Lưu kết quả vào Redis cache (redis_client.setex, TTL)
6. Endpoint `POST /v1/completions` - Generate text completion với caching (check cache → generate → save cache)
7. Endpoint `GET /health` - Health check endpoint
8. Endpoint `GET /cache/stats` - Lấy cache statistics (hits, misses, hit_rate, keys)
9. Endpoint `GET /metrics` - Prometheus metrics endpoint (GPU memory, cache stats)

### Bước 4: Chạy bài thực hành
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start FastAPI server
python starter.py
```

### Bước 5: Kiểm tra kết quả
Test API bằng curl:
```bash
# Health check
curl http://localhost:8000/health

# Generate text (cache miss)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'

# Generate text again (cache hit - nhanh hơn nhiều)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'

# Cache stats
curl http://localhost:8000/cache/stats

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | Without Cache | With Cache | Improvement | Đạt? |
|--------|---------------|------------|-------------|------|
| Latency p50 (ms) | ... | ... | ... | ✅/❌ |
| Throughput (req/s) | ... | ... | ... | ✅/❌ |
| Cache hit rate | 0% | ... | - | ✅/❌ |
| Error rate (%) | ... | ... | ... | ✅/❌ |

### Kiến thức đã nắm được
- Cách xây dựng LLM serving API với FastAPI
- Redis caching strategy cho LLM responses (key generation, TTL, eviction)
- Rate limiting và input validation với Pydantic
- Prometheus metrics format cho monitoring

### Kỹ năng đã thành thạo
- Tạo FastAPI endpoints cho LLM inference
- Implement Redis caching với cache key generation (MD5 hash)
- Đo cache hit rate và latency improvement
- Export Prometheus metrics cho monitoring

### Dữ liệu cho báo cáo
```json
{
  "lesson": "07-fastapi-redis",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "latency_without_cache_ms": "[giá trị]",
    "latency_with_cache_ms": "[giá trị]",
    "cache_hit_rate_pct": "[giá trị]",
    "throughput_improvement_pct": "[giá trị]"
  },
  "status": "completed"
}
```
