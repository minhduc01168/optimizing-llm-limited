# Báo cáo Kỹ thuật: Khóa học LLM Engineering

**Giảng viên:** Vũ Minh Khang
**Đối tượng:** Học sinh THPT
**Hardware:** NVIDIA T4 (16GB VRAM)
**Thời lượng:** 8 buổi thực hành

---

## Tóm tắt Khóa học

Khóa học LLM Engineering cung cấp kiến thức thực tế về Large Language Model (LLM) inference optimization và production deployment. Học sinh được thực hành trên GPU T4 (16GB VRAM) với các bài tập từ cơ bản đến nâng cao.

### Kết quả đạt được sau khóa học:
- Hiểu kiến trúc Transformer và cách quản lý GPU VRAM
- Thành thạo các kỹ thuật lượng tử hóa (GPTQ, AWQ, GGUF)
- Biết cách sử dụng vLLM với PagedAttention cho hiệu quả cao
- Xây dựng được LLM serving API với FastAPI và Redis
- Deploy ứng dụng lên Docker với GPU support

---

## Chương trình học

### Bài 1: Kiến trúc Transformer & VRAM Analysis

**Mục tiêu:** Hiểu cấu trúc Transformer và cách GPU VRAM được sử dụng

**Nội dung thực hành:**
- Load các model khác nhau (distilgpt2 → opt-1.3b)
- Đo VRAM usage với pynvml
- Benchmark inference latency

**Kết quả mẫu:**

| Model | Parameters | VRAM (GB) | Latency (ms/token) |
|-------|------------|-----------|-------------------|
| distilgpt2 | 82M | 0.5 | 15 |
| gpt2 | 124M | 1.0 | 18 |
| gpt2-medium | 355M | 2.5 | 25 |
| gpt2-large | 774M | 5.0 | 35 |
| opt-1.3b | 1.3B | 8.0 | 50 |

**Kiến thức rút ra:**
- VRAM usage tỉ lệ tuyến tính với số parameters (FP16)
- Mỗi parameter chiếm ~2 bytes trong FP16
- T4 (16GB) có thể chạy models lên đến ~8B parameters

---

### Bài 2: Kỹ thuật Lượng tử hóa (Quantization)

**Mục tiêu:** So sánh FP32, FP16, INT8, INT4 quantization

**Nội dung thực hành:**
- Sử dụng bitsandbytes cho INT8/INT4 quantization
- So sánh VRAM, model size, inference speed
- Đo perplexity delta

**Kết quả mẫu (Model: opt-2.7b):**

| Method | Bits | VRAM (GB) | Size (GB) | Tok/s | VRAM Reduction |
|--------|------|-----------|-----------|-------|----------------|
| FP16 | 16 | 5.4 | 5.4 | 45 | Baseline |
| INT8 | 8 | 3.2 | 3.2 | 38 | -41% |
| INT4-NF4 | 4 | 2.1 | 2.1 | 35 | -61% |

**Kiến thức rút ra:**
- INT4 quantization giảm 61% VRAM với chỉ ~22% speed loss
- NF4 (Normal Float 4) cho kết quả tốt hơn standard INT4
- Perplexity increase < 5% khi quantize từ FP16 xuống INT4

---

### Bài 3: Thực hành Lượng tử hóa Llama-3-8B

**Mục tiêu:** Quantize Llama-3-8B với GPTQ, AWQ, GGUF

**Nội dung thực hành:**
- GPTQ: Sử dụng auto-gptq với calibration dataset
- AWQ: Sử dụng autoawq cho activation-aware quantization
- GGUF: Convert sang format cho llama.cpp

**Kết quả mẫu:**

| Method | Original | Quantized | Compression | Time (min) |
|--------|----------|-----------|-------------|------------|
| GPTQ-4bit | 16GB | 4GB | 4x | 15 |
| AWQ-4bit | 16GB | 4GB | 4x | 10 |
| GGUF-Q4_K_M | 16GB | 5GB | 3.2x | 5 |

**Kiến thức rút ra:**
- AWQ nhanh hơn GPTQ và cho kết quả tương đương
- GGUF tối ưu cho CPU/GPU hybrid inference
- Chọn method phụ thuộc vào serving framework

---

### Bài 4: PagedAttention & vLLM

**Mục tiêu:** Hiểu PagedAttention và benchmark vLLM serving

**Nội dung thực hành:**
- Khởi động vLLM server với model đã quantize
- Load test với 100 concurrent requests
- Thu thập throughput, latency, cache metrics

**Kết quả mẫu:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Throughput | 25 req/s | >5 | ✅ |
| TTFT p50 | 150ms | <500ms | ✅ |
| TTFT p99 | 800ms | <1500ms | ✅ |
| Cache hit rate | 85% | >70% | ✅ |
| Max concurrent | 100 | >50 | ✅ |

**Kiến thức rút ra:**
- PagedAttention tăng throughput 3-5x so với naive serving
- KV cache sharing giúp phục vụ nhiều hơn 2x requests
- T4 đủ mạnh cho production với <100 concurrent users

---

### Bài 5: Continuous Batching & Scheduling

**Mục tiêu:** So sánh Static Batching vs Continuous Batching

**Nội dung thực hành:**
- Implement static và continuous batching
- Benchmark với 50 requests, độ dài khác nhau
- Đo throughput và utilization

**Kết quả mẫu:**

| Metric | Static | Continuous | Improvement |
|--------|--------|------------|-------------|
| Throughput (tok/s) | 120 | 200 | +67% |
| Batch utilization | 50% | 85% | +35% |
| Queue wait p95 | 3000ms | 1200ms | -60% |

**Kiến thức rút ra:**
- Continuous batching tối ưu GPU utilization
- Requests ngắn benefit nhiều nhất
- Kết hợp với PagedAttention cho hiệu quả tối đa

---

### Bài 6: So sánh Serving Frameworks

**Mục tiêu:** So sánh vLLM, TGI, llama.cpp, FastAPI

**Kết quả mẫu:**

| Framework | Req/s | Latency p50 | VRAM | Cold Start |
|-----------|-------|-------------|------|------------|
| vLLM | 25 | 150ms | 8GB | 30s |
| TGI | 20 | 180ms | 8GB | 45s |
| llama.cpp | 15 | 200ms | 6GB | 10s |
| FastAPI | 10 | 250ms | 8GB | 20s |

**Kiến thức rút ra:**
- vLLM tốt nhất cho throughput-intensive workloads
- llama.cpp tốt cho resource-constrained environments
- FastAPI linh hoạt nhất cho custom logic

---

### Bài 7: FastAPI & Redis Microservices

**Mục tiêu:** Xây dựng LLM API với caching

**Nội dung thực hành:**
- Tạo FastAPI endpoints cho LLM inference
- Implement Redis caching
- Đo cache hit rate và latency improvement

**Kết quả mẫu:**

| Metric | Without Cache | With Cache | Improvement |
|--------|---------------|------------|-------------|
| Latency p50 | 200ms | 50ms | -75% |
| Throughput | 10 req/s | 40 req/s | +300% |
| Cache hit rate | 0% | 70% | - |

**Kiến thức rút ra:**
- Redis caching giảm latency 4x cho repeated queries
- Cache key design quan trọng (prompt + params hash)
- TTL strategy cần tune theo use case

---

### Bài 8: Docker & GPU Production Deployment

**Mục tiêu:** Containerize và deploy LLM service

**Nội dung thực hành:**
- Tạo Dockerfile với GPU support
- Docker Compose với multi-service
- Health check và monitoring

**Kết quả mẫu:**

| Metric | Value | Target |
|--------|-------|--------|
| Image size | 6.5GB | <8GB |
| Build time | 180s | <300s |
| Container start | 45s | <60s |
| GPU overhead | 3% | <5% |
| Health check | 50ms | <100ms |

**Kiến thức rút ra:**
- Multi-stage builds giảm image size 40%
- NVIDIA Container Toolkit cần cài đặt chính xác
- Health checks quan trọng cho production

---

## Kỹ năng đạt được

### Kỹ năng kỹ thuật
- **LLM Inference:** Transformer architecture, attention mechanisms, KV cache
- **Quantization:** GPTQ, AWQ, GGUF, bitsandbytes
- **Serving:** vLLM, PagedAttention, continuous batching
- **API Development:** FastAPI, Redis caching, async programming
- **DevOps:** Docker, GPU containers, monitoring

### Công cụ sử dụng
```
PyTorch | Transformers | vLLM | bitsandbytes | auto-gptq | autoawq
FastAPI | Redis | Docker | NVIDIA Container Toolkit
pynvml | prometheus-client | locust
```

---

## Tóm tắt cho CV

```
LLM Engineering Project - GPU T4 (16GB VRAM)
──────────────────────────────────────────────

THÀNH TÍCH:
• Tăng throughput inference lên 4x (50→200+ tok/s) thông qua 
  PagedAttention và continuous batching

• Giảm footprint bộ nhớ model 70% bằng INT4 quantization 
  (GPTQ/AWQ) với accuracy loss <5%

• Xây dựng LLM serving infrastructure với FastAPI và Redis 
  caching, đạt 70% cache hit rate và giảm latency 4x

• Deploy containerized LLM service với Docker, 
  cold start <60s và GPU overhead <5%

TECH STACK:
PyTorch | vLLM | FastAPI | Redis | Docker | NVIDIA T4
```

---

## Cách sử dụng báo cáo này

### Cho học sinh:
1. Điền tên của bạn vào phần header
2. Chỉnh sửa số liệu dựa trên kết quả thực tế của bạn
3. Copy phần "Tóm tắt cho CV" vào CV/portfolio của bạn
4. Đính kèm link GitHub repo chứa source code

### Cho giảng viên:
1. Sử dụng làm template cho các khóa học sau
2. Cập nhật số liệu dựa trên hardware thực tế
3. Điều chỉnh difficulty theo trình độ học sinh

---

## Liên hệ

**Giảng viên:** Vũ Minh Khang
**Email:** [Email]
**GitHub:** [GitHub Link]

---

*Báo cáo này được tạo tự động bởi BMAD Technical Report Generator*
