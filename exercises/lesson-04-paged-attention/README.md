# Lesson 04: PagedAttention & vLLM

## Mục tiêu

Hiểu PagedAttention và sử dụng vLLM để phục vụ LLM inference hiệu quả

## Acceptance Criteria

- [ ] AC-4.1: Khởi động vLLM server thành công
- [ ] AC-4.2: Benchmark throughput với nhiều requests
- [ ] AC-4.3: Đo KV cache efficiency

## Hardware Requirement

- GPU: NVIDIA T4 (16GB VRAM)

## PagedAttention là gì?

PagedAttention giải quyết vấn đề memory fragmentation trong KV cache:
- Traditional: Cấp phát contiguous memory cho mỗi request
- PagedAttention: Chia KV cache thành pages, share giữa requests

```
Traditional KV Cache:
[Request 1: ████████........]  ← Waste
[Request 2: ████............]  ← Waste

PagedAttention:
[Page 1][Page 2][Page 3][Page 4]  ← Shared pool
  ↓       ↓       ↓       ↓
Req1    Req2    Req1    Req3
```

## Metrics cần thu thập

| Metric | Mô tả | Mục tiêu T4 |
|--------|-------|-------------|
| Throughput (req/s) | Requests per second | > 5 |
| TTFT p50 (ms) | Time to first token | < 500ms |
| TTFT p99 (ms) | 99th percentile | < 1500ms |
| Cache hit rate (%) | KV cache reuse | > 70% |
| Max concurrent | Max parallel requests | > 50 |
| VRAM efficiency | Useful VRAM % | > 80% |

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-04-paged-attention
pip install vllm aiohttp torch transformers
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `start_vllm_server()` - Khởi động vLLM server với model, port, gpu_memory_utilization
2. Hàm `wait_for_server()` - Đợi server sẵn sàng (health check URL, timeout)
3. Hàm `stop_vllm_server()` - Dừng vLLM server
4. Hàm `send_request()` - Gửi một request đến vLLM server (async, streaming, đo TTFT)
5. Hàm `run_load_test()` - Chạy load test với nhiều concurrent requests (semaphore, percentiles)
6. Hàm `get_vllm_metrics()` - Lấy metrics từ vLLM server (cache hit rate, VRAM usage)
7. Hàm `benchmark_vllm()` - Chạy benchmark vLLM hoàn chỉnh (start → load test → metrics → compile)
8. Hàm `save_results()` - Lưu kết quả vào file JSON

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- Throughput (requests/sec)
- TTFT p50 và p99 (ms)
- Cache hit rate (%)
- VRAM usage (GB)
- Success rate (successful/total requests)

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | Giá trị đạt được | Mục tiêu T4 | Đạt? |
|--------|-----------------|-------------|------|
| Throughput (req/s) | ... | > 5 | ✅/❌ |
| TTFT p50 (ms) | ... | < 500ms | ✅/❌ |
| TTFT p99 (ms) | ... | < 1500ms | ✅/❌ |
| Cache hit rate (%) | ... | > 70% | ✅/❌ |
| Max concurrent | ... | > 50 | ✅/❌ |
| VRAM efficiency (%) | ... | > 80% | ✅/❌ |

### Kiến thức đã nắm được
- PagedAttention giải quyết memory fragmentation như thế nào
- Cơ chế page-based KV cache và sharing giữa requests
- Sự khác biệt giữa traditional contiguous memory và paged memory
- Cách vLLM tối ưu throughput với continuous batching và PagedAttention

### Kỹ năng đã thành thạo
- Khởi động và quản lý vLLM server
- Chạy load test với async HTTP requests
- Đo TTFT (Time To First Token) và latency percentiles
- Phân tích KV cache hit rate và VRAM efficiency

### Dữ liệu cho báo cáo
```json
{
  "lesson": "04-paged-attention",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "throughput_req_per_sec": "[giá trị]",
    "ttft_p50_ms": "[giá trị]",
    "ttft_p99_ms": "[giá trị]",
    "cache_hit_rate_pct": "[giá trị]",
    "vram_usage_gb": "[giá trị]"
  },
  "status": "completed"
}
```
