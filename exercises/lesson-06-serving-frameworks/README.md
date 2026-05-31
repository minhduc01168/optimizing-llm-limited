# Lesson 06: Serving Frameworks Comparison

## Mục tiêu

So sánh các LLM serving frameworks: vLLM, TGI, llama.cpp, FastAPI

## Acceptance Criteria

- [ ] AC-6.1: Benchmark ít nhất 3 frameworks
- [ ] AC-6.2: Đo requests/sec và latency
- [ ] AC-6.3: Tạo bảng so sánh comprehensive

## Hardware Requirement

- GPU: NVIDIA T4 (16GB VRAM)

## Frameworks cần so sánh

| Framework | Backend | T4 Support | Highlights |
|-----------|---------|------------|------------|
| vLLM | PyTorch | ✅ | PagedAttention, continuous batching |
| TGI | Rust/Candle | ✅ | Production-ready, streaming |
| llama.cpp | C++ | ✅ | CPU/GPU hybrid, GGUF |
| FastAPI + Transformers | PyTorch | ✅ | Simple, customizable |

## Metrics cần thu thập

| Metric | vLLM | TGI | llama.cpp | FastAPI |
|--------|------|-----|-----------|---------|
| Requests/sec | TBD | TBD | TBD | TBD |
| Latency p50 (ms) | TBD | TBD | TBD | TBD |
| Latency p99 (ms) | TBD | TBD | TBD | TBD |
| VRAM usage (GB) | TBD | TBD | TBD | TBD |
| Cold start (s) | TBD | TBD | TBD | TBD |
| Memory efficiency | TBD | TBD | TBD | TBD |

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-06-serving-frameworks
pip install vllm aiohttp torch transformers fastapi uvicorn
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `start_vllm()` - Khởi động vLLM server (process, base_url)
2. Hàm `start_tgi()` - Khởi động TGI server qua Docker (process, base_url)
3. Hàm `start_llama_cpp()` - Khởi động llama.cpp server (process, base_url)
4. Hàm `start_fastapi()` - Khởi động FastAPI server (tạo temp app file, process, base_url)
5. Hàm `benchmark_endpoint()` - Benchmark một endpoint (async, semaphore, latency percentiles, success rate)
6. Hàm `measure_cold_start()` - Đo thời gian cold start (start process, đợi health check)
7. Hàm `cleanup_process()` - Dừng process (terminate, kill nếu cần)
8. Hàm `run_comparison()` - Chạy benchmark cho tất cả frameworks
9. Hàm `save_results()` - Lưu kết quả vào file JSON
10. Hàm `print_comparison_table()` - In bảng so sánh (req/s, latency, TTFT, VRAM, cold start)

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- Requests/sec cho mỗi framework
- Latency p50, p99 (ms)
- TTFT p50 (ms)
- VRAM usage (GB)
- Cold start time (s)
- Best Throughput và Best Latency winners

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | vLLM | TGI | llama.cpp | FastAPI |
|--------|------|-----|-----------|---------|
| Requests/sec | ... | ... | ... | ... |
| Latency p50 (ms) | ... | ... | ... | ... |
| Latency p99 (ms) | ... | ... | ... | ... |
| VRAM usage (GB) | ... | ... | ... | ... |
| Cold start (s) | ... | ... | ... | ... |

### Kiến thức đã nắm được
- Điểm mạnh và yếu của mỗi serving framework (vLLM, TGI, llama.cpp, FastAPI)
- Khi nào nên dùng framework nào (throughput vs latency vs simplicity)
- PagedAttention trong vLLM vs token streaming trong TGI
- Cold start time và tác động đến production deployment

### Kỹ năng đã thành thạo
- Khởi động và benchmark nhiều LLM serving frameworks
- Đo throughput, latency percentiles, TTFT
- So sánh frameworks theo nhiều metrics khác nhau
- Đánh giá trade-off giữa performance và ease of use

### Dữ liệu cho báo cáo
```json
{
  "lesson": "06-serving-frameworks",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "vllm_req_per_sec": "[giá trị]",
    "tgi_req_per_sec": "[giá trị]",
    "llama_cpp_req_per_sec": "[giá trị]",
    "fastapi_req_per_sec": "[giá trị]",
    "best_framework": "[tên framework]"
  },
  "status": "completed"
}
```
