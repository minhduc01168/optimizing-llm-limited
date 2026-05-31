# Lesson 01: Transformer Architecture & VRAM Analysis

## Mục tiêu

Hiểu cấu trúc Transformer và cách GPU VRAM được sử dụng trong quá trình inference.

## Acceptance Criteria

- [ ] AC-1.1: Load model và đo VRAM usage chính xác
- [ ] AC-1.2: Tạo bảng so sánh VRAM cho các model khác nhau
- [ ] AC-1.3: Visualize memory allocation theo layer

## Hardware Requirement

- GPU: NVIDIA T4 (16GB VRAM)
- RAM: 8GB system RAM
- Storage: 10GB free space

## Thời gian ước tính

- Phần lý thuyết: 30 phút
- Phần thực hành: 60 phút
- Phần benchmark: 30 phút

## Cài đặt

```bash
pip install torch transformers pynvml matplotlib
```

## Hướng dẫn

### Phần 1: Load Model và Measure VRAM

1. Mở `starter.py` và hoàn thành các hàm chưa implement
2. Chạy script để load model và ghi nhận VRAM:
   ```bash
   python starter.py
   ```

### Phần 2: Benchmark nhiều models

`starter.py` đã tích hợp sẵn benchmark cho tất cả models trong hàm `run_all_benchmarks()`. Khi chạy `python starter.py`, script sẽ tự động:
1. Load và benchmark từng model (distilgpt2, gpt2, gpt2-medium, gpt2-large, facebook/opt-1.3b)
2. Đo VRAM, latency, model size cho mỗi model
3. Lưu kết quả vào `results/metrics.json`
4. In bảng SUMMARY ra terminal

## Metrics cần thu thập

| Metric | Công cụ | Mục tiêu |
|--------|---------|----------|
| Peak VRAM (GB) | pynvml | < 14GB |
| Load time (s) | time.time() | < 30s |
| Inference latency (ms/token) | custom timer | < 100ms |
| Model size (MB) | os.path.getsize | Chính xác |

## Models được sử dụng

| Model | Parameters | Expected VRAM |
|-------|------------|---------------|
| distilgpt2 | 82M | ~0.5GB |
| gpt2 | 124M | ~1.0GB |
| gpt2-medium | 355M | ~2.5GB |
| gpt2-large | 774M | ~5.0GB |
| facebook/opt-1.3b | 1.3B | ~8.0GB |

## Files trong thư mục

- `starter.py` - Code skeleton chứa toàn bộ logic benchmark (load model, đo VRAM, inference latency, lưu kết quả)
- `solution.py` - Reference implementation
- `results/` - Thư mục chứa kết quả (tự động tạo khi chạy `starter.py`)

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-01-transformer-vram
pip install torch transformers pynvml matplotlib
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `get_gpu_info()` - Lấy thông tin GPU (tên, CUDA version, tổng VRAM)
2. Hàm `get_vram_usage()` - Đo VRAM hiện tại đang sử dụng (GB)
3. Hàm `load_model()` - Load model và tokenizer từ HuggingFace
4. Hàm `measure_inference_latency()` - Đo thời gian inference (ms/token)
5. Hàm `get_model_size()` - Lấy kích thước model trên disk (MB)
6. Hàm `count_parameters()` - Đếm số parameters của model
7. Hàm `get_vram_peak()` - Đo VRAM peak usage (GB)
8. Hàm `benchmark_model()` - Benchmark một model hoàn chỉnh
9. Hàm `run_all_benchmarks()` - Chạy benchmark cho tất cả models
10. Hàm `save_results()` - Lưu kết quả benchmark vào file JSON

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- VRAM usage cho từng model
- Inference latency (ms/token)
- Bảng so sánh SUMMARY in ra terminal

## Submission

Sau khi hoàn thành, nộp:
1. File `starter.py` đã hoàn thành
2. File `results/metrics.json` chứa kết quả benchmark

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | Giá trị đạt được | Mục tiêu | Đạt? |
|--------|-----------------|----------|------|
| Peak VRAM (GB) | ... | < 14GB | ✅/❌ |
| Load time (s) | ... | < 30s | ✅/❌ |
| Inference latency (ms/token) | ... | < 100ms | ✅/❌ |
| Model size (MB) | ... | Chính xác | ✅/❌ |

### Kiến thức đã nắm được
- Cấu trúc Transformer và cách attention mechanism hoạt động
- Cách GPU VRAM được phân bổ khi load model (weights, KV cache, activations)
- Sự khác biệt giữa các kích thước model và tác động đến VRAM
- Cách sử dụng pynvml để đo VRAM chính xác

### Kỹ năng đã thành thạo
- Đo và phân tích VRAM usage với PyTorch và pynvml
- Benchmark inference latency cho các model khác nhau
- Tạo bảng so sánh metrics cho nhiều models
- Visualize memory allocation theo layer

### Dữ liệu cho báo cáo
```json
{
  "lesson": "01-transformer-vram",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "peak_vram_gb": "[giá trị]",
    "load_time_s": "[giá trị]",
    "inference_latency_ms": "[giá trị]",
    "model_size_mb": "[giá trị]"
  },
  "status": "completed"
}
```
