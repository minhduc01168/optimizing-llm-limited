# Lesson 03: Hands-on Quantization với Llama-3-8B

## Mục tiêu

Thực hành lượng tử hóa mô hình Llama-3-8B với GPTQ, AWQ, và GGUF

## Acceptance Criteria

- [ ] AC-3.1: Quantize model với GPTQ thành công
- [ ] AC-3.2: Quantize model với AWQ thành công
- [ ] AC-3.3: Convert sang GGUF format
- [ ] AC-3.4: So sánh inference speed giữa các methods

## Hardware Requirement

- GPU: NVIDIA T4 (16GB VRAM)
- RAM: 16GB system RAM
- Storage: 20GB free space

## Lưu ý quan trọng

Llama-3-8B FP16 cần ~16GB VRAM (vừa khít T4). Khi quantize:
- INT4: ~4GB VRAM → có thể batch inference
- INT8: ~8GB VRAM → comfortable inference

## Quantization Pipeline

```
Original Model (FP16, 16GB)
    ├── GPTQ (4-bit, ~4GB)
    │   └── Inference với vLLM
    ├── AWQ (4-bit, ~4GB)
    │   └── Inference với vLLM
    └── GGUF (4-bit, ~5GB)
        └── Inference với llama.cpp
```

## Metrics cần thu thập

| Metric | GPTQ | AWQ | GGUF |
|--------|------|-----|------|
| VRAM usage (GB) | ~4 | ~4 | ~5 |
| Model size (GB) | ~4 | ~4 | ~5 |
| Quantization time (min) | ~15 | ~10 | ~5 |
| Tokens/sec | TBD | TBD | TBD |
| Perplexity | TBD | TBD | TBD |

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-03-hands-on-quant
pip install torch transformers auto-gptq autoawq llama-cpp-python
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `quantize_gptq()` - Quantize model với GPTQ (load model, calibration data, quantize, save, benchmark)
2. Hàm `quantize_awq()` - Quantize model với AWQ (load model, calibration data, quantize, save, benchmark)
3. Hàm `convert_to_gguf()` - Convert model sang GGUF format (sử dụng llama.cpp convert script)
4. Hàm `benchmark_inference()` - Benchmark inference speed cho GPTQ, AWQ, GGUF
5. Hàm `get_directory_size()` - Tính kích thước thư mục (GB)
6. Hàm `measure_vram()` - Đo VRAM hiện tại (GB)
7. Hàm `run_quantization_pipeline()` - Chạy toàn bộ pipeline quantization (GPTQ → AWQ → GGUF)
8. Hàm `save_results()` - Lưu kết quả vào file JSON

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- So sánh size giữa GPTQ, AWQ, GGUF
- VRAM usage cho mỗi method
- Tokens per second và latency
- Bảng QUANTIZATION COMPARISON in ra terminal

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | GPTQ | AWQ | GGUF | Mục tiêu |
|--------|------|-----|------|----------|
| VRAM usage (GB) | ... | ... | ... | ~4-5GB |
| Model size (GB) | ... | ... | ... | ~4-5GB |
| Quantization time (min) | ... | ... | ... | < 20 min |
| Tokens/sec | ... | ... | ... | TBD |

### Kiến thức đã nắm được
- Sự khác biệt giữa GPTQ (post-training, layer-wise), AWQ (activation-aware), GGUF (llama.cpp format)
- Calibration data và vai trò của nó trong quantization
- Compression ratio và tác động đến model quality
- Khi nào nên dùng GPTQ vs AWQ vs GGUF

### Kỹ năng đã thành thạo
- Quantize model với GPTQ sử dụng auto-gptq
- Quantize model với AWQ sử dụng autoawq
- Convert model sang GGUF format cho llama.cpp
- Benchmark inference speed trên các quantized models

### Dữ liệu cho báo cáo
```json
{
  "lesson": "03-hands-on-quant",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "gptq_size_gb": "[giá trị]",
    "awq_size_gb": "[giá trị]",
    "gguf_size_gb": "[giá trị]",
    "gptq_tokens_per_sec": "[giá trị]",
    "awq_tokens_per_sec": "[giá trị]"
  },
  "status": "completed"
}
```
