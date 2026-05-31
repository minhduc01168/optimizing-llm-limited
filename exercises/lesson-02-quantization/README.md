# Lesson 02: Quantization Techniques

## Mục tiêu

Hiểu và thực hành các kỹ thuật lượng tử hóa: GPTQ, AWQ, GGUF

## Acceptance Criteria

- [ ] AC-2.1: So sánh VRAM usage giữa FP32, FP16, INT8, INT4
- [ ] AC-2.2: Tính phần trăm giảm VRAM khi quantize
- [ ] AC-2.3: Đo perplexity delta giữa các quantization methods

## Hardware Requirement

- GPU: NVIDIA T4 (16GB VRAM)
- RAM: 16GB system RAM

## Quantization Methods

| Method | Bits | Library | T4 Support |
|--------|------|---------|------------|
| FP32 | 32 | PyTorch | ✅ |
| FP16 | 16 | PyTorch | ✅ |
| INT8 | 8 | bitsandbytes | ✅ |
| INT4-NF4 | 4 | bitsandbytes | ✅ |
| GPTQ | 4 | auto-gptq | ✅ |
| AWQ | 4 | autoawq | ✅ |

## Metrics cần thu thập

| Metric | Công cụ | Mục tiêu |
|--------|---------|----------|
| VRAM usage (GB) | pynvml | Giảm ≥50% |
| Model size (MB) | file size | Giảm ≥60% |
| Perplexity delta | lm-eval | < 10% increase |
| Inference speed (tok/s) | custom timer | ≥ 80% FP16 |
| Quantization time (min) | time | < 20 min |

## Models phù hợp T4

- `facebook/opt-2.7b` - FP16: ~5.4GB, INT4: ~2.1GB
- `meta-llama/Llama-2-7b-hf` - FP16: ~14GB, INT4: ~4GB (tight fit)

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-02-quantization
pip install torch transformers bitsandbytes accelerate
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `get_fp32_config()` - Cấu hình FP32 baseline (torch.float32)
2. Hàm `get_fp16_config()` - Cấu hình FP16 half precision (torch.float16)
3. Hàm `get_int8_config()` - Cấu hình INT8 với BitsAndBytesConfig (load_in_8bit=True)
4. Hàm `get_int4_config()` - Cấu hình INT4-NF4 với BitsAndBytesConfig (load_in_4bit, nf4, double quant)
5. Hàm `load_model_with_quantization()` - Load model với quantization method chỉ định
6. Hàm `measure_vram()` - Đo VRAM hiện tại (GB)
7. Hàm `measure_inference_speed()` - Đo tốc độ inference (latency_ms, tokens_per_second)
8. Hàm `get_model_size_mb()` - Ước tính kích thước model sau quantization
9. Hàm `benchmark_quantization()` - Benchmark một quantization method hoàn chỉnh
10. Hàm `run_comparison()` - So sánh tất cả quantization methods (fp16, int8, int4)
11. Hàm `save_results()` - Lưu kết quả vào file JSON
12. Hàm `print_comparison_table()` - In bảng so sánh

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- VRAM usage cho FP16, INT8, INT4
- Model size savings (phần trăm giảm)
- Tokens per second cho mỗi method
- Bảng QUANTIZATION COMPARISON in ra terminal
- KEY FINDINGS: VRAM Savings và Model Size Savings

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | Giá trị đạt được | Mục tiêu | Đạt? |
|--------|-----------------|----------|------|
| VRAM reduction (%) | ... | ≥ 50% | ✅/❌ |
| Model size reduction (%) | ... | ≥ 60% | ✅/❌ |
| Perplexity delta (%) | ... | < 10% increase | ✅/❌ |
| Inference speed vs FP16 (%) | ... | ≥ 80% FP16 | ✅/❌ |

### Kiến thức đã nắm được
- Sự khác biệt giữa FP32, FP16, INT8, INT4 trong terms of precision và memory
- Cách BitsAndBytesConfig hoạt động cho quantization
- Trade-off giữa model size, VRAM usage, và inference speed
- Kỹ thuật NF4 (Normal Float 4-bit) và double quantization

### Kỹ năng đã thành thạo
- Cấu hình quantization với BitsAndBytesConfig
- Đo VRAM usage trước và sau quantization
- Benchmark inference speed với nhiều quantization methods
- Tính phần trăm tiết kiệm VRAM và model size

### Dữ liệu cho báo cáo
```json
{
  "lesson": "02-quantization",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "vram_reduction_pct": "[giá trị]",
    "size_reduction_pct": "[giá trị]",
    "fp16_vram_gb": "[giá trị]",
    "int4_vram_gb": "[giá trị]"
  },
  "status": "completed"
}
```
