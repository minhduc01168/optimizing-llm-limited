# Lesson 05: Continuous Batching & Iteration-level Scheduling

## Mục tiêu

Hiểu và benchmark Continuous Batching so với Static Batching

## Acceptance Criteria

- [ ] AC-5.1: So sánh throughput static vs continuous batching
- [ ] AC-5.2: Đo batch utilization percentage
- [ ] AC-5.3: Phân tích queue wait time

## Hardware Requirement

- GPU: NVIDIA T4 (16GB VRAM)

## Static vs Continuous Batching

```
Static Batching:
┌─────────────────────────────────────┐
│ Batch 1: [Req1 ████][Req2 ██][Req3 ██████] │
│          ↑ Wait for longest to finish      │
│ Batch 2: [Req4 ███][Req5 █████]            │
└─────────────────────────────────────┘
→ GPU idle khi request ngắn hoàn thành sớm

Continuous Batching:
┌─────────────────────────────────────┐
│ Step 1: [Req1][Req2][Req3]                 │
│ Step 2: [Req1][Req2][Req4]  ← Req3 done, add Req4 │
│ Step 3: [Req1][Req5][Req4]  ← Req2 done, add Req5 │
└─────────────────────────────────────┘
→ GPU luôn busy, swap requests ngay khi done
```

## Metrics cần thu thập

| Metric | Static | Continuous | Improvement |
|--------|--------|------------|-------------|
| Throughput (tok/s) | TBD | TBD | Target: +50% |
| Batch utilization (%) | ~50% | >80% | +30% |
| Queue wait p95 (ms) | TBD | TBD | Target: -40% |
| OOM errors | TBD | TBD | Target: 0 |

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-05-continuous-batching
pip install torch transformers
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `run_static_batch()` - Chạy static batching (tokenize all prompts cùng lúc, generate batch, đếm tokens, tính utilization)
2. Hàm `run_continuous_batch()` - Chạy continuous batching (chia prompts thành chunks, generate từng chunk, simulate swap requests, tính utilization)
3. Hàm `generate_test_prompts()` - Tạo test prompts với độ dài khác nhau
4. Hàm `benchmark_batching()` - Benchmark so sánh static và continuous batching (load model, run both, cleanup)
5. Hàm `save_results()` - Lưu kết quả vào file JSON
6. Hàm `print_comparison()` - In bảng so sánh (throughput, utilization, improvement %)

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- Throughput (tok/s) cho static vs continuous
- Batch utilization (%)
- Queue wait time p95, p99 (ms)
- Improvement percentage
- OOM errors

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | Static | Continuous | Improvement | Mục tiêu |
|--------|--------|------------|-------------|----------|
| Throughput (tok/s) | ... | ... | ... | +50% |
| Batch utilization (%) | ... | ... | ... | > 80% |
| Queue wait p95 (ms) | ... | ... | ... | -40% |
| OOM errors | ... | ... | ... | 0 |

### Kiến thức đã nắm được
- Static Batching: tất cả requests trong batch phải đợi request chậm nhất hoàn thành
- Continuous Batching: swap requests ngay khi done, GPU luôn busy
- Iteration-level scheduling và lợi ích của nó
- Tại sao continuous batching giảm GPU idle time và tăng throughput

### Kỹ năng đã thành thạo
- Implement static batching với padding và batch generation
- Simulate continuous batching với chunk-based processing
- Đo batch utilization (GPU active time / total time)
- So sánh throughput và latency giữa hai approaches

### Dữ liệu cho báo cáo
```json
{
  "lesson": "05-continuous-batching",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "static_throughput_tok_s": "[giá trị]",
    "continuous_throughput_tok_s": "[giá trị]",
    "improvement_pct": "[giá trị]",
    "batch_utilization_pct": "[giá trị]"
  },
  "status": "completed"
}
```
