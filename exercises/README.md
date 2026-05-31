# LLM Engineering Lab - Khóa học Thực hành LLM Production

## Tổng quan

Khóa học bao gồm 8 bài thực hành về LLM Engineering, tập trung vào inference optimization và deployment trên GPU T4 (16GB VRAM).

**Hardware Requirement:** NVIDIA T4 GPU (16GB VRAM) - Colab Free Tier hoặc cloud instance

## Cấu trúc khóa học

```
Phase 1: Foundation (Bài 1-3)
├── Lesson 01: Transformer Architecture & VRAM Analysis
├── Lesson 02: Quantization Techniques (GPTQ, AWQ, GGUF)
└── Lesson 03: Hands-on Quantization với Llama-3-8B

Phase 2: Optimization (Bài 4-5)
├── Lesson 04: PagedAttention & vLLM
└── Lesson 05: Continuous Batching & Scheduling

Phase 3: Production (Bài 6-8)
├── Lesson 06: Serving Frameworks Comparison
├── Lesson 07: FastAPI & Redis Microservices
└── Lesson 08: Docker & GPU Production Deployment
```

## Cài đặt nhanh

```bash
# Clone repository
git clone <repo-url>
cd exercises

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Kiểm tra GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Chạy bài thực hành

Mỗi bài thực hành có cấu trúc:
```
lesson-XX-name/
├── README.md          # Hướng dẫn chi tiết
├── starter.py         # Code skeleton để học sinh hoàn thành
├── solution.py        # Reference implementation
├── benchmark.py       # Script đánh giá metrics
└── results/           # Thư mục chứa kết quả
```

**Thứ tự thực hiện:**
```bash
# Bài 1: Baseline benchmark
cd lesson-01-transformer-vram
python starter.py
python benchmark.py

# Bài 2: Quantization lab
cd ../lesson-02-quantization
python starter.py
python benchmark.py

# ... tiếp tục cho các bài còn lại
```

## Metrics Collection

Chạy benchmark runner để thu thập tất cả metrics:
```bash
cd benchmark
python runner.py --all
```

Kết quả sẽ được lưu vào `benchmark/results/` dưới dạng JSON.

## Technical Report

Sau khi hoàn thành tất cả bài thực hành, tạo báo cáo kỹ thuật:
```bash
cd reports
python generate_report.py
```

Report sẽ được tạo tại `reports/technical_report.pdf`

## Đánh giá

| Tiêu chí | Điểm |
|----------|------|
| Code chạy thành công | 40% |
| Metrics đạt yêu cầu | 30% |
| Technical report | 20% |
| Code quality | 10% |

## Liên hệ

Giảng viên: [Tên giảng viên]
Email: [Email]
