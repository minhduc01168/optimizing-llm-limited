# LLM Engineering Lab - Tóm tắt Hoàn chỉnh

## Trạng thái triển khai

| Bài | Lesson | Code | README | Kết quả |
|-----|--------|------|--------|---------|
| 01 | Transformer VRAM | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 02 | Quantization | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 03 | Hands-on Quant | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 04 | PagedAttention | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 05 | Continuous Batching | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 06 | Serving Frameworks | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 07 | FastAPI Redis | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |
| 08 | Docker GPU | ✅ Hoàn thành | ✅ Step-by-step | ✅ Metrics template |

## Cấu trúc thư mục

```
exercises/
├── README.md                           # Hướng dẫn chính
├── requirements.txt                    # Dependencies
├── check_gpu.py                        # Kiểm tra GPU
│
├── lesson-01-transformer-vram/         # Bài 1
│   ├── README.md                       # Hướng dẫn + Kết quả đạt được
│   ├── starter.py                      # Code đã implement đầy đủ
│   └── solution.py                     # Reference implementation
│
├── lesson-02-quantization/             # Bài 2
│   ├── README.md
│   └── starter.py
│
├── lesson-03-hands-on-quant/           # Bài 3
│   ├── README.md
│   └── starter.py
│
├── lesson-04-paged-attention/          # Bài 4
│   ├── README.md
│   └── starter.py
│
├── lesson-05-continuous-batching/      # Bài 5
│   ├── README.md
│   └── starter.py
│
├── lesson-06-serving-frameworks/       # Bài 6
│   ├── README.md
│   └── starter.py
│
├── lesson-07-fastapi-redis/            # Bài 7
│   ├── README.md
│   └── starter.py
│
├── lesson-08-docker-gpu/               # Bài 8
│   ├── README.md
│   └── starter.py
│
├── benchmark/                          # Benchmark framework
│   ├── runner.py                       # Auto-run all lessons
│   └── reporter.py                     # Generate report
│
└── reports/                            # Technical reports
    ├── technical_report.md             # Report mẫu
    ├── OFFICIAL_REPORT.md              # Báo cáo chính thức
    ├── BAO_CAO_KY_THUAT_HOAN_CHINH.md # Báo cáo đầy đủ
    └── generate_report.py              # Report generator
```

## Cách sử dụng

### Bước 1: Kiểm tra GPU
```bash
cd exercises
python check_gpu.py
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Thực hiện từng bài
```bash
cd lesson-01-transformer-vram
python starter.py
```

### Bước 4: Chạy tất cả benchmarks
```bash
cd benchmark
python runner.py --all
```

### Bước 5: Tạo technical report
```bash
cd reports
python generate_report.py --name "Tên học sinh"
```

## Metrics cần thu thập cho báo cáo

| Bài | Metrics chính | Mục tiêu T4 | Kết quả mẫu | Đạt? |
|-----|---------------|-------------|-------------|------|
| 1 | VRAM usage, Latency | < 14GB, < 100ms | 7.92GB, 49.8ms | ✅ |
| 2 | VRAM reduction, Speed | -60%, +20% | -81%, 35 tok/s | ✅ |
| 3 | Compression ratio, Tok/s | 4x, > 25 | 4x, 38 tok/s | ✅ |
| 4 | Throughput, TTFT | > 5 req/s, < 500ms | 25 req/s, 150ms | ✅ |
| 5 | Batch utilization | > 80% | 85% | ✅ |
| 6 | Req/s, Latency | > 3, < 800ms | 25 req/s, 150ms | ✅ |
| 7 | Cache hit rate | > 40% | 70% | ✅ |
| 8 | Container start | < 60s | 45s | ✅ |

## Key Achievements (Tổng hợp cho báo cáo)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Model Size (Llama-3-8B) | 16GB (FP16) | 4GB (INT4) | **-75%** |
| Inference Throughput | 50 tok/s | 200+ tok/s | **+300%** |
| API Latency (p50) | 200ms | 50ms | **-75%** |
| Cache Hit Rate | 0% | 70% | **New** |
| Container Start | N/A | <60s | **Production Ready** |
| Batch Utilization | 50% | 85% | **+35%** |
| VRAM Efficiency | ~50% | >80% | **+30%** |

## CV-Ready Summary

```
LLM Engineering Project - NVIDIA T4 (16GB VRAM)
────────────────────────────────────────────────

ACHIEVEMENTS:
• Tăng inference throughput 4x (50→200+ tok/s) 
  với PagedAttention và continuous batching

• Giảm model memory 75% bằng INT4 quantization 
  (GPTQ/AWQ), accuracy loss <5%

• Xây dựng LLM API với FastAPI + Redis caching,
  đạt 70% cache hit rate, giảm latency 75%

• Deploy containerized service với Docker,
  cold start <60s, GPU overhead <5%

TECH STACK:
PyTorch | vLLM | FastAPI | Redis | Docker | NVIDIA T4
```

## Liên hệ

Giảng viên: Vũ Minh Khang
