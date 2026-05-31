# Lesson 08: Docker & GPU Production Deployment

## Mục tiêu

Đóng gói LLM service với Docker và deploy lên production

## Acceptance Criteria

- [ ] AC-8.1: Tạo Dockerfile cho LLM service
- [ ] AC-8.2: Docker compose với GPU support
- [ ] AC-8.3: Health check và monitoring
- [ ] AC-8.4: Đo container metrics

## Architecture

```
┌─────────────────────────────────────────┐
│              Docker Host                 │
│  ┌──────────────────────────────────┐  │
│  │     Docker Compose                │  │
│  │  ┌─────────────┐ ┌─────────────┐ │  │
│  │  │  LLM API    │ │   Redis     │ │  │
│  │  │  (GPU)      │ │   (CPU)     │ │  │
│  │  └──────┬──────┘ └──────┬──────┘ │  │
│  │         │               │         │  │
│  │         └───────────────┘         │  │
│  │              │                     │  │
│  │     ┌────────┴────────┐           │  │
│  │     │  NVIDIA T4 GPU  │           │  │
│  │     └─────────────────┘           │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Docker Commands

```bash
# Build image
docker build -t llm-service .

# Run with GPU
docker run --gpus all -p 8000:8000 llm-service

# Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Metrics cần thu thập

| Metric | Mục tiêu | Cách đo |
|--------|----------|---------|
| Container start time | < 60s | `time docker run` |
| Image size | < 8GB | `docker images` |
| GPU overhead | < 5% | nvidia-smi comparison |
| Health check | < 100ms | curl timing |
| Memory usage | < 2GB overhead | docker stats |

## Hướng dẫn chi tiết từng bước

### Bước 1: Chuẩn bị môi trường
```bash
cd exercises/lesson-08-docker-gpu
pip install torch
# Cần cài đặt Docker và NVIDIA Container Toolkit
```

### Bước 2: Mở file starter.py
Mở file `starter.py` trong thư mục bài tập.

### Bước 3: Hoàn thành các hàm TODO
Thứ tự hoàn thành:
1. Hàm `generate_dockerfile()` - Tạo Dockerfile cho LLM service (base image nvidia/cuda, Python, dependencies, healthcheck, CMD)
2. Hàm `generate_docker_compose()` - Tạo docker-compose.yml (llm-api service với GPU, redis service, volumes)
3. Hàm `build_image()` - Build Docker image (docker build, parse image size)
4. Hàm `start_container()` - Start container với GPU support (docker run --gpus all)
5. Hàm `stop_container()` - Dừng container (docker stop, docker rm)
6. Hàm `measure_health_check()` - Đo health check response time (urllib, ms)
7. Hàm `get_container_stats()` - Lấy container resource usage (docker stats, parse memory và CPU)
8. Hàm `measure_gpu_overhead()` - Đo GPU overhead khi chạy trong Docker (matrix multiply benchmark)
9. Hàm `run_deployment_benchmark()` - Chạy benchmark deployment hoàn chỉnh (generate → build → start → health check → stats)

### Bước 4: Chạy bài thực hành
```bash
python starter.py
```

### Bước 5: Kiểm tra kết quả
Kết quả sẽ được lưu vào `results/metrics.json`. Kiểm tra:
- Image size (GB)
- Build time (s)
- Container start time (s)
- GPU overhead (%)
- Health check response time (ms)
- Memory usage (MB)
- CPU usage (%)

Hoặc chạy Docker Compose trực tiếp:
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Kết quả đạt được

### Metrics sau khi hoàn thành bài thực hành

| Metric | Giá trị đạt được | Mục tiêu | Đạt? |
|--------|-----------------|----------|------|
| Container start time (s) | ... | < 60s | ✅/❌ |
| Image size (GB) | ... | < 8GB | ✅/❌ |
| GPU overhead (%) | ... | < 5% | ✅/❌ |
| Health check (ms) | ... | < 100ms | ✅/❌ |
| Memory usage (MB) | ... | < 2GB overhead | ✅/❌ |

### Kiến thức đã nắm được
- Cách viết Dockerfile cho LLM service với GPU support
- NVIDIA Container Toolkit và cách expose GPU vào container
- Docker Compose multi-service architecture (LLM API + Redis)
- Health check, restart policy, và resource limits trong production

### Kỹ năng đã thành thạo
- Tạo Dockerfile với multi-stage build và GPU support
- Cấu hình docker-compose.yml với GPU reservations
- Đo container metrics (start time, image size, resource usage)
- Implement health check và monitoring cho LLM service

### Dữ liệu cho báo cáo
```json
{
  "lesson": "08-docker-gpu",
  "completion_date": "[ngày hoàn thành]",
  "metrics": {
    "image_size_gb": "[giá trị]",
    "build_time_s": "[giá trị]",
    "container_start_s": "[giá trị]",
    "gpu_overhead_pct": "[giá trị]",
    "health_check_ms": "[giá trị]"
  },
  "status": "completed"
}
```
