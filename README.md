# Lab 16 — Cloud AI Environment Setup on GCP (CPU Fallback)

**Sinh viên:** Cao Việt Hoàng  
**MSSV:** 2A202600779  
**Ngày thực hiện:** 18/06/2026  
**Platform:** Google Cloud Platform (GCP)  
**Phương án:** CPU dự phòng (e2-standard-8) — do không xin được GPU quota

---

## 1. Tổng quan dự án

Dự án Lab 16 yêu cầu thiết lập một môi trường Cloud AI hoàn chỉnh bằng **Terraform (Infrastructure as Code)**, triển khai mô hình AI lên Cloud server nằm trong **Private VPC**, và phục vụ API inference ra bên ngoài qua **Load Balancer**.

Do tài khoản GCP Free Tier không có GPU quota, phương án **CPU dự phòng** được sử dụng: triển khai **LightGBM** (gradient boosting) trên instance **e2-standard-8** (8 vCPU, 32GB RAM) với dataset **Credit Card Fraud Detection** (284,807 giao dịch).

---

## 2. Kiến trúc hạ tầng đã triển khai

```
┌──────────────────────────────────────────────────────┐
│                    INTERNET                          │
│                       │                              │
│              ┌────────▼────────┐                     │
│              │ External HTTP   │                     │
│              │ Load Balancer   │                     │
│              │ IP: 8.233.200.115                     │
│              │ Port 80 → 8000  │                     │
│              └────────┬────────┘                     │
│                       │                              │
│  ┌────────────────────▼──────────────────────────┐   │
│  │           GCP VPC: ai-vpc                     │   │
│  │                                               │   │
│  │  ┌─────────────────────────────────────────┐  │   │
│  │  │  Private Subnet: 10.0.0.0/24            │  │   │
│  │  │  (us-central1)                          │  │   │
│  │  │                                         │  │   │
│  │  │  ┌───────────────────────────────────┐  │  │   │
│  │  │  │  VM: ai-gpu-node                  │  │  │   │
│  │  │  │  Type: e2-standard-8              │  │  │   │
│  │  │  │  (8 vCPU, 32GB RAM)               │  │  │   │
│  │  │  │  OS: Ubuntu 22.04 LTS             │  │  │   │
│  │  │  │  Disk: 50GB SSD                   │  │  │   │
│  │  │  └───────────────────────────────────┘  │  │   │
│  │  └─────────────────────────────────────────┘  │   │
│  │                                               │   │
│  │  Cloud Router ──► Cloud NAT (egress internet) │   │
│  │  IAP SSH (35.235.240.0/20 → port 22)          │   │
│  │  LB Health Check (130.211.0.0/22 → port 8000) │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

**16 tài nguyên GCP** được tạo bởi Terraform:

| # | Resource | Tên |
|---|---------|-----|
| 1 | VPC Network | `ai-vpc` |
| 2 | Private Subnet | `ai-private-subnet` (10.0.0.0/24) |
| 3 | Cloud Router | `ai-router` |
| 4 | Cloud NAT | `ai-nat` (AUTO_ONLY IP) |
| 5 | Firewall — IAP SSH | `allow-iap-ssh` (35.235.240.0/20 → port 22) |
| 6 | Firewall — LB Health | `allow-lb-healthcheck` (130.211.0.0/22 → port 8000) |
| 7 | Service Account | `gpu-node-sa` |
| 8 | IAM — Log Writer | `roles/logging.logWriter` |
| 9 | IAM — Metric Writer | `roles/monitoring.metricWriter` |
| 10 | Compute Instance | `ai-gpu-node` (e2-standard-8, us-central1-a) |
| 11 | Instance Group | `ai-gpu-group` |
| 12 | Health Check | `vllm-health-check` (HTTP /health:8000) |
| 13 | Backend Service | `vllm-backend` (timeout 300s) |
| 14 | URL Map | `vllm-url-map` |
| 15 | HTTP Proxy | `vllm-http-proxy` |
| 16 | Forwarding Rule | `vllm-forwarding-rule` (port 80, External IP) |

---

## 3. Các thao tác đã thực hiện (theo thứ tự)

### 3.1 Cài đặt công cụ trên máy local (macOS)

```bash
# Kiểm tra gcloud CLI (đã có sẵn v570.0.0)
gcloud --version

# Cài Terraform qua Homebrew
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
# → Terraform v1.15.6 on darwin_arm64
```

### 3.2 Xác thực GCP

```bash
# Đăng nhập tài khoản Google
gcloud auth login
# → Logged in as caohoang5969@gmail.com

# Cấp Application Default Credentials cho Terraform
gcloud auth application-default login

# Đặt project mặc định
gcloud config set project polished-core-433217-b1
```

### 3.3 Kích hoạt APIs

```bash
gcloud services enable compute.googleapis.com iam.googleapis.com --project=polished-core-433217-b1
gcloud services enable iap.googleapis.com --project=polished-core-433217-b1
```

### 3.4 Sửa Terraform cho phương án CPU

**File `terraform-gcp/main.tf`** — 2 thay đổi:

1. **Đổi image** từ Deep Learning VM sang Ubuntu 22.04 LTS (không cần CUDA):
```hcl
# Trước (GPU):
image = "projects/deeplearning-platform-release/global/images/family/common-cu121-debian-11"
size  = 100

# Sau (CPU):
image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
size  = 50
```

2. **Comment out GPU accelerator** và đổi scheduling:
```hcl
# guest_accelerator {        ← Comment out
#   type  = var.gpu_type
#   count = var.gpu_count
# }

scheduling {
  on_host_maintenance = "MIGRATE"   # Đổi từ "TERMINATE" → "MIGRATE"
  automatic_restart   = true
}
```

### 3.5 Triển khai hạ tầng bằng Terraform

```bash
cd terraform-gcp

# Thiết lập biến môi trường
export TF_VAR_project_id="polished-core-433217-b1"
export TF_VAR_machine_type="e2-standard-8"
export TF_VAR_gpu_count=0
export TF_VAR_hf_token="dummy"
export TF_VAR_zone="us-central1-a"

# Khởi tạo
terraform init
# → Installed hashicorp/google v5.45.2

# Triển khai
terraform apply -auto-approve
# → Apply complete! Resources: 16 added
```

**Lưu ý gặp phải:**
- Lần 1: Lỗi image Deep Learning VM không tồn tại → Đã đổi sang Ubuntu 22.04 LTS
- Lần 2-4: Lỗi `n2-standard-8` hết tài nguyên ở zones us-central1-a/b/c → Đã đổi sang `e2-standard-8` thành công

**Terraform Outputs:**
```
api_endpoint     = "http://8.233.200.115/v1"
gpu_node_name    = "ai-gpu-node"
gpu_node_zone    = "us-central1-a"
iap_ssh_command  = "gcloud compute ssh ai-gpu-node --zone=us-central1-a --tunnel-through-iap"
load_balancer_ip = "8.233.200.115"
```

### 3.6 SSH vào VM qua IAP và cài đặt môi trường ML

```bash
# SSH vào VM (không cần key, dùng IAP)
gcloud compute ssh ai-gpu-node --zone=us-central1-a --tunnel-through-iap --project=polished-core-433217-b1

# Trên VM: Cài Python packages
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv
pip3 install lightgbm scikit-learn pandas numpy

# Packages đã cài:
# - lightgbm 4.6.0
# - scikit-learn 1.7.2
# - pandas 2.3.3
# - numpy 2.2.6
# - scipy 1.15.3
```

### 3.7 Upload và chạy Benchmark

```bash
# Upload benchmark.py lên VM
gcloud compute scp benchmark.py ai-gpu-node:~/ml-benchmark/benchmark.py \
  --zone=us-central1-a --tunnel-through-iap --project=polished-core-433217-b1

# Chạy benchmark trên VM
gcloud compute ssh ai-gpu-node --zone=us-central1-a --tunnel-through-iap \
  --project=polished-core-433217-b1 --command="cd ~/ml-benchmark && python3 benchmark.py"
```

### 3.8 Dọn dẹp tài nguyên

```bash
terraform destroy -auto-approve
# → Destroy complete! Resources: 16 destroyed.
```

---

## 4. Kết quả Benchmark

**Dataset:** Credit Card Fraud Detection — 284,807 giao dịch (tỷ lệ fraud: 0.17%)  
**Model:** LightGBM (gradient boosting) — 181 iterations, early stopping 50 rounds  
**Instance:** GCP e2-standard-8 (8 vCPU, 32GB RAM, Ubuntu 22.04)

| Metric | Kết quả |
|--------|---------|
| Thời gian load data | **2.26s** |
| Thời gian training | **3.94s** |
| Best iteration | **181** |
| AUC-ROC | **0.9666** |
| Accuracy | **99.53%** |
| F1-Score | **0.3836** |
| Precision | **24.71%** |
| Recall | **85.71%** |
| Inference latency (1 row) | **0.34ms** |
| Inference throughput (1000 rows) | **2.43ms** |

**Hyperparameters đã sử dụng:**
```python
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": 577,  # Xử lý mất cân bằng lớp (99.83% vs 0.17%)
    "num_threads": 8,
}
```

---

## 5. So sánh CPU vs GPU

| Tiêu chí | CPU (e2-standard-8) | GPU (n1-standard-4 + T4) |
|----------|---------------------|--------------------------|
| Chi phí/giờ | ~$0.27 | ~$0.54 |
| vCPU / RAM | 8 vCPU / 32GB | 4 vCPU / 15GB + 16GB VRAM |
| Cần quota đặc biệt | Không | Có (GPU quota) |
| Thời gian sẵn sàng | Ngay lập tức | Chờ duyệt 1-24h |
| Workload phù hợp | ML cổ điển (LightGBM, XGBoost) | Deep Learning, LLM inference |
| Model deploy | LightGBM (gradient boosting) | vLLM + Gemma-4-E2B-it |

**Lý do chọn CPU:** Tài khoản GCP Free Tier mặc định khóa GPU quota ở mức 0. Việc xin tăng quota NVIDIA T4 thường bị từ chối hoặc mất thời gian duyệt lâu. Instance `e2-standard-8` có sẵn ngay, chi phí rẻ hơn 50%, và đủ mạnh để chạy gradient boosting trên dataset 284K rows.

---

## 6. Cấu trúc thư mục dự án

```
Day16-2A202600779-CaoVietHoang/
├── README.md                    ← File này
├── README_aws.md                # Hướng dẫn Lab trên AWS (tham khảo)
├── README_gcp.md                # Hướng dẫn Lab trên GCP (tham khảo)
├── .env.example                 # Template biến môi trường
├── .gitignore                   # Ignore Terraform state, SSH keys
│
├── terraform/                   # Terraform cho AWS (không sử dụng)
│   ├── main.tf
│   ├── variables.tf
│   ├── providers.tf
│   ├── outputs.tf
│   └── user_data.sh
│
└── terraform-gcp/               # ★ Terraform cho GCP (đã sử dụng)
    ├── main.tf                  # Hạ tầng GCP (đã sửa cho CPU)
    ├── variables.tf             # Biến: project_id, machine_type, gpu_type...
    ├── providers.tf             # Provider Google ~> 5.0
    ├── outputs.tf               # Output: LB IP, API endpoint, SSH command
    ├── user_data.sh             # Startup script (gốc — cho GPU/vLLM)
    ├── benchmark.py             # ★ Script benchmark LightGBM
    ├── benchmark_result.json    # ★ Kết quả benchmark (JSON)
    └── bao_cao_lab16.md         # ★ Báo cáo ngắn nộp bài
```

---

## 7. Các lỗi gặp phải và cách xử lý

| # | Lỗi | Nguyên nhân | Cách xử lý |
|---|-----|-------------|------------|
| 1 | `Billing account not found` | Chưa liên kết Billing cho Project | Vào Billing Console → Link billing account |
| 2 | `Image not found: common-cu121-debian-11` | Deep Learning VM image không tồn tại | Đổi sang `ubuntu-os-cloud/ubuntu-2204-lts` |
| 3 | `Not enough resources: n2-standard-8` (3 zones) | Region us-central1 hết n2 capacity | Đổi sang `e2-standard-8` (dòng E2 phổ biến hơn) |
| 4 | `pip3 --break-system-packages not recognized` | Ubuntu 22.04 dùng pip 22.0 (cũ) | Bỏ flag `--break-system-packages` |
| 5 | `SCP dest not found` | Thư mục ~/ml-benchmark chưa tồn tại trên VM | Chạy `mkdir -p ~/ml-benchmark` trước khi SCP |

---

## 8. Deliverables (Bài nộp)

- [x] Screenshot terminal chạy `benchmark.py` với output kết quả
- [x] File `benchmark_result.json` chứa metrics đầy đủ
- [x] Screenshot GCP Billing Reports
- [x] Mã nguồn `terraform-gcp/` đã chỉnh sửa cho CPU
- [x] Báo cáo ngắn so sánh CPU vs GPU (`bao_cao_lab16.md`)
- [x] `terraform destroy` đã chạy — tài nguyên đã dọn dẹp
