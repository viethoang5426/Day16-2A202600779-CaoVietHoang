# Báo cáo Lab 16 — Cloud AI Environment Setup (GCP - CPU)

**Sinh viên:** Cao Việt Hoàng — 2A202600779  
**Ngày thực hiện:** 18/06/2026  
**Platform:** Google Cloud Platform  
**Phương án:** CPU (dự phòng — không có GPU quota)

---

## Thông tin triển khai

- **Project ID:** polished-core-433217-b1
- **Instance type:** e2-standard-8 (8 vCPU, 32GB RAM)
- **Zone:** us-central1-a
- **OS Image:** Ubuntu 22.04 LTS
- **ML Framework:** LightGBM 4.6.0
- **Dataset:** Credit Card Fraud Detection (284,807 giao dịch)
- **Thời gian triển khai hạ tầng (Terraform):** ~5 phút
- **Tổng thời gian từ apply → benchmark xong:** ~10 phút

## Kết quả Benchmark

| Metric | Kết quả |
|--------|---------|
| Thời gian load data | 2.26s |
| Thời gian training | 3.94s |
| Best iteration | 181 |
| AUC-ROC | **0.9666** |
| Accuracy | 99.53% |
| F1-Score | 0.3836 |
| Precision | 24.71% |
| Recall | **85.71%** |
| Inference latency (1 row) | 0.34ms |
| Inference throughput (1000 rows) | 2.43ms |

## Giải thích lý do dùng CPU thay GPU

Tài khoản GCP Free Tier mặc định khóa quota GPU ở mức 0 cho mọi Project mới.
Quá trình xin tăng quota NVIDIA T4 có thể mất từ vài phút đến 24 giờ và thường
bị từ chối đối với tài khoản chưa có lịch sử thanh toán. Do đó, phương án CPU
`e2-standard-8` (8 vCPU, 32GB RAM) được sử dụng để chạy LightGBM — một thuật toán
gradient boosting phù hợp với CPU, đạt AUC-ROC 0.9666 trên bộ dữ liệu Credit Card
Fraud Detection. Chi phí ~$0.27/giờ, rẻ hơn phương án GPU (~$0.54/giờ), và có sẵn
ngay trên tài khoản mới mà không cần chờ duyệt quota.

## So sánh CPU vs GPU (Lý thuyết)

| Tiêu chí | CPU (e2-standard-8) | GPU (n1-standard-4 + T4) |
|----------|---------------------|--------------------------|
| Chi phí/giờ | ~$0.27 | ~$0.54 |
| Cần quota đặc biệt | Không | Có (GPU quota) |
| Phù hợp cho | Gradient Boosting, ML cổ điển | Deep Learning, LLM inference |
| Thời gian sẵn sàng | Ngay lập tức | Chờ duyệt quota |
