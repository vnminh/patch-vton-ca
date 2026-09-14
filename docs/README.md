# Bản đồ tài liệu VTON-PFT

Tài liệu được sắp theo thứ tự từ tổng quát đến chuyên sâu. Nếu mới tiếp cận
repository, hãy đọc theo thứ tự dưới đây.

## Kiến trúc hiện tại

1. [`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md) — tài liệu chính: mục tiêu, sơ đồ
   end-to-end, trách nhiệm từng khối, loss, gradient, train và inference.
2. [`MODEL_FLOW_VI.md`](MODEL_FLOW_VI.md) — bảng tra cứu shape và tensor tại từng
   ranh giới của model 512x384.
3. [`VTON_DETAIL.md`](VTON_DETAIL.md) — chi tiết thuật toán correspondence,
   coherent warp và latent refiner 64x48.
4. [`VTON_HF_CONTROL.md`](VTON_HF_CONTROL.md) — chi tiết signed-RGB HF,
   RGB+HF feature fusion, zero-init và migration checkpoint.
5. [`VTON_GARMENT_FIX.md`](VTON_GARMENT_FIX.md) — quy tắc mask/supervision được
   giữ lại từ revision garment-mask.

Config chuẩn đang được mô tả là:

```text
configs/experiment/viton-pft-xl-512x384-detail-logo-hf.yaml
```

## Thiết kế nền và lịch sử

- [`VTON_PFT_DESIGN.md`](VTON_PFT_DESIGN.md) mô tả thiết kế PFT/VTON nền, flow
  convention và patch-wise sampling. Một số phần garment/refiner trong đó có
  trước kiến trúc fused-HF; dùng `ARCHITECTURE_VI.md` khi có khác biệt.
- [`VTON_TRAINING_AUDIT_20260908.md`](VTON_TRAINING_AUDIT_20260908.md) là audit
  của một run cũ. Các số liệu trong đó là bằng chứng lịch sử, không phải mô tả
  graph hiện tại hay cam kết chất lượng.

## Quy tắc duy trì docs

- `ARCHITECTURE_VI.md` là nguồn sự thật cho graph hiện tại.
- `MODEL_FLOW_VI.md` chỉ giữ shape/tensor contract, không lặp lại lập luận thiết kế.
- `VTON_DETAIL.md` và `VTON_HF_CONTROL.md` chỉ đào sâu khối tương ứng.
- Audit phải ghi rõ ngày, checkpoint và revision; không dùng audit làm hướng dẫn
  chạy model hiện tại.
- Khi đổi graph, cập nhật sơ đồ output và checkpoint migration trước khi cập nhật
  số liệu thực nghiệm.
