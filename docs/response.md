# Tài liệu đã được hợp nhất

Nội dung refiner trước đây trong file này đã được hợp nhất và cập nhật vào:

- [`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md): luồng model đầy đủ;
- [`VTON_DETAIL.md`](VTON_DETAIL.md): Q/K/V, coherent warp và refiner;
- [`VTON_HF_CONTROL.md`](VTON_HF_CONTROL.md): RGB+HF feature fusion.

Không dùng mô tả `detail_velocity + hf_velocity` từ các revision cũ. Graph hiện
tại chỉ có một detail residual:

```text
rgb_warped_feature + hf_warped_feature
  -> shared refiner
  -> fine_velocity

final_velocity = backbone_velocity + fine_velocity
```
