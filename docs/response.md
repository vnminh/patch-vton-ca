# Tài liệu đã được hợp nhất

Nội dung refiner trước đây trong file này đã được hợp nhất và cập nhật vào:

- [`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md): luồng model đầy đủ;
- [`VTON_DETAIL.md`](VTON_DETAIL.md): Q/K/V, coherent warp và refiner;
- [`VTON_HF_CONTROL.md`](VTON_HF_CONTROL.md): RGB+HF feature fusion.

Không dùng mô tả `detail_velocity + hf_velocity` từ các revision cũ. Graph hiện
tại chỉ có một detail residual:

```text
hf_delta_raw = RGB_RMS * tanh(zero_init_projection(normalize(hf_warped_feature)))
hf_delta = spatial_center(hf_delta_raw, edit_support)
rgb_warped_feature + hf_delta
  -> shared refiner
  -> fine_velocity

final_velocity = backbone_velocity + fine_velocity
```

`fine_velocity` không được cộng tự do: warped HF energy đã detach tạo activity gate
`[0.25,1]`, learned spatial/channel gate tinh chỉnh vị trí, spatial centering loại
latent DC theo từng channel, và hard RMS gate giới hạn authority ở 10% backbone
(floor 0.05) trong cả train lẫn inference. Decoded garment RGB/low-pass/channel-mean
loss giữ màu tuyệt đối, thay vì để edge objective đổi hue hoặc độ sáng.

Garment K được LayerNorm để routing ổn định nhưng V không còn bị khóa ở per-token
LayerNorm: scalar zero-init học mở payload raw/global-RMS giữ magnitude màu và logo.
Decoded supervision chạy cả `t=0`. Config này dùng trực tiếp student (`ema_rate=0`),
nên preview phản ánh đúng graph đang học và không tốn thêm một bản PFT-XL trong VRAM.
