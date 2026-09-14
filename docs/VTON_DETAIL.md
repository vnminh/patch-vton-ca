# Correspondence, coherent warp và latent refiner

Tài liệu này đào sâu khối detail 64x48 của kiến trúc hiện tại. Xem sơ đồ tổng thể
tại [`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md) và bảng shape tại
[`MODEL_FLOW_VI.md`](MODEL_FLOW_VI.md).

## 1. Tại sao cần refiner 64x48

Backbone PFT dùng patch 2x2 trên latent nên chỉ có lưới 32x24. Một token backbone
đại diện cho vùng ảnh khoảng 16x16 pixel; quá thô để phân biệt stroke nhỏ trong
logo. `GarmentLatentRefiner` chạy sau 28 DiT block và tạo một query/output riêng
cho từng latent cell 64x48, tương ứng vùng ảnh khoảng 8x8.

Refiner không thay backbone. Nó chỉ dự đoán residual:

```text
final_velocity = backbone_velocity + fine_velocity
```

## 2. Tạo query phía người

```text
x cuối DiT: B x 768 x 1152
  -> Linear(1152, 4*256)
  -> reshape + PixelShuffle(2)
  -> B x 256 x 64 x 48
```

Query expansion tạo bốn subquery học được cho bốn latent cell bên trong một PFT
patch. Feature này được cộng với:

```text
state_conv([noisy latent, agnostic latent, DensePose latent])
```

`state_conv` zero-init để checkpoint cũ giữ nguyên output ở bước đầu, nhưng nó có
gradient ngay vì downstream Q/K path đã khác zero. DensePose đi vào đây để routing
phân biệt tay, thân và hình dạng garment ở full latent resolution.

## 3. Tạo key và value phía garment

Garment detail VAE feature được embed thành `B x 3072 x 1152`, đúng grid 64x48.

```text
Q = projected(normalized person query + PE)
K = per-token-LN(projected garment detail) + PE
V_old = per-token-LN(projected garment detail)
V_mag = raw projected garment detail / global sample RMS
V = V_old + clamp_ST(value_mix, 0, 1) * (V_mag - V_old)
```

Sau đó Q/K được normalize theo từng head và nhân cosine temperature có giới hạn.
V không chứa PE: vị trí quyết định **lấy ở đâu**, còn value mang **appearance gì**.
`value_mix` có một scalar cho coarse/middle/detail và khởi tạo zero, nên checkpoint
cũ không đổi output ở bước đầu. Khác với per-token LayerNorm, global RMS không xóa
mean/scale của từng vị trí; đó là payload cần để phân biệt nét logo với màu nền áo.

Garment mask được max-pool về 64x48 để loại background keys. Empty mask được xử lý
để SDPA luôn finite, rồi `active` gate ép residual cuối về đúng zero.

## 4. Coarse-to-fine coherent warp

Soft attention toàn cục dễ trung bình hai vùng áo có màu/chữ khác nhau. Refiner
dùng một deformation grid nhất quán:

1. Pool Q/K theo head từ 64x48 xuống 32x24.
2. Trung bình score của 8 head rồi hard-select một garment anchor chung.
3. Upsample coarse displacement, không upsample absolute coordinate trực tiếp.
4. Tại 64x48, trung bình local score để chọn một residual chung trong radius 2.
5. Dùng cùng grid vật lý để bilinear-sample tám nhóm channel của garment V.

```text
warped = sample_attention_heads(V, sampling_grid)
rgb_warped_feature = attention_out(warped)
```

Trong config logo, global RGB `A@V/warp_mix` và global HF đều bị tắt. Chỉ coherent
grid hoạt động; mỗi nhóm channel không còn được lấy từ một garment location khác.

## 5. Quan hệ với HF

HF không tự học một attention map thứ hai. `GarmentHighFrequencyControl` sao chép
Q/K/key mask/grid của RGB route, detach chúng trên HF path và chỉ thay V:

```text
rgb_warped_feature = warp(RGB V, shared grid)
hf_warped_feature  = warp(HF V,  shared grid)
hf_delta           = RGB_RMS * tanh(Conv_zero(GroupNorm(hf_warped_feature)))
fused_feature      = rgb_warped_feature + hf_delta
```

Nhờ đó logo edge và RGB/chroma cùng đến một person location. HF feature/decoded loss
không thể kéo correspondence sang một vị trí dễ tạo edge nhưng sai garment; routing
vẫn do DINO anchor, paired RGB/VAE feature và raw-HF source consistency quản lý.

## 6. Refine fused feature thành velocity

Refiner dùng backbone state để biết residual cần sửa tại timestep hiện tại:

```text
preliminary_clean = noisy + (1-t) * backbone_velocity
condition = Conv([stopgrad(backbone_velocity), stopgrad(preliminary_clean)])
modulated = fused_feature * (1 + tanh(condition))
decoded = modulated + local(modulated)
raw_fine = output_bias_free(decoded)
learned_gate = 2 * sigmoid(gate_zero_init(decoded))
activity_gate = .25 + .75 * dilate(normalize_energy(stopgrad(hf_warped)))
fine_ac = spatial_center(raw_fine * activity_gate * learned_gate, edit_support)
fine_velocity = fine_ac * min(1, max(.1 * backbone_rms, .05) / rms(fine_ac))
```

Đây là modulation nhân, không phải person-only additive shortcut. Nếu garment/HF
rỗng thì fused feature và fine velocity đều bằng zero.

Activity gate bảo đảm fine correction tập trung quanh detail thực sự đã warp; learned
gate tinh chỉnh theo channel. Spatial centering cấm latent DC shift. Trust
region cứng chạy cả inference và giới hạn fine RMS ở 10% backbone (floor 0.05);
penalty mềm trên raw fine giúp refiner không phụ thuộc lâu dài vào clipping.

## 7. Supervision cho routing

### 7.1 DINO teacher chỉ là sparse anchor

DINO tạo match garment-person đáng tin ở lưới teacher. Match được lọc bằng:

- similarity/margin;
- cycle return trong tolerance;
- local displacement consistency;
- garment/edit mask.

`correspondence_mutual: false` cho phép cycle quay về trong bán kính 1.5 person
token thay vì đòi exact mutual nearest neighbor. DINO không được coi là ground
truth cho từng stroke 8x8.

### 7.2 Dense propagation

Displacement từ reliable anchors được truyền sang vùng garment lân cận với confidence
thấp hơn. Reliable anchors điều khiển coarse correspondence; RGB/value reconstruction
và local grid học fine displacement dưới độ phân giải DINO.

### 7.3 Các loss chính

- `fine_correspondence_loss`: xác suất/mass quanh reliable target;
- `fine_warp_coordinate_loss`: grid gần anchor/propagated target;
- `fine_warp_smoothness_loss`: phạt bending không hợp lý;
- `fine_warp_mask_loss`: không sample background garment;
- `fine_rgb_loss`: RGB được warp phải giống paired worn garment;
- `fine_value_loss`: transported VAE feature phải đúng target feature.

Mọi loss trên chỉ chấm garment pixels. Decoded reconstruction riêng chấm toàn edit
region để giữ pose/tay.

## 8. Điều không được hiểu nhầm

- `rgb_warped_feature` được tạo **sau DiT**, nhưng không phải chính output `x` của DiT.
- Refiner Q đến từ người; garment K/V đến từ ảnh sản phẩm.
- K có PE, V không có PE.
- HF bổ sung feature trước refiner, không cộng edge vào velocity.
- DINO hướng dẫn geometric anchor, không trực tiếp encode ý nghĩa chữ/logo.
- Fine grid 64x48 là latent resolution, không phải full 512x384 pixel attention.

## 9. Metric chẩn đoán

| Triệu chứng | Metric cần xem |
|---|---|
| Match sai vị trí | `fine_top1_accuracy`, `fine_target_mass`, coordinate loss |
| Match đúng nhưng sai màu | `fine_rgb_loss`, value loss, decoded chroma |
| Edge có nhưng logo vô nghĩa | decoded RGB/chroma + preview, không chỉ edge loss |
| Refiner không có authority | `fine_velocity_rms`, `garment_grad/refiner/output` |
| Refiner lấn át backbone | raw/final RMS, `fine_velocity_norm_gate`, limit |
| Áo bị lệch màu đồng đều | `fine_velocity_dc_rms`, `hf_fusion_removed_dc_rms` |
| Routing tốt nhưng chỉ ra màu trung bình | `garment_value_mix_*`, decoded RGB/mean |
| Gate vẫn gần identity | activity/effective gate, không chỉ learned gate mean |
| HF không sống | `hf_feature_rms`, `hf_fusion_delta_rms`, fusion/encoder gradients |

## 10. Code map

- `GarmentLatentRefiner.route`: tạo Q/K/V và coherent sampling grid;
- `GarmentLatentRefiner.refine`: modulation + local decoder thành fine velocity;
- `GarmentHighFrequencyControl.forward`: transport HF V theo shared grid;
- `LatentVTONPatchForcingTrainer._fine_losses`: loss routing/value/RGB;
- `LatentVTONPatchForcingTrainer._hf_source_consistency_loss`: paired HF transport.
