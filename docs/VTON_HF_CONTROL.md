# Garment high-frequency feature fusion

Tài liệu này giải thích riêng nhánh HF của config
`viton-pft-xl-512x384-detail-logo-hf`. Đọc [`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md)
trước để biết vị trí của nhánh này trong toàn model.

## 1. Vai trò của HF

Garment RGB/VAE mang màu, vùng logo và texture nhưng các stroke nhỏ có thể suy giảm
qua VAE/attention. HF bổ sung signed detail và gradient màu để refiner biết nơi cần
tái tạo biên, nét chữ và color-block.

HF **không tự sinh ảnh RGB**. Nó bổ sung thông tin cho RGB warped feature; refiner
chung mới học cách chuyển thông tin kết hợp thành residual RGB/latent hợp lệ.

```text
rgb_warped_feature + bounded_hf_delta
  -> shared detail refiner
  -> fine_velocity

final_velocity = backbone_velocity + fine_velocity
```

Không có `hf_velocity` bốn channel cộng độc lập.

## 2. Vì sao không dùng grayscale Canny

Canny chỉ cho biết có biên hay không. Nó làm mất:

- dấu sáng/tối của stroke;
- kênh màu tạo nên chữ/logo;
- isoluminant color boundary, ví dụ đỏ/xanh cùng độ sáng;
- màu bên trong logo và color-block.

Revision hiện tại dùng sáu channel:

| Channel | Nội dung |
|---|---|
| `0:3` | signed RGB high-pass/DoG đa tỉ lệ |
| `3` | luma gradient magnitude |
| `4` | opponent-chroma gradient magnitude |
| `5` | max RGB gradient magnitude |

Map được tính **sau** flip/shift/scale của garment và bị giới hạn trong eroded
garment mask. Như vậy RGB, mask và HF luôn cùng hệ tọa độ.

## 3. Tại sao HF chỉ lấy từ garment

Inference unpaired không có ground-truth người mặc garment đích. Vì vậy model
condition chỉ dùng:

```text
in-shop garment + garment mask -> garment HF
```

`person_high_frequency` có thể tồn tại trong paired batch nhưng chỉ làm target cho
source-consistency loss. Nó không đi vào forward condition và không cần ở inference.

## 4. HF encoder

Sáu input channel được tách thành hai ảnh ba channel:

1. signed RGB DoG;
2. luma/chroma/RGB gradients.

Mỗi ảnh đi qua cùng kiểu SD-VAE half-resolution stem, tạo 128 feature channel tại
256x192. Response VAE của blank input tương ứng được trừ riêng, sau đó concat:

```text
signed VAE detail      B x 128 x 256 x 192
gradient VAE detail    B x 128 x 256 x 192
------------------------------------------------
HF condition           B x 256 x 256 x 192
```

HF condition stem được init từ pretrained VAE và được phép học với LR multiplier
0.05. Target VAE và decoder VAE chính vẫn frozen.

## 5. Shared correspondence, different value

RGB refiner đã tạo:

- Q từ person/DiT state;
- K từ garment detail;
- `key_valid` từ garment mask;
- coherent `sampling_grid` từ coarse anchor + local residual.

HF dùng lại đúng bốn tensor trên. Q/K/grid được detach trên HF path:

```text
RGB: shared Q/K/grid + learned RGB V + direct garment latent -> rgb_warped_feature
HF : shared Q/K/grid + HF  V -> hf_warped_feature
```

Điều này bảo đảm nét HF và RGB được lấy từ cùng garment location. Auxiliary edge
loss không thể tự bẻ attention map sang nơi có biên mạnh nhưng sai logo.

## 6. High-resolution transport

`GarmentHighFrequencyControl` thực hiện:

1. `encoder`: 1x1 bias-free, 256 -> 256, có feature ngay từ đầu;
2. trừ spatial mean trên valid garment pixels;
3. local high-resolution processing ở 256x192;
4. upsample displacement grid và warp HF value tại 256x192;
5. stride-4 downsample sau warp về 64x48;
6. local block + `feature_out` bias-free tạo 256 channel;
7. trừ spatial DC lần cuối trên edit support;
8. edit/active gate ép vùng không hợp lệ về zero.

Warp trước rồi mới downsample giữ các phase nhỏ của stroke tốt hơn downsample HF
trước correspondence.

Spatial mean được loại ở HF delta và ở fine velocity cuối, không loại khỏi warped RGB
carrier. RGB carrier phải giữ màu nền thật; phần residual chỉ được phép mô tả contrast
không gian, không được dùng một offset latent đồng đều để đổi hue.

Global HF và global RGB attention đều bị tắt trong config logo. Chúng có xu hướng
trung bình các vùng xa nhau và tạo glyph-like texture; coherent grid là route duy
nhất đang hoạt động. Tám value head dùng cùng một grid vật lý, không còn tám warp.

## 7. Zero initialization

Zero-init nằm tại projection `garment_refiner.hf_fusion`, không nằm trong HF
extractor hay velocity head.

- Bước đầu: raw HF feature có nghĩa, nhưng `hf_delta` bằng đúng zero nên checkpoint
  cũ giữ nguyên prediction.
- `hf_fusion.weight` nhận gradient ngay update đầu.
- Khi fusion mở, flow/decoded loss tiếp tục cập nhật upstream HF extractor. Source
  consistency dùng raw HF map để giám sát grid, không đi qua extractor.

Nếu zero cả output weight, gradient encoder cũng bằng zero cho tới khi output head
đã thay đổi; revision cũ từng gần như không học vì vấn đề này.

Empty garment, garment dropout và unconditional CFG đều bị `active` gate ép HF
output chính xác zero; các projection có thể tạo DC đều được thiết kế bias-free.

## 8. Fusion/refiner

Sau routing:

```text
rgb_warped_feature = learned_V_warp + Conv_zero(direct_garment_latent_warp)
fused = rgb_warped_feature + bounded_hf_delta   # 256 x 64 x 48
fine_velocity = garment_refiner.refine(fused, detached backbone state)
final_velocity = backbone_velocity + fine_velocity
```

Hai feature không được cộng raw vì chúng là hai basis học độc lập. Fusion hiện tại:

```text
hf_delta_raw = rms(stopgrad(rgb_warped))
               * tanh(Conv_zero(GroupNorm(hf_warped)))
hf_delta = spatial_center(hf_delta_raw, edit_support)
fused = rgb_warped + hf_delta
```

Conv không bias và `tanh` giới hạn biên độ. Năng lượng của `hf_warped` được detach,
chuẩn hóa theo sample, dilate 3x3 và ánh xạ vào `[0.25,1]` để làm activity gate thật.
Sau shared refiner, fine output đi qua activity gate và learned spatial/channel gate,
masked high-pass kernel 9, garment-support gate dự đoán từ person/DensePose, khử DC
theo channel trong support đó, rồi hard RMS gate giới hạn nó ở
`max(0.05, 10% backbone_rms)`. Đây vẫn là một refiner duy nhất: HF không có
quyền viết trực tiếp vào latent velocity.

## 9. Loss cho HF

### Source consistency

Trên paired sample, shared grid được dùng để warp garment HF và so với person HF.
Sparse weighting tăng trọng số vùng có detail thật để logo nhỏ không biến mất trong
trung bình toàn áo. Loss này giám sát `sampling_grid`/Q/K bằng raw six-channel map;
nó không đi qua HF feature encoder hay `feature_out`.

### Joint latent/decoded supervision

Clean estimate cho HF loss là:

```text
z_t.detach() + (1-t) * (
  backbone_velocity.detach() + fine_velocity
)
```

`hf_detail_loss` bị tắt vì sau fusion nó trùng số với main latent `detail_loss`, tức
là cùng edge objective bị đếm hai lần. Loss auxiliary dùng absolute RGB, spatial
contrast, opponent chroma và edge. Absolute RGB cần thiết vì zero-mean latent residual
không bảo đảm zero-mean RGB qua nonlinear VAE decoder. Nó cập nhật HF fusion/extractor và shared refiner,
nhưng không cập nhật backbone qua auxiliary objective này.

HF decoded supervision chỉ dùng sparse support bên trong `person_garment_mask`.
Decoded RGB/edge **chính** của trainer dùng toàn edit region để tái tạo cả tay/pose.
Correspondence, source consistency và garment feature loss cũng giới hạn ở garment.

## 10. Checkpoint migration

Checkpoint của revision cũ chứa `garment_high_frequency_control.output.*` bốn
channel. Graph hiện tại cần `feature_out.*` 256 channel. Loader sẽ:

- bỏ toàn bộ old HF control branch nếu thấy tensor không tương thích;
- init branch HF bias-free và reset `hf_fusion` về zero-output;
- bỏ output bias cũ và identity-init learned `fine_gate` mới;
- giữ DiT, RGB routing, refiner và pretrained/learned HF condition stem;
- yêu cầu warm-start bằng `load_weights` với optimizer mới.

Sau khi graph fused đã lưu checkpoint riêng, có thể dùng `resume_checkpoint` để
khôi phục cả optimizer và step. Không truyền hai lựa chọn cùng lúc.

## 11. Metric và cách đọc

| Metric | Câu hỏi nó trả lời |
|---|---|
| `garment_grad/refiner/hf_fusion` | projection đổi HF sang RGB basis có học không? |
| `garment_grad/hf/encoder` | HF extractor có nhận gradient không? |
| `garment_grad/hf/feature_out` | projection HF có học không? |
| `hf_feature_rms` | HF feature có authority hay vẫn zero? |
| `hf_fusion_delta_rms` | phần HF thực sự được cộng vào RGB lớn bao nhiêu? |
| `fine_velocity_rms` | joint refiner có đóng góp vào output không? |
| `fine_velocity_limit` | residual có vượt trust-region không? |
| `fine_velocity_raw_rms` | refiner trước hard gate có đang cố lấn át không? |
| `fine_velocity_learned_gate` | learned spatial gate có hoạt động không? |
| `fine_velocity_activity_gate` | HF activity prior có thực sự giới hạn spatial support? |
| `fine_velocity_effective_gate` | tích activity x learned gate có authority bao nhiêu? |
| `fine_velocity_norm_gate` | hard gate đang phải clip mạnh đến đâu? |
| `fine_velocity_dc_rms` | output còn rò DC làm lệch màu không? (phải gần zero) |
| `hf_fusion_removed_dc_rms` | HF fusion đã loại bao nhiêu offset màu? |
| `hf_source_sparse_loss` | logo/detail source có được copy đúng vị trí không? |
| decoded RGB/chroma/edge | reconstructed detail có đúng màu và cấu trúc không? |
| decoded garment RGB/low-frequency/mean | màu tuyệt đối có bị drift/dark shift không? |

Không kết luận logo đã học chỉ vì edge loss giảm. Preview held-out phải cho chữ/logo
có nghĩa và RGB/chroma loss phải cải thiện cùng edge.

## 12. Verification

`scripts/verify_vton_rgb_hf.py` kiểm tra:

- zero/empty HF không đổi output và warm start giữ prediction;
- HF output có width 256 thay vì bốn velocity channel;
- HF chỉ làm đổi shared `fine_velocity`;
- hiệu final output đúng bằng hiệu fine output;
- fine output zero-DC và không vượt 10% backbone RMS (có floor 0.05);
- auxiliary HF loss không có gradient vào backbone final head;
- một sampling grid chung cho mọi head;
- global RGB/HF mixers đều bị tắt;
- gradient đến HF fusion, encoder, feature projection và RGB refiner.
