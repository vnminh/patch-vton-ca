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
rgb_warped_feature + hf_warped_feature
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
RGB: shared Q/K/grid + RGB V -> rgb_warped_feature
HF : shared Q/K/grid + HF  V -> hf_warped_feature
```

Điều này bảo đảm nét HF và RGB được lấy từ cùng garment location. Auxiliary edge
loss không thể tự bẻ attention map sang nơi có biên mạnh nhưng sai logo.

## 6. High-resolution transport

`GarmentHighFrequencyControl` thực hiện:

1. `encoder`: 1x1, 256 -> 256, zero-init;
2. trừ spatial mean trên valid garment pixels;
3. local high-resolution processing ở 256x192;
4. upsample displacement grid và warp HF value tại 256x192;
5. stride-4 downsample sau warp về 64x48;
6. local block + `feature_out` tạo 256 channel;
7. edit/active gate ép vùng không hợp lệ về zero.

Warp trước rồi mới downsample giữ các phase nhỏ của stroke tốt hơn downsample HF
trước correspondence.

Spatial mean được loại chỉ ở nhánh HF. RGB refiner giữ garment mean vì đó là màu
nền thật; HF mean chủ yếu là VAE/DC bias và từng gây flat colour shift.

Global HF attention bị tắt trong config logo. Nó có xu hướng trung bình các vùng xa
nhau và tạo glyph-like texture; coherent grid là route duy nhất đang hoạt động.

## 7. Zero initialization

Zero-init nằm tại HF `encoder`, không nằm tại output head.

- Bước đầu: HF feature bằng đúng zero, checkpoint cũ giữ nguyên prediction.
- `feature_out.weight` vẫn non-zero theo standard initialization.
- Gradient từ refiner đi xuyên `feature_out/downsample` đến encoder ngay update đầu.

Nếu zero cả output weight, gradient encoder cũng bằng zero cho tới khi output head
đã thay đổi; revision cũ từng gần như không học vì vấn đề này.

Empty garment, garment dropout và unconditional CFG đều bị `active` gate ép HF
output chính xác zero, kể cả khi bias đã được train.

## 8. Fusion/refiner

Sau routing:

```text
fused = rgb_warped_feature + hf_warped_feature  # 256 x 64 x 48
fine_velocity = garment_refiner.refine(fused, detached backbone state)
final_velocity = backbone_velocity + fine_velocity
```

Đây là một refiner duy nhất. HF không có quyền viết trực tiếp vào latent velocity,
nên muốn giảm decoded RGB/chroma loss nó phải giúp refiner reconstruct appearance.

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

Loss gồm latent detail và decoded RGB/chroma/edge. Nó cập nhật HF encoder và shared
refiner, nhưng không cập nhật backbone qua auxiliary objective này.

HF decoded supervision chỉ dùng sparse support bên trong `person_garment_mask`.
Decoded RGB/edge **chính** của trainer dùng toàn edit region để tái tạo cả tay/pose.
Correspondence, source consistency và garment feature loss cũng giới hạn ở garment.

## 10. Checkpoint migration

Checkpoint của revision cũ chứa `garment_high_frequency_control.output.*` bốn
channel. Graph hiện tại cần `feature_out.*` 256 channel. Loader sẽ:

- bỏ toàn bộ old HF control branch nếu thấy tensor không tương thích;
- reset branch mới về zero-output;
- giữ DiT, RGB routing, refiner và pretrained/learned HF condition stem;
- yêu cầu warm-start bằng `load_weights` với optimizer mới.

Sau khi graph fused đã lưu checkpoint riêng, có thể dùng `resume_checkpoint` để
khôi phục cả optimizer và step. Không truyền hai lựa chọn cùng lúc.

## 11. Metric và cách đọc

| Metric | Câu hỏi nó trả lời |
|---|---|
| `garment_grad/hf/encoder` | zero-init encoder có nhận gradient không? |
| `garment_grad/hf/feature_out` | projection HF có học không? |
| `hf_feature_rms` | HF feature có authority hay vẫn zero? |
| `fine_velocity_rms` | joint refiner có đóng góp vào output không? |
| `hf_source_sparse_loss` | logo/detail source có được copy đúng vị trí không? |
| decoded RGB/chroma/edge | reconstructed detail có đúng màu và cấu trúc không? |

Không kết luận logo đã học chỉ vì edge loss giảm. Preview held-out phải cho chữ/logo
có nghĩa và RGB/chroma loss phải cải thiện cùng edge.

## 12. Verification

`scripts/verify_vton_rgb_hf.py` kiểm tra:

- zero/empty HF không đổi output;
- HF output có width 256 thay vì bốn velocity channel;
- HF chỉ làm đổi shared `fine_velocity`;
- hiệu final output đúng bằng hiệu fine output;
- auxiliary HF loss không có gradient vào backbone final head;
- gradient đến HF encoder, HF feature projection và RGB refiner.
