# Kiến trúc VTON-PFT hiện tại

Tài liệu này là **nguồn sự thật** cho kiến trúc đang dùng bởi:

```text
experiment=viton-pft-xl-512x384-detail-logo-hf
```

Mục tiêu của model là giữ pose/identity của người, thay đúng vùng quần áo và tái
tạo được màu khối, logo, chữ nhỏ từ ảnh garment. Model kết hợp PFT-XL ở lưới token
32x24 với một refiner garment ở lưới latent đầy đủ 64x48.

## 1. Mô hình trong một câu

DiT tạo cấu trúc người và `backbone_velocity`; garment RGB/VAE được warp vào người,
HF của garment được warp bằng cùng correspondence rồi đổi sang RGB feature basis qua
một projection zero-init có giới hạn; refiner duy nhất sinh `fine_velocity`.

```text
final_velocity = backbone_velocity + fine_velocity
```

Không có `hf_velocity` độc lập và HF không phải một channel của `x_embedder`.

## 2. Sơ đồ end-to-end

```text
PERSON PATH
person image + agnostic mask ──> person_agnostic ──> frozen VAE ──> z_agnostic
dense pose RGB ─────────────────────────> frozen VAE ────────────> z_densepose
target person (train only) ──────────────> frozen VAE ───────────> z_target
noise z0 + z_target + token time ────────────────────────────────> z_t

GARMENT RGB PATH
in-shop garment + garment mask ──> frozen VAE pyramid
                                  ├─ coarse latent
                                  ├─ middle feature
                                  └─ detail feature ─────────────> garment RGB K/V

GARMENT HF PATH
in-shop garment + garment mask ──> signed RGB DoG + colour gradients (6 channels)
                                  ──> pretrained/learnable HF VAE stem
                                  ──> blank-baseline subtraction
                                  ───────────────────────────────> garment HF V

BACKBONE
concat[z_t, z_agnostic, edit mask, z_densepose] = 13 channels
        ── patchify 2x2 ──> 32x24 person tokens
        ── 28 DiT blocks + multiscale garment cross-attention
        ├────────────────────────────────────────────────────────> backbone_velocity
        └────────────────────────────────────────────────────────> person feature x

FULL-LATENT GARMENT ROUTE
person feature x + current latent state ─────────────────────────> Q
garment detail RGB feature + position ───────────────────────────> K
garment detail RGB feature, no position ─────────────────────────> RGB V
Q/K ──> coarse hard anchor + local fine grid ──> warp RGB V
                                                   = rgb_warped_feature

same detached Q/K/grid + HF V ──> high-resolution warp/downsample
                                                   = hf_warped_feature

FUSION AND OUTPUT
hf_delta = RGB_RMS * tanh(zero_init_conv(group_norm(hf_warped_feature)))
fused_feature = rgb_warped_feature + hf_delta
fused_feature + detached backbone state ──> one shared refiner ──> fine_velocity
final_velocity = backbone_velocity + fine_velocity
```

## 3. Kích thước chuẩn

| Đại lượng | Shape chuẩn | Ý nghĩa |
|---|---:|---|
| Ảnh người/garment/HF | `B x C x 512 x 384` | C=3 cho RGB, C=6 cho HF |
| VAE latent | `B x 4 x 64 x 48` | stride ảnh/latent = 8 |
| PFT patch | `2 x 2` latent | một token quản lý bốn latent cell |
| Person token grid | `32 x 24 = 768` | input/output token của DiT |
| DiT hidden | `1152` | PFT-XL, 28 block, 16 head |
| Fine refiner grid | `64 x 48 = 3072` | một output trên mỗi latent cell |
| Refiner width | `256` | 8 attention head |
| HF VAE feature | `B x 256 x 256 x 192` | hai stream 128 kênh được concat |

Xem bảng đầy đủ tại [`MODEL_FLOW_VI.md`](MODEL_FLOW_VI.md).

## 4. Khối dữ liệu và mask

### 4.1 Input có sẵn ở cả train và inference

- `person`: ảnh người ban đầu;
- `person_agnostic`: ảnh người sau khi xoá garment cũ;
- `agnostic_mask`: vùng model được phép chỉnh;
- `dense_pose`: điều kiện hình học cơ thể;
- `garment`: ảnh sản phẩm in-shop;
- `garment_mask`: foreground của ảnh sản phẩm.

HF inference chỉ được trích từ `garment + garment_mask`. Model không dùng crop
garment ground-truth trên người để suy luận unpaired.

### 4.2 Mask có ba trách nhiệm khác nhau

| Mask | Đi vào model? | Dùng cho |
|---|---|---|
| `agnostic_mask` | Có | vùng chỉnh, token time, hard update, decoded reconstruction |
| `garment_mask` | Có | bỏ key background của ảnh garment, active gate |
| `person_garment_mask` | Không | supervision train-only cho correspondence/feature garment |

`person_garment_mask` được lấy từ parse label quần áo của người và nhận đúng cùng
flip/shift/scale với ảnh người. Nó không được cấp cho sampler, nên không gây leak
ở inference.

Correspondence, RGB transport và garment feature loss chỉ chấm trên garment pixel.
Decoded reconstruction chấm toàn bộ edit region, gồm tay/bàn tay bị agnostic xoá.

## 5. Khối encoder

### 5.1 Frozen SD-VAE cho người

- `target image -> z_target`: target flow khi train;
- `person_agnostic -> z_agnostic`: context giữ identity/background;
- `dense_pose -> z_densepose`: 4 channel hình học đưa trực tiếp vào DiT và fine Q.

Các VAE parameter này frozen. `person_context = z_agnostic * (1-mask_latent)`.

### 5.2 VAE pyramid cho garment RGB

Garment được encode một lần thành ba cấp:

- `coarse`: latent 4 kênh, giữ bố cục/màu tổng quát;
- `middle`: tap VAE 1/4 resolution, giữ cấu trúc trung gian;
- `detail`: tap VAE 1/2 resolution, giữ logo, seam và texture.

Ba embedder đưa chúng về hidden 1152 cho cross-attention backbone. Ở backbone,
các route được match về person token grid 32x24. Bản detail chưa pool được giữ
riêng và embed thành lưới 64x48 cho latent refiner.

### 5.3 Encoder garment HF

Dataset tạo sáu channel sau mọi augmentation của garment:

- channel `0:3`: signed RGB high-pass/DoG đa tỉ lệ;
- channel `3:6`: gradient magnitude của luma, opponent chroma và max RGB.

Hai nhóm ba channel đi qua hai lượt của HF VAE stem. Response của ảnh blank tương
ứng được trừ đi trước khi concat, tránh VAE bias biến “không có detail” thành một
feature khác zero. HF stem được khởi tạo từ pretrained VAE và có LR nhỏ hơn.

## 6. Khối flow/interpolant

VTON dùng convention:

```text
z_t = t * z_target + (1 - t) * z_noise
u_target = z_target - z_noise
```

Mỗi token 32x24 có timestep riêng. Ngoài edit mask, timestep bị đặt về trạng thái
context và latent luôn lấy từ `person_context`. Training trộn pure-noise garment
forcing, high-time refinement và lịch timestep thông thường để model học cả dựng
hình ban đầu lẫn hoàn thiện detail.

## 7. Input và backbone DiT

### 7.1 Input 13 channel

```text
z_t                 4
z_agnostic          4
edit/person mask    1
z_densepose         4
---------------------
total              13
```

Patch embedder dùng kernel/stride 2 trên latent 64x48, tạo 768 token 1152 chiều.
Timestep embedding và class/dropout embedding tạo điều kiện AdaLN cho mỗi token.

### 7.2 Một DiT block

Mỗi block gồm:

1. self-attention trên person token;
2. garment cross-attention nếu block có route;
3. MLP;
4. AdaLN modulation từ timestep/class condition.

Garment được inject mỗi hai block theo route coarse/middle/detail. Trong
cross-attention:

- Q đến từ person token;
- K là garment content cộng positional embedding;
- V chỉ chứa garment content, **không chứa positional embedding**;
- garment padding mask loại background key;
- edit token mask giới hạn nơi garment residual được viết.

PE ở K giúp định vị để match; không đặt PE trong V tránh copy tọa độ như appearance.

### 7.3 Backbone output

Sau block 28, `final_layer + unpatchify` trả:

- `backbone_velocity`: `B x 4 x 64 x 48`;
- `logvar`: `B x 1 x 64 x 48` cho uncertainty.

Feature token cuối `x` đồng thời được đưa sang refiner để tạo person Q. Vì vậy:
refiner chạy **sau** DiT, nhưng `rgb_warped_feature` không phải output thô của DiT.

## 8. GarmentLatentRefiner: RGB coherent warp

### 8.1 Tạo person query 64x48

`query_expand` biến mỗi token 1152 thành `4 x 256`, rồi PixelShuffle tách bốn
latent cell trong mỗi PFT patch. Query này được cộng với convolution state của:

```text
[z_t, z_agnostic, z_densepose]
```

Nhờ đó bốn subquery khác nhau theo nội dung/pose thực tế, không chỉ theo absolute PE.

### 8.2 Tạo garment key/value

- `K = key_norm(key(detail_rgb)) + position`;
- `V = value(detail_rgb)`;
- Q/K được normalize theo head và dùng cosine scale có giới hạn;
- V không chứa PE.

Person và garment detail đều có đúng lưới 64x48.

### 8.3 Coarse anchor rồi local fine correspondence

Refiner không lấy trung bình softmax toàn garment:

1. pool Q/K xuống 32x24;
2. hard attention chọn coarse garment anchor;
3. lấy trung bình routing evidence của 8 head để chọn **một** coarse grid vật lý;
4. upsample displacement về 64x48;
5. tìm một local residual đồng thuận trong bán kính cấu hình quanh anchor;
6. dùng cùng grid đó bilinear-sample tám nhóm channel của V.

Kết quả qua `attention_out` là `rgb_warped_feature` kích thước
`B x 256 x 64 x 48`.

Global RGB `A@V/warp_mix` bị tắt trong config logo. Nó từng trộn các vị trí garment
xa nhau; hơn nữa per-head grid cũ cho phép tám nhóm channel lấy từ tám vị trí khác
nhau, nên feature concat có edge nhưng không còn là một logo/màu coherent.

## 9. GarmentHighFrequencyControl

HF dùng lại `Q`, `K`, `key_valid` và `sampling_grid` của refiner RGB. Q/K/grid được
detach trên đường HF để HF không tự bẻ correspondence nhằm giảm edge loss.

HF control:

1. 1x1 encoder học được ánh xạ 256 HF channel sang width 256;
2. trừ spatial mean chỉ trên valid garment keys;
3. warp feature ở 256x192 bằng displacement grid đã upsample;
4. downsample sau warp về 64x48 để giữ sub-latent phase;
5. local processing + `feature_out` bias-free tạo `hf_warped_feature` 256 channel;
6. trừ spatial DC lần cuối trong edit support để HF không đổi màu trung bình.

Global HF attention bị tắt trong config logo vì đường global dễ trung bình hoá các
vùng garment xa nhau. Xem [`VTON_HF_CONTROL.md`](VTON_HF_CONTROL.md).

## 10. Fusion và fine velocity

HF và RGB có cùng width nhưng không mặc nhiên cùng feature basis. Vì vậy không cộng
raw trực tiếp. Fusion chuẩn hoá HF, đổi basis bằng convolution zero-init và giới hạn
biên độ theo RMS của RGB:

```text
hf_delta = rms(stopgrad(rgb)) * tanh(Conv_zero(GroupNorm(hf_warped)))
fused = rgb_warped_feature + hf_delta
```

Refiner nhận thêm hai state đã stop-gradient:

- `backbone_velocity`;
- `preliminary_clean = z_t + (1-t) * backbone_velocity`.

`velocity_condition` sinh modulation nhân có giới hạn cho `fused`. Local block và
output convolution sau đó tạo **một** `fine_velocity` bốn channel:

```text
fine_velocity = Refiner(fused, stopgrad(backbone state))
final_velocity = backbone_velocity + fine_velocity
```

Trainer thêm trust-region mềm: RMS của `fine_velocity` không được tăng tự do quá
`max(0.1, 0.5 * backbone_velocity_rms)`. Đây là guard chống residual chiếm quyền
backbone như run lỗi (fine RMS tăng khoảng 67 lần).

Không có person-only additive shortcut qua refiner và không có HF velocity head.
Garment-transported feature luôn là carrier của residual detail.

## 11. Loss và khối nào được học

| Loss | Vùng chấm | Mục tiêu chính |
|---|---|---|
| flow MSE | edit latent | toàn bộ final velocity |
| outside velocity | ngoài edit | không làm đổi vùng giữ nguyên |
| uncertainty NLL | edit latent | log-variance/difficulty |
| backbone correspondence | garment token | multiscale Q/K routing |
| fine correspondence/coordinate | garment pixel/token | coarse anchor + local grid |
| fine RGB/value | garment pixel | warped RGB/VAE V phải đúng content |
| warp smoothness/mask | garment region | flow mượt và không lấy background |
| decoded RGB/edge chính | toàn edit region | reconstruction và tay/pose bị agnostic xoá |
| HF decoded RGB/chroma/edge | sparse garment detail | logo/màu/biên tại vùng garment |
| HF source consistency/sparse | paired garment detail | shared sampling grid copy HF đúng vị trí |
| fine velocity trust-region | edit latent | không cho refiner lấn át backbone/màu |

HF auxiliary clean prediction dùng:

```text
z_clean_hf = z_t.detach()
             + (1-t) * (stopgrad(backbone_velocity) + fine_velocity)
```

`hf_detail_loss` trùng với main `detail_loss` sau fusion nên bị tắt. Decoded
RGB/chroma được ưu tiên hơn edge. Do đó decoded HF loss cập nhật joint refiner và HF
encoder nhưng không kéo
backbone velocity đến local optimum của auxiliary edge objective. Flow loss trên
`final_velocity` vẫn cập nhật toàn graph theo cấu hình optimizer.

## 12. Gradient ownership

| Thành phần | Nhận gradient chính từ |
|---|---|
| DiT backbone | flow, backbone correspondence, decoded reconstruction chính |
| garment K/Q routing | correspondence, RGB/value, coordinate/smoothness, HF source consistency |
| garment RGB V/refiner | flow, fine RGB/value, joint decoded HF loss |
| HF fusion/encoder/feature_out | flow và joint decoded RGB/chroma/edge |
| uncertainty head | uncertainty NLL |

HF Q/K được detach chỉ trên HF-value path; Q/K vẫn học bình thường từ các loss
correspondence/RGB và raw-HF source consistency của shared sampling grid.

## 13. Train và inference khác nhau ở đâu

### Training paired

Có `z_target`, `person_garment_mask`, DINO teacher target và `person_high_frequency`
để tính loss. Đây là supervision, không phải model condition.

### Inference unpaired

Chỉ dùng person agnostic, DensePose, garment RGB, garment mask và HF trích từ chính
garment. CFG nhân đôi batch; nửa unconditional nhận garment RGB/HF/mask bằng zero.
Sampler tích phân velocity chỉ trong edit mask và compose vùng ngoài từ người gốc.

## 14. Checkpoint và zero initialization

Khi đổi từ kiến trúc `backbone + detail_velocity + hf_velocity` sang fused feature:

- giữ DiT, garment RGB routing, refiner và HF condition stem tương thích;
- bỏ toàn bộ tensor HF control cũ có head 4 channel;
- init lại HF extractor bias-free, zero-init `hf_fusion` nên output bước đầu không đổi;
- dùng `load_weights` với optimizer mới.

Chỉ dùng `resume_checkpoint` sau khi checkpoint đã được tạo bởi chính graph fused
hiện tại. Không dùng đồng thời hai tùy chọn.

## 15. Dấu hiệu model hoạt động đúng

### Invariant kiến trúc

- `x_embedder` có 13 input channel;
- person và garment detail refiner cùng grid 64x48;
- `hf_warped_feature` và bounded `hf_fusion_delta` có 256 channel, không phải 4;
- tám attention head dùng cùng một `sampling_grid`;
- RGB/HF global attention đều tắt trong config logo;
- chỉ có một `fine_velocity` và một phép cộng velocity cuối;
- zero garment/HF hoặc empty garment mask cho residual chính xác bằng zero.

### Metric cần theo dõi cùng nhau

- routing: `fine_top1_accuracy`, `fine_rgb_loss`, `fine_warp_coordinate_loss`;
- branch sống: `garment_grad/refiner/output`, `garment_grad/hf/encoder`,
  `garment_grad/hf/feature_out`;
- authority: `fine_velocity_rms`, `fine_velocity_limit`, `hf_feature_rms`,
  `hf_fusion_delta_rms`;
- reconstruction: `hf_decoded_rgb_loss`, `hf_decoded_chroma_loss`,
  `hf_decoded_edge_loss`;
- chất lượng thật: preview paired và swapped cố định mỗi 50 optimizer step.

Edge loss giảm một mình không chứng minh logo được học. Cần thấy RGB/chroma cải thiện
và chữ/logo held-out trở nên đúng nghĩa, không chỉ sắc cạnh hơn.

## 16. Bản đồ code

| Khối | File / symbol |
|---|---|
| Dataset, mask, HF map | `patch_flow/vton_data.py::VTONHDDataset` |
| VAE/HF encode, loss | `patch_flow/trainer_vton.py::LatentVTONPatchForcingTrainer` |
| Flow/interpolant/CFG | `patch_flow/flow_vton.py::VTONPatchFlowForcing` |
| Mask token/latent | `patch_flow/vton_utils.py::prepare_vton_masks` |
| DiT backbone | `patch_flow/models/pf_transformer_vton.py::VTONPatchForcingDiT` |
| Fine RGB route/refiner | `...::GarmentLatentRefiner` |
| HF feature transport | `...::GarmentHighFrequencyControl` |
| Current experiment | `configs/experiment/viton-pft-xl-512x384-detail-logo-hf.yaml` |

## 17. Lệnh warm-start hiện tại

```bash
cd /workspace/vton-fix
source /venv/ai/bin/activate
export VITONHD_ROOT=/workspace/high-resolution-viton-zalando-dataset
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
python train.py \
  experiment=viton-pft-xl-512x384-detail-logo-hf \
  load_weights=/workspace/vton-fix/logs/vton/pft-xl-512x384-detail-logo-hf/2026-09-12/T174907/checkpoints/step001000.ckpt
```

Không chạy lệnh này song song với một process train khác trên cùng GPU 24 GB.
