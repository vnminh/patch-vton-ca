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

same grid + garment VAE latent 4ch ──> exact latent warp ──> zero-init projection
                                      ────────────> rgb_warped_feature bổ sung

same detached Q/K/grid + HF V ──> high-resolution warp/downsample
                                                   = hf_warped_feature

FUSION AND OUTPUT
hf_delta_raw = RGB_RMS * tanh(zero_init_conv(group_norm(hf_warped_feature)))
hf_delta = spatial_center(hf_delta_raw, edit_support)
fused_feature = rgb_warped_feature + hf_delta
fused_feature + detached backbone state ──> one shared refiner ──> raw_fine
person/DensePose query ──> supervised support head ──> predicted garment support
raw_fine ──> HF activity x learned gate ──> high-pass/support/center/RMS ──> fine_velocity
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

Sau embedder, content được tách thành hai payload. `K` dùng LayerNorm từng token
để độ sáng/vải tối không chi phối score. `V` bắt đầu đúng bằng route LayerNorm của
checkpoint cũ, rồi một scalar zero-init cho mỗi scale học trộn sang raw feature chỉ
chia global RMS. Cách này giữ scale số học ổn định nhưng không xóa mean/amplitude
riêng của từng token; logo và color-block vì thế còn thông tin để reconstruct.

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
- K là `LayerNorm(garment content)` cộng positional embedding;
- V chỉ chứa garment appearance, **không chứa positional embedding**; route V mới
  giữ magnitude bằng global-RMS thay vì LayerNorm riêng từng token;
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

### 8.2 Tạo garment key/value độc lập

- `Q = query(query_norm(person_content)) + shared_position`;
- `K = key_norm(key(per_token_LN(detail_rgb))) + shared_position`;
- `V = value(raw/global_RMS)` trong config logo;
- Q/K được normalize theo head và dùng cosine scale có giới hạn;
- V không chứa PE.

Person và garment detail đều có đúng lưới 64x48.

Q và K cộng cùng positional basis **sau** content projection. Bản cũ đưa PE qua
`query()` ở Q nhưng cộng PE sau projection ở K; vì thế cùng tọa độ không nằm trong
cùng basis. K và V cũng không dùng chung input: appearance magnitude của V không được
phép đổi score/routing của K. Config logo đặt minimum V mix bằng 1 vì audit cho thấy
learned scalar mới chỉ đạt 0.007 sau 1.6k step và gần như vẫn xóa payload màu.

### 8.3 Coarse anchor rồi local fine correspondence

Refiner không lấy trung bình softmax toàn garment:

1. pool Q/K xuống 32x24;
2. hard attention chọn coarse garment anchor;
3. lấy trung bình routing evidence của 8 head để chọn **một** coarse grid vật lý;
4. upsample displacement về 64x48;
5. tìm một local residual đồng thuận trong bán kính cấu hình quanh anchor;
6. dùng cùng grid đó bilinear-sample tám nhóm channel của V;
7. dùng chính grid đó sample trực tiếp `garment_latent` bốn kênh.

Kết quả learned V qua `attention_out` được cộng với
`latent_fusion(warped_garment_latent)`, tạo `rgb_warped_feature` kích thước
`B x 256 x 64 x 48`. `latent_fusion` là zero-init để checkpoint cũ có prediction đầu
giống hệt, nhưng sau khi mở nó cung cấp carrier trực tiếp từ latent garment đã được
chứng minh còn giữ chữ/logo qua VAE. Nhánh learned detail không còn là con đường duy
nhất có thể co thành style/màu trung bình.

Global RGB `A@V/warp_mix` bị tắt trong config logo. Nó từng trộn các vị trí garment
xa nhau; hơn nữa per-head grid cũ cho phép tám nhóm channel lấy từ tám vị trí khác
nhau, nên feature concat có edge nhưng không còn là một logo/màu coherent.

Correspondence NLL được tính trên chính logits đã trung bình head này. Trước đây loss
tối ưu tám argmax riêng nhưng forward/inference lại dùng argmax của mean logits; metric
top-1 và objective vì thế không mô tả sampling grid thực sự.

### 8.4 Curriculum chống nghiệm màu trung bình

Ở run đã đo, hard grid chỉ top-1 đúng khoảng 22–35%. Nếu refiner luôn phải decode
từ một warp sai trong 65–78% vị trí, nghiệm MSE dễ nhất là màu/style trung bình dù
loss routing vẫn giảm. Trong 2000 optimizer step đầu của một warm restart:

```text
teacher_probability = 0.75 * max(0, 1 - local_step/2000)
transport_grid = DINO_anchor_plus_mask_propagation  # sample được chọn, train-only
predicted_grid vẫn nhận correspondence/RGB/coordinate loss
```

Teacher grid chỉ thay đường transport trên garment pixel đã có confidence; nó không
thay Q/K dùng để tính routing loss. Xác suất giảm chính xác về zero, nên giai đoạn cuối
train giống inference. DINO, target person và `person_garment_mask` vẫn không phải input
inference.

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
hf_delta_raw = rms(stopgrad(rgb)) * tanh(Conv_zero(GroupNorm(hf_warped)))
hf_delta = spatial_center(hf_delta_raw, edit_support)
fused = rgb_warped_feature + hf_delta
```

Refiner nhận thêm hai state đã stop-gradient:

- `backbone_velocity`;
- `preliminary_clean = z_t + (1-t) * backbone_velocity`.

`velocity_condition` sinh modulation nhân có giới hạn cho `fused`. Local block tạo
feature decoder. HF energy đã warp tạo activity gate cố định; tensor này được detach
để HF encoder không thể tự tăng magnitude nhằm mua thêm authority. Gate có floor 0.25
cho RGB/color-block interior và mở tới 1 quanh stroke/logo. Learned gate theo từng
latent channel tiếp tục tinh chỉnh; nó dùng `2*sigmoid`, zero-init nên bắt đầu ở 1:

```text
decoded_feature = fused_modulated + Local(fused_modulated)
raw_fine = Conv_bias_free(decoded_feature)
learned_gate = 2 * sigmoid(Conv_zero(decoded_feature))
activity_gate = 0.25 + 0.75 * dilate(normalize_energy(stopgrad(hf_warped)))
garment_support = sigmoid(SupportHead(person_query_with_DensePose))
detail = masked_highpass(raw_fine * activity_gate * learned_gate, kernel=9)
fine_ac = spatial_center(detail, edit_support * stopgrad(garment_support))
hard_gate = min(1, max(0.10 * backbone_rms, 0.05) / rms(fine_ac))
fine_velocity = fine_ac * stopgrad(hard_gate)
final_velocity = backbone_velocity + fine_velocity
```

`garment_support` được train bằng paired person parse nhưng inference tự dự đoán từ
person/DensePose; target parse không được feed vào model. Nó ngăn cloth residual ghi
lên tay/nền. Masked high-pass loại trường sáng/tối rộng, còn `spatial_center` loại mean
riêng cho từng latent channel trong garment support, nên fine branch không thể tạo
DC/low-frequency offset làm cả áo đổi hue. Hard gate chạy ngay trong `forward`, cả train lẫn
inference, và bảo đảm RMS không vượt `max(0.05, 0.10 * backbone_rms)`. Trainer còn
phạt `fine_ac` trước hard gate để branch học tự giữ trong trust region, thay vì luôn
dựa vào clipping.

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
| fine garment support | full edit parse, train-only | tách vùng áo khỏi tay/nền khi infer |
| decoded RGB/edge chính | toàn edit region | reconstruction và tay/pose bị agnostic xoá |
| decoded garment RGB/low-pass/mean | garment pixels | khóa màu tuyệt đối và color-block |
| HF decoded RGB/contrast/chroma/edge | sparse garment detail | logo/màu/biên tại vùng garment |
| HF source consistency/sparse | paired garment detail | shared sampling grid copy HF đúng vị trí |
| fine velocity trust-region | edit latent | không cho refiner lấn át backbone/màu |

HF auxiliary clean prediction dùng:

```text
z_clean_hf = z_t.detach()
             + (1-t) * (stopgrad(backbone_velocity) + fine_velocity)
```

`hf_detail_loss` trùng với main `detail_loss` sau fusion nên bị tắt. Decoded HF dùng
đồng thời absolute RGB, contrast và chroma; edge có weight thấp hơn. Main decoded loss
còn chấm garment RGB, low-pass và channel mean riêng, không để tay/background pha loãng
colour objective. Do đó decoded HF loss cập nhật joint refiner và HF
encoder nhưng không kéo
backbone velocity đến local optimum của auxiliary edge objective. Flow loss trên
`final_velocity` vẫn cập nhật toàn graph theo cấu hình optimizer.

## 12. Gradient ownership

| Thành phần | Nhận gradient chính từ |
|---|---|
| DiT backbone | flow, backbone correspondence, decoded reconstruction chính |
| garment K/Q routing | correspondence, RGB/value, coordinate/smoothness, HF source consistency |
| garment RGB V/refiner | flow, fine RGB/value, joint decoded HF loss |
| direct latent fusion | flow và decoded RGB/chroma tại `t=0` |
| HF fusion/encoder/feature_out | flow và joint decoded RGB/chroma/edge |
| uncertainty head | uncertainty NLL |

`x_embedder` và garment cross-attention cùng dùng `adapter_lr_multiplier=0.1`; source
checkpoint đã học các adapter này, nên không cho chúng chạy nhanh gấp 10 backbone nữa.
Các `garment_value_mix_{scale}` zero-init nhưng dùng riêng base LR `1e-4`: prediction
đầu không đổi, còn ba scalar bounded có thể mở payload giữ magnitude đủ nhanh. Các
matrix adapter lớn vẫn ở LR thấp để không drift. `latent_fusion` cũng dùng base LR:
nó chỉ là projection 4->256 zero-init và output sau nó vẫn bị DC/RMS gate bảo vệ.

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
- bỏ `garment_refiner.output.bias`, vì bias này chính là đường tắt tạo latent DC;
- init `fine_gate` mới về identity (`2*sigmoid(0)=1`);
- init `garment_value_mix` bằng zero: K và prediction đầu giữ nguyên, V mới mở bằng loss;
- init `garment_refiner.latent_fusion` bằng zero: latent carrier mới không tạo cú nhảy output;
- config này giữ `ema_rate=0`; preview và checkpoint sampling dùng trực tiếp student,
  đồng thời tránh thêm một bản PFT-XL vào VRAM 24 GB;
- dùng `load_weights` với optimizer mới.

Chỉ dùng `resume_checkpoint` sau khi checkpoint đã được tạo bởi chính graph fused
hiện tại. Không dùng đồng thời hai tùy chọn.

## 15. Dấu hiệu model hoạt động đúng

### Invariant kiến trúc

- `x_embedder` có 13 input channel;
- person và garment detail refiner cùng grid 64x48;
- `hf_warped_feature` và bounded `hf_fusion_delta` có 256 channel, không phải 4;
- K refiner lấy normalized key content, V lấy appearance payload; hai input không alias;
- `warped_garment_latent` là `B x 4 x 64 x 48` và dùng cùng physical grid;
- tám attention head dùng cùng một `sampling_grid`;
- RGB/HF global attention đều tắt trong config logo;
- chỉ có một `fine_velocity` và một phép cộng velocity cuối;
- `fine_velocity` bị giới hạn vào predicted garment support, high-pass, có spatial
  mean bằng zero theo từng channel và RMS bị hard-limit;
- zero garment/HF hoặc empty garment mask cho residual chính xác bằng zero.

### Metric cần theo dõi cùng nhau

- routing: `fine_top1_accuracy`, `fine_rgb_loss`, `fine_warp_coordinate_loss`;
- target support: `fine_support_loss`, `fine_support_iou`;
- branch sống: `garment_grad/refiner/output`, `garment_grad/hf/encoder`,
  `garment_grad/hf/feature_out`;
- authority: `fine_velocity_raw_rms`, `fine_velocity_rms`, `fine_velocity_limit`,
  `fine_velocity_activity_gate`, `fine_velocity_learned_gate`,
  `fine_velocity_effective_gate`, `fine_velocity_norm_gate`;
- DC: `fine_velocity_dc_rms`, `fine_velocity_removed_dc_rms`,
  `hf_fusion_removed_dc_rms`;
- HF: `hf_feature_rms`, `hf_fusion_delta_rms`;
- V payload: `garment_value_mix_coarse|middle|detail`;
- curriculum/carrier: `fine_teacher_forcing_ratio`, `fine_teacher_forcing_fraction`,
  `warped_garment_latent_rms`, `garment_grad/refiner/latent_fusion`;
- reconstruction: `hf_decoded_rgb_loss`, `hf_decoded_chroma_loss`,
  `hf_decoded_contrast_loss`, `hf_decoded_edge_loss`,
  `decoded_garment_rgb_loss`, `decoded_garment_low_frequency_loss`,
  `decoded_garment_mean_loss`;
- chất lượng thật: preview paired và swapped cố định mỗi 50 optimizer step.

Edge loss giảm một mình không chứng minh logo được học. Cần thấy RGB/chroma cải thiện
và chữ/logo held-out trở nên đúng nghĩa, không chỉ sắc cạnh hơn.

Audit frozen SD-VAE trên sample cố định `00055_00.jpg` cho
`person_garment rgb/edge = 0.01248/0.01168` và in-shop garment
`0.00975/0.01008`; chữ VANS vẫn đọc rõ sau encode/decode. Vì vậy trong trường hợp
đã đo, VAE không phải trần làm mất logo. Khoảng cách validation khoảng `0.10` nằm ở
garment transport/value/refiner, phù hợp với lỗi V từng bị per-token LayerNorm.

Run `2026-09-14/T232158` đến step 2000 cho thêm bằng chứng kiến trúc: test garment
RGB MAE chỉ dao động `0.1025 -> 0.0996`, edge MAE gần như đứng yên quanh `0.0818`,
trong khi fine target mass đạt `0.49–0.67` nhưng top-1 chỉ `0.22–0.35`. Preview dựng
đúng pose/loại áo và màu tổng quát nhưng xóa stripe/chữ. Vì vậy tiếp tục riêng graph
cũ không giải quyết được bottleneck: refiner đang học từ phần lớn warp sai và learned
V là carrier duy nhất. K/V separation, direct latent carrier và curriculum ở trên xử
lý đúng ba điểm đó.

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
