# Tensor flow và shape reference

Đây là bảng tra cứu shape cho kiến trúc hiện tại. Đọc mô hình tổng thể tại
[`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md) trước khi dùng tài liệu này.

Config tham chiếu:

```text
viton-pft-xl-512x384-detail-logo-hf
```

Ký hiệu: `B` là microbatch; config hiện tại dùng `B=1`, accumulation 32.

## 1. Dataset output

| Key | Shape | Range / vai trò |
|---|---:|---|
| `image`, `person` | `B x 3 x 512 x 384` | `[-1,1]`, target paired khi train |
| `person_agnostic` | `B x 3 x 512 x 384` | người đã xoá vùng edit |
| `garment` | `B x 3 x 512 x 384` | ảnh garment in-shop |
| `agnostic_mask` | `B x 1 x 512 x 384` | vùng được chỉnh |
| `garment_mask` | `B x 1 x 512 x 384` | foreground của garment in-shop |
| `person_garment_mask` | `B x 1 x 512 x 384` | mask supervision, không feed model |
| `dense_pose` | `B x 3 x 512 x 384` | DensePose RGB đã transform cùng person |
| `garment_high_frequency` | `B x 6 x 512 x 384` | signed RGB DoG + gradient |
| `person_high_frequency` | `B x 6 x 512 x 384` | paired target cho source-consistency |

`person_high_frequency` và `person_garment_mask` chỉ phục vụ loss khi paired.

## 2. Sau encoder

| Tensor | Shape | Nguồn |
|---|---:|---|
| `target` | `B x 4 x 64 x 48` | frozen VAE của `image` |
| `person_context` | `B x 4 x 64 x 48` | VAE agnostic, zero trong edit mask |
| `dense_pose` | `B x 4 x 64 x 48` | frozen VAE của DensePose |
| `garment_latent` | `B x 4 x 64 x 48` | coarse garment VAE |
| `garment_middle` | `B x 256 x 128 x 96` | tap VAE 1/4 |
| `garment_detail` | `B x 128 x 256 x 192` | tap VAE 1/2 |
| `garment_high_frequency` | `B x 256 x 256 x 192` | hai HF VAE stream 128 kênh |

HF response blank được trừ trước concat, nên map detail rỗng có condition zero.

## 3. Mask sau resize

| Mask | Shape | Cách tạo |
|---|---:|---|
| `masks.token` | `B x 768` | edit coverage trên lưới 32x24 |
| `masks.latent` | `B x 1 x 64 x 48` | hard generated/update region |
| `masks.condition` | `B x 1 x 64 x 48` | soft/edit conditioning mask |

## 4. Rectified-flow tensor

```text
t                 B x 768
t_latent          B x 1 x 64 x 48
z_noise           B x 4 x 64 x 48
z_t               B x 4 x 64 x 48
u_target           B x 4 x 64 x 48
```

```text
z_t = t_latent * target + (1-t_latent) * noise
u_target = target - noise
```

Ngoài `masks.latent`, `z_t` được thay bằng `person_context`.

## 5. DiT input và token

```text
concat input:
  z_t              B x 4 x 64 x 48
  person_context   B x 4 x 64 x 48
  person mask      B x 1 x 64 x 48
  dense pose       B x 4 x 64 x 48
                   ----------------
                   B x 13 x 64 x 48

PatchEmbed(kernel=2,stride=2)
  x                B x 768 x 1152
  position         1 x 768 x 1152
  cond             B x 768 x 1152
```

## 6. Garment branch trong backbone

Trước `garment_match_query_grid`:

| Scale | Source | Embedded grid | Hidden |
|---|---|---:|---:|
| coarse | `4 x 64 x 48` | `32 x 24` | 1152 |
| middle | `256 x 128 x 96` | theo embedder, sau đó pool | 1152 |
| detail | `128 x 256 x 192` | `64 x 48`, sau đó pool | 1152 |

Trong 28 backbone block, cả ba route dùng:

```text
Q_person       B x 768 x 1152
K_garment      B x 768 x 1152 = content + PE
V_garment      B x 768 x 1152 = content only
attention      16 heads
```

`fine_values` giữ bản detail **trước pool**:

```text
B x 3072 x 1152  <=>  64 x 48 garment grid
```

## 7. Backbone output

```text
x_after_block_28     B x 768 x 1152
final token output   B x 768 x 20
unpatchify           B x 5 x 64 x 48
  backbone_velocity  B x 4 x 64 x 48
  logvar             B x 1 x 64 x 48
```

## 8. Fine RGB route

```text
query_expand(x)              B x (256*4) x 32 x 24
pixel_shuffle(2)             B x 256 x 64 x 48
state[z_t,agnostic,dense]    B x 256 x 64 x 48

Q                            B x 8 x 3072 x 32
K_detail                     B x 8 x 3072 x 32
V_detail                     B x 8 x 3072 x 32
```

Coherent grid:

```text
coarse Q/K                   B x 8 x 768 x 32
coarse hard grid             B x 8 x 768 x 2
upsampled base grid          B x 8 x 3072 x 2
local fine sampling grid     B x 8 x 3072 x 2
warped RGB values            B x 8 x 3072 x 32
rgb_warped_feature           B x 256 x 64 x 48
```

## 9. HF route

HF dùng cùng `Q/K/key_valid/sampling_grid` nhưng V riêng:

```text
HF input                     B x 256 x 256 x 192
1x1 zero-init encoder        B x 256 x 256 x 192
high-resolution warp         B x 256 x 256 x 192
stride-4 downsample          B x 256 x 64 x 48
local + feature_out          B x 256 x 64 x 48
                            = hf_warped_feature
```

Đầu ra HF là 256 feature channel, không phải velocity 4 channel.

## 10. Fusion/refiner output

```text
fused_feature = rgb_warped_feature + hf_warped_feature
              = B x 256 x 64 x 48

velocity_condition(
  backbone_velocity,
  preliminary_clean
)             = B x 256 x 64 x 48

fine_velocity = B x 4 x 64 x 48
final_velocity= B x 4 x 64 x 48
```

```text
preliminary_clean = z_t + (1-t) * backbone_velocity
final_velocity = backbone_velocity + fine_velocity
```

## 11. Clean prediction dùng cho loss

Flow/main detail:

```text
predicted_clean = z_t + (1-t) * final_velocity
```

Joint HF/refiner auxiliary:

```text
fused_predicted_clean = z_t.detach()
  + (1-t) * (backbone_velocity.detach() + fine_velocity)
```

Sau frozen VAE decoder, supervision image trở lại `B x 3 x 512 x 384` hoặc
resolution phụ được cấu hình riêng để giảm memory.

## 12. Inference

Mỗi sampler step:

```text
velocity = model(z_t, token_t, person/garment conditions)
z_next = z_t + (t_next-t_current) * velocity * masks.latent
z_next = z_next * masks.latent + person_context * (1-masks.latent)
```

Với CFG, conditional và unconditional có cùng person/DensePose; nửa unconditional
nhận garment RGB, garment mask và HF bằng zero.
