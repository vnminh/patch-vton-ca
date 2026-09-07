# Luồng dữ liệu chi tiết của mô hình VTON-PFT

Tài liệu mô tả **shape thực tế** của từng tensor đi qua mô hình, lấy theo cấu hình
đang chạy: `configs/experiment/viton-pft-xl-512x384-detail.yaml`
(kế thừa `...-garment-fix` → `...-512x384`).

Ký hiệu chung:

| Tên | Giá trị | Nguồn |
|---|---|---|
| `B` | batch (1/GPU, `accumulate_grad_batches: 32`) | `configs/experiment/viton-pft-xl-512x384-detail.yaml` |
| Ảnh | `512 x 384` (H x W, letterbox) | `configs/data/vitonhd512x384-garment.yaml` |
| Latent | `64 x 48`, 4 kênh (VAE stride 8) | `sd_ae` |
| `p` (patch) | 2 | `configs/model/vton-pft-xl.yaml` |
| Lưới token | `32 x 24` = **768 token** | latent / p |
| `D` (hidden) | 1152, `depth` 28, `heads` 16 | PFT-XL |

---

## 1. Dataset → batch

`patch_flow/vton_data.py::VTONHDDataset.__getitem__`

Người và áo được letterbox độc lập, rồi **shift/scale ngẫu nhiên riêng biệt**
(kiểu StableVITON) để correspondence không suy biến thành toạ độ tuyệt đối.

| Khoá | Shape | Miền giá trị |
|---|---|---|
| `image` / `person` | `(B,3,512,384)` | `[-1,1]` |
| `person_agnostic` | `(B,3,512,384)` | `person * (1 - agnostic_mask)` |
| `garment` | `(B,3,512,384)` | `[-1,1]` |
| `agnostic_mask` | `(B,1,512,384)` | `{0,1}` — vùng cần sinh |
| `garment_mask` | `(B,1,512,384)` | `{0,1}` — pixel áo hợp lệ |
| `person_garment_mask` | `(B,1,512,384)` | parse label `[5,6,7]`, **chỉ dùng cho loss** |
| `has_ground_truth` | `(B,)` bool | `person_name == garment_name` |

---

## 2. Encode VAE

`patch_flow/trainer_vton.py::_encode_batch` + `patch_flow/vae_features.py::encode_vae_pyramid`

### 2.1 Nhánh người

```
image           (B,3,512,384) --VAE.encode--> target          (B,4,64,48)
person_agnostic (B,3,512,384) --VAE.encode--> agnostic_latent (B,4,64,48)
person_context = agnostic_latent * (1 - masks.latent)          (B,4,64,48)
```

### 2.2 Kim tự tháp VAE của áo (`encode_vae_pyramid`)

Tap trực tiếp vào `encoder.down[*]` của SD-VAE, **một lần forward**:

```
garment (B,3,512,384)
  ├── sau down[0]  → detail  (B,128,256,192)   # 1/2 độ phân giải
  ├── sau down[1]  → middle  (B,256,128, 96)   # 1/4 độ phân giải
  └── mid + conv_out + quant_conv
                   → garment_latent (B,4,64,48)  # đã (x + shift) * scale
```

### 2.3 Ba loại mask (`patch_flow/vton_utils.py::prepare_vton_masks`)

```
agnostic_mask (B,1,512,384)
  ├── masks.token     (B,768)      bool   — adaptive_max_pool → 32x24, token nào có pixel mask
  ├── masks.latent    (B,1,64,48)  float  — token nở lại lưới latent (repeat_interleave)
  └── masks.condition (B,1,64,48)  float  — area-pool, làm mềm biên bằng avg_pool 3x3
```

Không dilate quá lưới token: nới rộng sẽ buộc mô hình tổng hợp lại bằng chứng
danh tính (hàm, cổ, tóc) mà lẽ ra chỉ cần copy.

---

## 3. Flow / interpolant

`patch_flow/flow_vton.py::VTONPatchFlowForcing.get_interpolants`

### 3.1 Lấy mẫu timestep **theo từng token**

```
t ~ LogitNormalTruncatedGaussian(loc=0.5, scale=1.0, std=0.25)   (B,768)
  ├── 35%  batch → t = 0 toàn bộ  (garment forcing: nhiễu thuần, buộc đọc áo)
  ├── 25%  batch → t = 1 - |N(0,1)|*0.25  (chế độ tinh chỉnh chi tiết t≈1)
  └── 40%  batch → giữ nguyên
```

### 3.2 Nội suy

```
t_effective = where(masks.token, t, 1)                (B,768)
t_latent    = t_effective → (B,1,32,24) → repeat 2x2  (B,1,64,48)

interpolated = t_latent * x1 + (1 - t_latent) * x0     x0 ~ N(0,I)
xt = masks.latent * interpolated + (1 - masks.latent) * person_context   (B,4,64,48)
ut = x1 - x0                                                            (B,4,64,48)
```

Ngoài vùng edit: `t = 1`, latent = pixel người thật → mô hình chỉ học sinh trong mask.

### 3.3 Garment dropout (CFG)

`_drop_garment`: 10% mẫu bị **zero hoá cả 3 nhánh áo + garment_mask**, và cờ `keep`
được trả về để mọi loss giám sát áo bỏ qua mẫu đó.

---

## 4. Backbone: `VTONPatchForcingDiT.forward`

`patch_flow/models/pf_transformer_vton.py:494`

### 4.1 Nhánh truy vấn (person query)

```
x               (B,4,64,48)   noisy latent
person_agnostic (B,4,64,48)   → interpolate về (64,48)
person_mask     (B,1,64,48)   = masks.condition
concat          (B,9,64,48)                       # 4 state + 4 agnostic + 1 mask
  → PatchEmbed(k=2,s=2, 9→1152)  → (B,768,1152)
  → + pos_embed                                     (1,768,1152)  bicubic từ 32x32 gốc
```

`x_embedder.proj.weight`: 5 kênh mới init **0**, 4 kênh đầu copy từ PFT pretrained →
khởi động bằng đúng hành vi backbone gốc.

### 4.2 Điều kiện `cond`

```
t (B,768) → t_embedder(t[...,None]) → (B,768,1152)
y (B,)    → y_embedder              → (B,1,1152)  broadcast
cond = t_emb + y_emb                              (B,768,1152)
```

### 4.3 Ba nhánh key/value của áo (`_garment_branches`)

| Nhánh | Nguồn | Embedder | Lưới gốc | Token |
|---|---|---|---|---|
| `coarse` | `garment_latent (B,4,64,48)` | `PatchEmbed(k=2,s=2, 4→1152)` (copy từ pretrained) | `32x24` | 768 |
| `middle` | `(B,256,128,96)` | `Conv2d(256→1152, k=4, s=4)` | `32x24` | 768 |
| `detail` | `(B,128,256,192)` | `Conv2d(128→1152, k=4, s=4)` | `64x48` | 3072 |

Mỗi nhánh:

```
values[s] = LayerNorm_s(embedded)      (B,N_s,1152)   # nội dung thuần, dùng cho V
keys[s]   = values[s] + pos_embed_s    (B,N_s,1152)   # vị trí chỉ nằm ở K
```

> LayerNorm là bắt buộc: đo tại 512x384, độ lớn token thô là 25 (coarse) / 143
> (middle) / 68 (detail) so với query đã LayerNorm ~34 → vải tối tự động sinh key lớn hơn.

### 4.4 `garment_match_query_grid: true` (chỉ có ở config *detail*)

```
detail:  (B,3072,1152) --adaptive_avg_pool2d(32,24)--> (B,768,1152)
keys/values/grids của MỌI scale ≡ lưới query 32x24
```

⚠️ Biến cục bộ `fine_values` được **giữ trước khi pool**, nên refiner vẫn nhận
`detail` nguyên bản `(B,3072,1152)`.

### 4.5 Padding mask theo áo

`_garment_padding_masks`: `garment_mask` → max_pool về từng lưới → `(B,N_s)` bool.
Mẫu bị dropout (mask rỗng) được ép `keep[:,0]=True` để SDPA không ra NaN.

### 4.6 Vòng 28 block

`cross_attention_every: 2` → 14 block có cross-attn (block 2,4,…,28).
`garment_scale_routes` = `[coarse, middle, detail] x4 + [detail, detail]`
→ **4 coarse / 4 middle / 6 detail**.

Mỗi `VTONPatchForcingBlock`:

```
shift,scale,gate (x6) = adaLN_modulation(cond)          (B,768,1152) mỗi cái

x = x + gate_msa * SelfAttn(modulate(LN1(x)))           (B,768,1152)

# chỉ block có route:
cross, A = MultiheadAttention(
      query = garment_norm(x)          (B,768,1152)
      key   = keys[s]                  (B,N_s,1152)
      value = values[s]                (B,N_s,1152)     # KHÔNG cộng pos
      key_padding_mask                 (B,N_s)
      average_attn_weights=False)      → cross (B,768,1152), A (B,16,768,N_s)
cross = cross * edit_token_mask[...,None]                # chỉ ghi vào vùng edit
x = x + cross                                            # residual không gate

x = x + gate_mlp * MLP(modulate(LN2(x)))                 (B,768,1152)
```

`out_proj` init `std = 1.3e-2` (~0.45x init chuẩn) — đủ nhỏ để không phá backbone,
đủ lớn để Q/K/V học được ngay từ bước đầu.

`gradient_checkpointing: true` → mỗi block bọc `torch.utils.checkpoint`.

Khi `return_garment_attention=True`, mỗi block route trả về dict:

```python
{"block": i, "scale": s, "weights": A (B,16,768,N_s),
 "output": cross (B,768,1152), "grid": (H_s,W_s),
 "query_grid": (32,24), "key_padding": (B,N_s)}
```

### 4.7 `GarmentLatentRefiner` (nhánh tinh, `garment_latent_refiner: true`)

`pf_transformer_vton.py:14` — chạy **sau** 28 block, ở **độ phân giải latent đầy đủ 64x48**
(mỗi ô = 1 vùng ảnh 8x8), `width=256`, `heads=8`.

```
tokens x       (B,768,1152) → Linear(1152 → 256*4) → (B,1024,32,24)
                            → pixel_shuffle(2)      → (B,256,64,48)   # giữ 4 pha subpixel
cond           (B,768,1152) → Linear(1152→256) → (B,256,32,24) → interp nearest → (B,256,64,48)
[noisy; agnostic] (B,8,64,48) → Conv2d(8→256,3x3)                    → (B,256,64,48)
query = tổng 3 thành phần → flatten                                   (B,3072,256)

pos = Linear(1152→256, no bias)(pos_embed 64x48)                      (3072,256)
q = Linear(LN(query)) + pos                                           (B,3072,256)
k = LN(Linear(fine_values)) + pos      fine_values (B,3072,1152)      (B,3072,256)
v = Linear(fine_values)                                               (B,3072,256)

SDPA(heads=8, head_dim=32, attn_mask = garment valid (B,1,1,3072))    (B,3072,256)
features = query + out_proj(transported) → (B,256,64,48)
residual = Conv2d(256→4, zero-init)(features + local(features))       (B,4,64,48)
residual *= gate(edit_mask, area-pool) * active                       (B,4,64,48)
```

Nếu `return_garment_centers=True`: SDPA thứ hai với `V = toạ độ key` (pad tới 32 chiều)
để lấy `centers (B,3072,2)` mà **không bao giờ vật chất hoá ma trận A** `3072x3072`.

### 4.8 Đầu ra

```
x → final_layer(x, cond)     (B,768, p*p*5) = (B,768,20)
  → unpatchify_rectangular   (B,5,64,48)
      ├── velocity  = x[:, :-1]   (B,4,64,48)
      └── logvar    = x[:, -1:]   (B,1,64,48)

velocity = velocity + fine_velocity        # cộng residual của refiner
```

---

## 5. Các thành phần loss (`trainer_vton.py::forward`)

```
loss = flow_loss
     + 0.01  * outside_velocity_loss
     + 0.01  * sigma_loss
     + 0.5   * detail_loss
     + 0.01  * ramp * attention_tv_loss
     + ramp  * correspondence_loss
     + 0.2   * decoded_rgb + 0.5 * decoded_edge
```

`ramp = min(1, step / 1000)` (`correspondence_warmup_steps: 1000`).

### 5.1 Loss cốt lõi

| Loss | Công thức | Mask |
|---|---|---|
| `flow_loss` | `mean((velocity - ut)^2)` | `masks.latent` |
| `outside_velocity_loss` | `mean(velocity^2)` | `1 - masks.latent` |
| `sigma_loss` | `DiagonalGaussian(velocity.detach(), logvar).nll(ut)` | `masks.latent` |

### 5.2 `detail_loss` (weight 0.5)

```
predicted_clean = xt + (1 - t_latent) * velocity            (B,4,64,48)
importance = 1 + 5 * clamp(edge(target)/mean_edge, 0, 1)    (B,1,64,48)
detail_loss = L1 của sai phân bậc nhất (ngang + dọc) giữa predicted_clean và target
```

Mask: `person_garment_mask` (coverage ≥ 0.8) ∧ `masks.latent` ∧ (t=0 thuần **hoặc**
`0.3 ≤ t ≤ 0.95`) ∧ `keep` ∧ `has_ground_truth`.

### 5.3 Correspondence / CORAL (`patch_flow/correspondence.py`)

**Teacher DINOv3 chỉ tồn tại khi train** — không có gì của nó được đưa vào mạng.

```
person  (B,3,512,384) --DINOv3 ViT-S/16--> (B,384,32,24) → resize về person_grid 32x24
garment (B,3,512,384) --DINOv3-----------> (B,384,32,24)
similarity = cosine  (B,768,768)
best_index → target uv (B,768,2) ∈ [-1,1]
weight (B,768): (sim ≥ 0.35) ∧ cycle-consistency ≤ 1.5 token ∧ coverage ≥ 0.8
                ∧ keep ∧ has_ground_truth ∧ garment_mask khác rỗng
```

Áp lên **từng head riêng** (`average_attn_weights=False`), cho từng entry trong
`attention_maps`:

| Term | Weight | Shape / mô tả |
|---|---|---|
| `nll` | 0.1 | `-log(mass trong bán kính)`; radius theo scale: coarse 0.04 / middle 0.03 / detail 0.025 |
| `center` | 0.05 | `‖A @ coords − target‖²`, `A (B,16,768,N_s) @ coords (N_s,2)` |
| `entropy` | 0.0 | tắt ở experiment này |
| `photometric` | 0.1 | `mean_heads(A) @ pool(garment RGB → lưới key)` phải khớp RGB thật của token đó trong ảnh mặc |
| `value` | 0.1 | `0.5*cosine + 0.5*huber` giữa `cross = out_proj(A@V)` và đặc trưng SD-VAE của **người đích**, qua embedder EMA (`decay 0.999`) đóng băng |

`value` là term **duy nhất** huấn luyện trực tiếp `V` và `out_proj` (mang logo/hoạ tiết);
mọi term còn lại chỉ dạy Q/K định tuyến.

### 5.4 `attention_tv_loss` (weight 0.01)

`patch_flow/attention_smoothing.py` — TV bậc nhất của **tâm attention** (`B,Q,2`) bên
trong mask áo, cho cả 14 block backbone lẫn refiner. Regularize hình học
correspondence, không phải RGB.

### 5.5 `decoded_*` loss (rgb 0.2 / edge 0.5)

Chọn ngẫu nhiên `decoded_max_samples: 1` mẫu có `0.3 ≤ t ≤ 0.95`, giải mã
`predicted_clean (1,4,64,48)` qua **decoder SD-VAE có gradient** (`_decode_with_grad`,
bọc checkpoint) → `(1,3,512,384)`, so L1 + L1-sai-phân với ảnh người thật trong mask áo.

### 5.6 Optimizer

```
adapter  (tên chứa "garment_" hoặc ".x_embedder")  → lr 1e-4
backbone (còn lại)                                  → lr 1e-4 * 0.1
AdamW, weight_decay 0, ema_rate 0 (tắt ở config garment-fix)
```

---

## 6. Sampling / inference

`patch_flow/flow_vton.py::generate` — `num_steps 30`, `cfg_scale 1.5`, `adaptive False`.

```
xt = masks.latent * noise + (1 - masks.latent) * person_context     (B,4,64,48)
token_times = 0 trong mask, 1 ngoài mask                            (B,768)

for (t_cur, t_next) in linspace(0,1,31) theo cặp:
    velocity = _predict(...)                                        (B,4,64,48)
    xt = xt + (t_next - t_cur) * velocity * masks.latent
    token_times += delta * masks.token
    xt = masks.latent * xt + (1 - masks.latent) * person_context    # re-anchor mỗi bước
```

**CFG**: nhân đôi batch, nhánh uncond nhận `garment/garment_middle/garment_detail`
và `garment_mask` **zero hoá** (đúng phân phối dropout khi train):

```
velocity = v_uncond + 1.5 * (v_cond - v_uncond)
```

Chế độ `adaptive=True`: dùng `logvar` chọn `uncertain_fraction=0.3` token khó nhất,
chạy `inner_steps` bước Euler nhỏ hơn chỉ cho các token đó.

Cuối cùng:

```
samples (B,4,64,48) --VAE.decode--> generated (B,3,512,384)
composed = compose_vton(generated, person, mask nở, feather 8)
```

---

## 7. Sơ đồ tổng quát

```
                  person ─┐
             agnostic_mask┤
                          ▼
   person_agnostic ─VAE─► person_context (B,4,64,48) ──┐
                                                        │
   noise x0 ──┐                                         │
   target x1 ─┴─ interpolant(t theo token) ─► xt (B,4,64,48)
                                                        │
                        concat[xt | agnostic | mask] (B,9,64,48)
                                     │ PatchEmbed p=2
                                     ▼
                              x (B,768,1152) + pos
                                     │
   garment ─VAE pyramid─┬─ coarse (B,4,64,48)  ─PatchEmbed─► (B,768,1152)
                        ├─ middle (B,256,128,96)─Conv4x4───► (B,768,1152)
                        └─ detail (B,128,256,192)─Conv4x4──► (B,3072,1152)
                                     │  LayerNorm → V ; V+pos → K
                                     ▼
        ┌──────────── 28 x VTONPatchForcingBlock ─────────────┐
        │  SelfAttn(adaLN) → [CrossAttn áo mỗi 2 block] → MLP │
        └──────────────────────┬──────────────────────────────┘
                               │ x (B,768,1152)
              ┌────────────────┴──────────────────┐
              ▼                                   ▼
     final_layer → unpatchify          GarmentLatentRefiner
     (B,5,64,48)                       q: 64x48 x 256, k/v: detail 3072
       ├── velocity (B,4,64,48) ◄──── + residual (B,4,64,48)
       └── logvar   (B,1,64,48)
```
