# Mask-Constrained Patch Forcing for Virtual Try-On

> **Phạm vi:** tài liệu này mô tả thiết kế PFT/VTON nền và lịch sử các lựa chọn
> flow/mask. Kiến trúc production hiện tại đã bổ sung DensePose, coherent warp và
> RGB+HF feature fusion. Khi nội dung garment/refiner ở đây khác với
> [`ARCHITECTURE_VI.md`](ARCHITECTURE_VI.md), tài liệu kiến trúc hiện tại được ưu tiên.

This document describes the VTON implementation in this repository. It focuses on the time distribution, latent construction, model architecture, conditioning paths, losses, inference procedure, and the reasons the design should transfer a pretrained PFT-XL model without leaking the paired ground-truth garment.

## 1. Task definition

The training sample contains:

| Symbol | Batch key | Meaning |
|---|---|---|
| \(I^*\) | `image` | Complete paired ground-truth person image |
| \(I_p\) | `person` | Person image used to construct the agnostic input |
| \(I_a\) | `person_agnostic` | Person with the original garment removed |
| \(I_g\) | `garment` | In-shop garment image; encoded by the SD-VAE pyramid |
| \(M\) | `agnostic_mask` | Region allowed to change; 1 means editable |
| \(M_g\) | `garment_mask` | Foreground mask for garment attention tokens |

Paired training normally has \(I_p=I^*\). This is safe only if the complete person is not used as model context. In this implementation, the full image is used as the supervised target and final RGB compositing reference. The context supplied to PFT is produced exclusively from the masked agnostic image.

At inference, \(I_p\) can contain a different old garment. The agnostic mask removes it before VAE encoding, while \(I_g\) provides the desired garment.

## 2. End-to-end data flow

```text
Paired target I* -------------------------> frozen VAE encoder ---> target latent z*

Person Ip ---> supplied agnostic mask M ---> token-grid rounding only
          ---> RGB mask M_T ---> remove garment before VAE
          ---> frozen VAE encoder -------------------------------> agnostic latent za

Garment Ig ---> frozen VAE encoder ---> latent zg -------------> coarse appearance tokens
                                   \--> 1/4-res feature map --> middle appearance tokens
                                   \--> 1/2-res feature map --> detail appearance tokens
Garment mask Mg -------------------------------------------------> garment token key mask

Noise epsilon + z* + za + token times --------------------------> evolving latent xt

[xt, za, interpolated mask] ---> zero-initialized input extension
zg, middle, detail + Mg ------> routed garment cross-attention
per-token time ---------------> per-token AdaLN modulation
                                     |
                              pretrained PFT-XL
                                     |
                              velocity + uncertainty
                                     |
                              final VAE latent
                                     |
                              frozen VAE decoder
                                     |
original person RGB + feathered edit mask ---> exact outside-mask composite
```

Garment appearance travels on three VAE branches at three resolutions. They answer
*what a garment region looks like*; the separate question of *which garment region
belongs where on the body* is no longer answered by a fourth key set but supervised
directly on these branches' attention maps (section 10.4).

There is deliberately no DINO conditioning branch. DINOv2 patch features are trained to
be invariant to exactly the appearance detail a try-on model has to copy — printed text,
logos, colour-block seams — so as keys they contributed correspondence and nothing else,
while costing an encoder at inference, a fourth attention route, and blocks that carried
no appearance at all. The correspondence signal is worth more applied as a loss than
offered as a key.

The full person latent is never used as clean context. This matters because masking a latent after encoding is insufficient: the VAE encoder has a spatial receptive field, so garment texture can already have spread into latent cells outside the nominal mask.

## 3. Mask representations

One mask representation is not suitable for every operation. The implementation derives three masks from the supplied agnostic mask. All three describe the **same** editable region: the mask is never grown beyond the token grid.

### 3.1 Token schedule mask

The input mask is max-pooled to the PFT token grid. A token becomes editable if any source-mask pixel contributes to it:

\[
M_T = \operatorname{MaxPoolToTokenGrid}(M).
\]

Max pooling is deliberate. Bilinear or average resizing could turn a small missed garment region into a low fractional value and then classify the token as preserved.

**No dilation is applied, and dilation is not configurable.** Token-grid rounding is the
only permitted growth, and \(M_T\) is used for every purpose: sampled times, noise, loss,
sampler updates, garment-attention queries, the person condition, and final compositing.

The rationale is quantitative. On the 11,647 VITON-HD training masks the supplied
agnostic mask covers 37.6% of the frame. Max-pooling to the 32-by-24 token grid raises
that to 45.0%, which is unavoidable at token granularity. One extra token of dilation
raised it to **60.0%** — a 60% increase over the supplied mask. Because the pixel
condition was masked at dilation 0 while generation ran at dilation 1, roughly 15% of the
frame was re-synthesised with no pixel conditioning at all, even though ground-truth
pixels existed there. At 512-by-384 that ring is the jawline, the neck, the hair falling
over the shoulder, and the waistband, so identity drifted and the majority of the loss
budget was spent re-inventing content that could simply have been copied.

Using one undilated mask therefore does three things: it restores ~15% of the frame to
exact copying, it concentrates the flow loss inside the garment, and it removes the
train-time discrepancy between the conditioned region and the generated region.

For 512-by-384 training, the VAE produces a 64-by-48 latent and PFT uses a 32-by-24 token grid. PFT-XL's pretrained convolutional patch projection transfers directly. Its frozen 16-by-16 positional embedding is bicubically interpolated to the rectangular grid, so rectangular training fine-tunes from the square pretrained model without changing checkpoint parameter shapes.

With the current SD VAE downsampling factor of 8 and PFT patch size 2, one PFT token covers approximately \(16\times16\) input pixels.

### 3.2 Hard latent update mask

Each token value is repeated over its \(2\times2\) latent cells:

\[
M_L = \operatorname{Repeat}_{2\times2}(M_T).
\]

This binary mask controls:

- which latent cells contain noise or target interpolation;
- which cells receive sampler updates;
- which cells contribute to the main flow loss;
- which agnostic latent cells are forced to zero;
- which cells are clamped back to agnostic context after every inference step.

### 3.3 Soft conditioning mask

A separate mask is area-resized to latent resolution and combined with a softened version of the hard mask. This one-channel map is supplied to the network as a condition. It tells the model where the boundary lies without being responsible for the safety-critical update decision. It is passed as both the person-condition mask and the edit mask, since the two regions now coincide.

### 3.4 RGB compositing mask

After VAE decoding, the token mask is upsampled and feathered inward. The final output is

\[
I_{out}=\alpha I_{gen}+(1-\alpha)I_p.
\]

The support of \(\alpha\) is restricted to the editable region. Therefore pixels outside that region are copied exactly from the original person image, even if the VAE decoder changes nearby pixels.

## 4. Preventing paired-data leakage

The following order is important.

1. Encode the target image to obtain its latent shape and the supervised target \(z^*=E(I^*)\).
2. Create the token mask from the supplied agnostic mask.
3. Upsample that token mask back to image resolution.
4. Apply it to the already agnostic RGB image:

   \[
   I_a^+=I_a\odot(1-M_T).
   \]

5. Encode only this masked agnostic image:

   \[
   z_a=E(I_a^+).
   \]

6. Zero the editable cells again after encoding:

   \[
   z_a\leftarrow z_a\odot(1-M_L).
   \]

This gives two defenses:

- removal happens before the VAE, preventing encoder receptive-field leakage from the original garment;
- editable latent cells are cleared again after the VAE.

The target image must still be encoded because supervised flow matching needs a target. Target information is used only inside the editable region and only at the noise level described by the corresponding timestep. Seeing progressively cleaner target signal as \(t\to1\) is normal diffusion/flow training, not label leakage.

## 5. Rectified-flow convention

This repository uses the convention

\[
t=0 \quad\text{means noise},\qquad t=1 \quad\text{means clean data}.
\]

For target latent \(z^*\) and Gaussian noise \(\epsilon\), the straight interpolation is

\[
z(t)=t z^*+(1-t)\epsilon.
\]

Its target velocity is constant:

\[
u=\frac{d z(t)}{dt}=z^*-\epsilon.
\]

PFT differs from ordinary rectified flow because every spatial token can have its own timestep.

## 6. Patchwise time sampling

### 6.1 Global ceiling

The configured `LogitNormalTruncatedGaussian` first samples a per-image time ceiling:

\[
\bar t=\sigma(\mu+s\eta),\qquad \eta\sim\mathcal N(0,1),
\]

where the current configuration uses

\[
\mu=0.5,\qquad s=1.0.
\]

This logit-normal distribution gives a broad range of global noise levels. Note the
structural consequence of the ceiling: because \(\bar t=\sigma(\mu+s\eta)\) is bounded away
from 1 and no token may exceed it, **no choice of \(\mu\) or \(s\) can put meaningful
probability mass near \(t=1\)**. Section 6.3 handles that regime explicitly instead.

### 6.2 Token times below the ceiling

For each token, define

\[
\delta=\min(\bar t/2,\,s_{tok}),
\]

then sample

\[
t_i=\bar t-\delta|\eta_i|,\qquad \eta_i\sim\mathcal N(0,1).
\]

If a rare draw produces \(t_i<0\), it is replaced by a uniform sample in \([0,\bar t]\).

Consequently, editable tokens have heterogeneous times but no editable token is cleaner than the sampled ceiling \(\bar t\). This is the LTG principle used by PFT to prevent the network from routinely seeing unrealistically clean generated patches during training.

The configured \(s_{tok}\) is 0.25. The previous value of 0.6 was inert: since
\(\bar t\in(0,1)\) implies \(\bar t/2<0.5<0.6\), the minimum always selected \(\bar t/2\) and the
parameter had no effect at all. At 0.25 the cap binds for \(\bar t>0.5\), which keeps
tokens closer to their ceiling and widens coverage of the cleaner half of the schedule.

### 6.3 Time mixture: garment forcing and detail refinement

Each training example draws one of three time regimes.

| Regime | Probability | Token times | Purpose |
|---|---|---|---|
| Garment forcing | 0.10 | all zero | editable state is pure noise, so prediction must use the person context and the garment condition rather than reading partially clean paired-target pixels out of the flow state |
| Detail refinement | 0.25 | ceiling pinned at \(t=1\), same LTG spread below it | trains the end of the trajectory, where high-frequency garment structure is written |
| LTG | 0.65 | \(\bar t=\sigma(0.5+\eta)\) with the spread of section 6.2 | the general patchwise-flow regime |

The detail-refinement regime exists because the LTG ceiling starves the very timesteps
that decide whether a logo appears. Measured over 20,000 draws of 768 tokens, the
previous configuration (\(\mu=0.7\), \(s_{tok}=0.6\), garment forcing 0.3) produced:

| | previous | current |
|---|---|---|
| mean \(t\) | 0.286 | 0.473 |
| \(P(t>0.8)\) | 2.17% | 8.7% |
| \(P(t>0.9)\) | **0.25%** | **7.8%** |
| \(P(t>0.95)\) | 0.02% | — |
| \(P(t>0.99)\) | **0.00%** | — |
| \(P(t=0)\) | 29.2% | 10.2% |

In rectified flow the interval \(t\to1\) is where fine structure — glyph edges, print
boundaries, colour-block seams — is committed. Receiving 0.25% of the training signal
there, and none at all above \(t=0.99\), meant the model was never taught to refine
garment detail, only to decide a plausible average garment. The final step of an
eight-step Euler sampler landed in a bin holding 0.5% of the training mass.

There is a real tension here, and it is why garment forcing is reduced rather than
removed. On paired data a high \(t\) means \(x_t\) already contains the true garment, so
high-\(t\) training partly rewards copying from the flow state instead of reading the
garment condition. Garment forcing is the counterweight. The resolution is that both
regimes are necessary: low \(t\) teaches *what to put there* from the condition, high \(t\)
teaches *how to sharpen it*. Starving either one is what produced a clean, correctly
coloured, entirely blank T-shirt.

The detail regime reuses the LTG spread with the ceiling set to 1, so it keeps the
heterogeneous per-token structure of Patch Forcing rather than collapsing to a single
clean timestep. `high_time_spread` controls that spread.

### 6.4 Known context versus generated tokens

After sampling, token times are overridden according to the edit mask:

\[
t_i^{eff}=
\begin{cases}
t_i,&M_{T,i}=1,\\
1,&M_{T,i}=0.
\end{cases}
\]

Outside-mask tokens are intentionally fixed at \(t=1\) because they represent known agnostic context, not generated content. This does not violate the train-inference matching goal: inference also starts with these same context tokens clean and fixed.

## 7. Training latent construction

The evolving latent given to PFT is

\[
x_t=
M_L\odot\left[t^{eff}z^*+(1-t^{eff})\epsilon\right]
+(1-M_L)\odot z_a.
\]

This equation has two spatial regimes:

### Editable garment region

\[
x_t=t_i z^*+(1-t_i)\epsilon.
\]

Each garment token has its own time. Difficult structures such as logos, folds, boundaries, and hands crossing clothing can therefore be trained under different local noise levels.

### Preserved context region

\[
x_t=z_a,\qquad t_i^{eff}=1.
\]

The context is clean but garment-agnostic. It provides identity, pose boundaries, background, hair, and other preserved evidence without exposing the paired garment.

## 8. PFT-XL backbone

The native pretrained model operates on:

- 4 latent channels;
- a \(32\times32\) SD-VAE latent for a \(256\times256\) image;
- a \(2\times2\) latent patch size;
- a \(16\times16=256\)-token sequence;
- hidden dimension 1152;
- 28 transformer blocks;
- 16 attention heads;
- a four-channel velocity output plus one uncertainty channel.

Every block contains self-attention and an MLP controlled by per-token AdaLN modulation. Because the timestep embedding has shape \((B,N,D)\), a clean context token and a noisy garment token can coexist in the same self-attention sequence while receiving different normalization parameters.

## 9. Person conditioning

The original PFT patch embedder accepts four state channels. The VTON model expands it to nine:

\[
[x_t\;(4),\;z_a\;(4),\;m_{cond}\;(1)].
\]

The original four-channel projection weights are copied from PFT-XL. The five new input-channel weights are initialized to exactly zero.

At initialization:

\[
\operatorname{Embed}_{VTON}([x_t,z_a,m])
=\operatorname{Embed}_{PFT}(x_t).
\]

Thus adding person conditioning does not disturb the pretrained denoiser on the first optimization step. Training gradually learns how agnostic appearance and mask location should modify token representations.

The agnostic latent is supplied twice for different purposes:

- as fixed clean context outside the edit region in \(x_t\);
- as an explicit condition channel, allowing every editable query to access spatial person information through transformer self-attention.

Both now use the same mask \(M_T\). Previously the condition channel used the undilated
mask while the flow context used a dilated one, so the ring between them was generated
without any pixel conditioning. With a single mask the two coincide and every preserved
pixel remains available.

Inside the hard editable region, the explicit agnostic latent is zero. Pose and identity cues therefore come from legitimate surrounding context rather than remnants of the old garment.

## 10. Garment conditioning

Garment appearance is injected through cross-attention in every fourth transformer block
— blocks 4, 8, 12, 16, 20, 24, and 28 — and each of those seven blocks is *routed* to one
of three garment branches. Every branch is an SD-VAE tap.

### 10.1 The three branches

| Branch | Source | Channels | Token grid at 512x384 | Tokens | What it carries |
|---|---|---|---|---|---|
| `coarse` | garment VAE latent \(z_g=E(I_g)\) | 4 | 32x24 | 768 | global appearance in the backbone's own latent space |
| `middle` | SD-VAE encoder 1/4-resolution feature map | 256 | 32x24 | 768 | mid-frequency appearance |
| `detail` | SD-VAE encoder 1/2-resolution feature map | 128 | **64x48** | **3072** | high-frequency appearance: glyphs, print edges, seams |

The `coarse` branch is embedded by a `PatchEmbed` whose weights are *copied from the
pretrained PFT-XL patch projection*, so garment tokens arrive in exactly the
representation the pretrained denoiser already reads. The `middle` and `detail` maps are
embedded by 4x4-stride convolutions. The `detail` branch is the only branch with a finer
grid than the person stream: four times the spatial resolution, at roughly 8 input pixels
per token instead of 16.

### 10.1.1 Why every branch is a VAE tap

An earlier revision carried a fourth `dino` branch: frozen DINOv2-small patch features as
an extra key set. It has been removed.

DINOv2 is trained for correspondence under heavy appearance augmentation, so its patch
tokens are deliberately invariant to the exact appearance a try-on model must reproduce.
It can report "dark navy panel here, lavender panel there" but not the shape of a letter.
So the branch could only ever contribute correspondence — and as a *key set* that
contribution is indirect: nothing forces the model to use those keys for placement rather
than ignoring them, and the blocks routed to `dino` carried no appearance at all. It also
cost an image encoder at inference and 22M parameters in every checkpoint.

Section 10.4 replaces it with the same information applied where it acts directly: a
training-time DINOv3 teacher that supervises *where the VAE branches' attention looks*.
The VAE branches are reconstruction-faithful by construction and therefore copyable; they
are the only thing that can transport a logo. Now they also get told where to put it.

### 10.2 Routing

`garment_scale_routes` assigns one branch to each cross-attention block. The 256 configuration uses:

```yaml
garment_scale_routes: [coarse, coarse, coarse, middle, middle, detail, detail]
```

The ordering is deliberate, coarse to fine. Placement — deciding which part of a flat-lay
garment belongs at which body location — is a low-resolution decision, so the `coarse`
branch comes first, while the residual stream is still settling layout; it is also the
branch the correspondence loss is cheapest to supervise. Transporting exact appearance is
a high-resolution problem and is placed last, close to the output, so fine structure is
not washed out by the remaining blocks.

### 10.3 Attention and token scale

\[
Q=W_Q h_{person},\qquad
K=W_K h_g,\qquad
V=W_V h_g,
\]

\[
CA(h,h_g)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}+B_{M_g}\right)V.
\]

The cross-attention residual is multiplied by the editable query mask, so garment
features are injected only into tokens that are allowed to change. Output projections are
zero-initialized, so at initialization the VTON model is numerically identical to
pretrained PFT.

Garment token magnitude matters as much as garment token content. The embedding gain is
`garment_embed_gain: 1.0`. It was previously 0.1, which had a measurable pathology: with
garment features projected at gain 0.1, garment tokens entered at standard
deviation 0.07 against a person stream at 1.0, giving cross-attention logits a standard
deviation of 0.035 across 768 keys. Measured at 768 keys, peak attention probability was
1.37x uniform — that is, the block returned an essentially unweighted average of every
garment token, a single global garment descriptor with no spatial selectivity. At unit
gain the same measurement gives 15.8x uniform. Attenuating the keys buys no stability,
because the zero-initialized output projection already guarantees a no-op at
initialization; it only flattens the attention.

`garment_mask` \(M_g\) does not replace garment features. It determines which garment
tokens are valid keys and values, and it is recomputed per branch at that branch's own
grid, so the finer `detail` grid gets a correspondingly finer foreground mask.

Each branch's tokens are LayerNormed before the positional embedding is added
(`garment_token_norm`). Nothing else on the key/value path normalises anything — the
queries get `garment_norm` inside each block, but keys and values were raw SD-VAE encoder
activations whose scale is decided by the VAE's internals. Measured at 512-by-384 that gave
per-token magnitudes of 25 (`coarse`), 143 (`middle`) and 68 (`detail`) against LayerNormed
queries at ~34; darker fabric produced systematically larger keys (navy 1.27–1.41x the
garment mean, biasing every query toward it); and the `coarse` keys were more than half
positional (content-to-position magnitude ratio 0.86, against 4.2 for `middle`).
Normalising the content *before* adding position fixes both the cross-branch scale spread
and that ratio.

Every branch receives the DiT grid positional embedding at full strength, interpolated to
that branch's own grid. This is what makes key index and spatial position interchangeable,
which the correspondence loss below depends on.

### 10.4 Correspondence and routing supervision (CORAL-style)

The flow loss supervises *pixels*, so it constrains placement only through a long and
noisy credit-assignment path: put the logo in the wrong place, get a slightly worse
latent MSE. The attention map is where placement is actually decided, so it is supervised
directly.

**Building the ground truth.** A frozen DINOv3 teacher (`facebook/dinov3-vits16-pretrain-lvd1689m`)
extracts patch features from the ground-truth person image \(I^*\) — where the garment is
already worn — and from the in-shop garment \(I_g\). Person features are resized to the
PFT token grid; garment features stay on the teacher's own grid. Both are L2-normalised
and matched by cosine similarity:

\[
j^*(i)=\arg\max_{j \in M_g} \; \langle \hat f^{person}_i, \hat f^{garment}_j \rangle .
\]

The matched garment token's normalised centre \(c_{j^*(i)} \in [0,1]^2\) is the target
position for person token \(i\). Only editable tokens participate, only garment tokens
inside \(M_g\) are candidates, and a match survives only if its similarity clears
`correspondence_min_similarity`. `correspondence_mutual` additionally requires cycle
consistency (the chosen garment token must choose that person token back): higher
precision, lower coverage.

**Target-mass loss (primary).** The barycentre and entropy terms below were measured to
be jointly insufficient, so the primary term constrains *which* keys are attended to:

\[
\mathcal L_{nll}=-\frac{\sum_i w_i \log \sum_{j:\lVert c_j-c_{j^*(i)}\rVert\le r} A_{ij}}{\sum_i w_i},
\]

with \(r=\)`correspondence_nll_radius` in normalised garment units, converted to an
integer key window per branch so every branch supervises the same physical area. Only that
window is gathered — a dense \((N_q,N_k)\) distance matrix would be several times larger
than the 768-by-3072 detail attention map it measures.

**Centre-of-mass loss (secondary).** For a supervised block with attention \(A_{ij}\) over
key positions \(c_j\),

\[
\mathcal L_{com}=\frac{\sum_i w_i \left\lVert \sum_j A_{ij} c_j - c_{j^*(i)} \right\rVert^2}{\sum_i w_i}.
\]

Because positions are normalised to \([0,1]^2\), one target serves every branch regardless
of its key-grid resolution: the 3072-key `detail` grid and the 768-key `coarse` grid are
scored in the same coordinate system.

It is weighted `0.25`, down from `1.0`. It provides a smooth long-range pull, but it is
not identifiable: measured on a real 32-by-24 run the coarse blocks reached a barycentre
1.7 tokens from the target with only 3.8 effective keys — and put their argmax 4.5 tokens
away, with 5% of their mass near the target. Sharp, barycentre-correct and *displaced* is a
bimodal map straddling the target, and what it retrieves is a blend of two fabrics. Navy
sleeves blended with a lavender body is the uniform violet that motivated section 10.5.

**Entropy loss.** A symmetric blur has the same centre of mass as a spike at that centre,
so uniform attention over a neighbourhood satisfies \(\mathcal L_{com}\) exactly as well as
attending to the right token. The entropy term removes that particular degenerate
solution — but note it penalises *spread*, not *displacement*, so it does not catch the
straddling failure above:

\[
\mathcal L_{ent}=\frac{\sum_i w_i H(A_i)/\log N_i}{\sum_i w_i},
\]

normalised by the entropy of the uniform distribution over that branch's \(N_i\) *usable*
keys (padded keys excluded), so branches of different key counts contribute comparably.
It is weighted an order of magnitude below the centre term: sharpening attention before
it points anywhere useful just locks in a wrong match.

**What this costs, and when.** Nothing at inference. The teacher runs under `no_grad` on
the ground-truth person image, which exists only during training, and no DINO feature ever
enters the network. The one training cost is real: `need_weights=True` disables fused
attention and materialises a \((B, N_q, N_k)\) map per supervised block. At 512-by-384 the
`detail` branch's map is four times larger than the others. The 512-by-384 configuration
nevertheless supervises it, `correspondence_scales: [coarse, detail]`, and drops `middle`
instead. Leaving `detail` unsupervised was measured to be the single largest contributor to
the mean-colour failure: it is routed to 6 of 14 blocks, accounts for the largest share of
garment influence (15.6% of the predicted velocity against 11.4% for `middle` and 5.1% for
`coarse`), and its attention spread over 500–1060 of 3072 keys with 0.4–0.7% of mass on the
target — a near-uniform average of the garment. `middle` is the cheaper thing to drop: its
keys were the least region-separable of the three (0.067 in value space, against 0.145 for
`coarse` and 0.219 for `detail`).

**Photometric routing loss.** The terms above are all in *position* space and depend on
the teacher. This one is in *appearance* space and depends on nothing but the paired data:
the mass-weighted garment colour a person token retrieves must match the colour that token
has in the ground-truth worn image,

\[
\mathcal L_{photo}=\frac{\sum_i a_i \left\lVert \sum_j A_{ij}\,g_j - p_i \right\rVert^2}{\sum_i a_i},
\]

with \(g_j\) the branch-grid mean RGB of garment key \(j\), \(p_i\) the token-grid mean RGB
of \(I^*\), and \(a_i\) covering every editable token — no confidence gate, because there
is no teacher to be unconfident. It is needed because DINOv3 correspondence is geometric,
not photometric: over 24 VITON-HD test pairs a delta at the DINOv3 target retrieves the
right colour to within 0.455 RGB, against 0.817 for the garment mean and 0.119 for the best
available garment token. Geometry recovers roughly half the colour signal; the rest has to
be asked for directly.

Because it needs no teacher, setting `correspondence_center_weight` and
`correspondence_nll_weight` to 0 leaves a fully functional routing loss and never
constructs the DINOv3 model at all.

**Diagnostics.** `train/correspondence_appearance_error` is \(\sqrt{\mathcal L_{photo}}\), in
the same RGB units as the numbers above: 0.817 is the mean-colour failure mode and 0.119 is
the oracle. `train/correspondence_target_mass` is the attention mass actually landing on
the target — measured at 0.05 for coarse and 0.005 for detail before the nll term existed.
`train/correspondence_coverage` is the fraction of editable tokens the
teacher was confident enough to supervise, and `train/correspondence_similarity` the mean
best cosine similarity. If coverage collapses toward zero, `correspondence_min_similarity`
is too aggressive and the loss is silently doing nothing. Per-block
`correspondence/block_NN_scale/{center,entropy}` show which blocks actually learn to
point.

Samples whose garment condition was dropped for classifier-free guidance get weight zero:
there is no garment to correspond to. `correspondence_warmup_steps` ramps both terms in
linearly, so the attention is not pinned before the freshly initialised garment
embedders produce anything worth pointing at.

## 11. Class and garment-free conditioning

The ImageNet class embedding from PFT-XL is retained. When no VTON category label is supplied, the model uses the pretrained null-class index 1000.

During training, all garment branches and the garment mask are dropped together for
approximately 10% of samples. Dropping them jointly matters: a null branch that still saw
one of the four conditions would not be unconditional. This teaches a
garment-unconditional branch.

At guided inference, the conditional velocity is

\[
v_{cfg}=v_{null}+w(v_{garment}-v_{null}),
\]

where \(w\) is `cfg_scale`. The agnostic person remains present in both branches; only the garment conditions are removed from the null branch, all four together and by the same zeroing used at training time. This makes guidance strengthen garment identity without discarding person identity or pose context.

## 12. Training losses

### 12.1 Editable-region flow loss

The main loss is evaluated only in the editable region:

\[
\mathcal L_{flow}
=\frac{\sum M_L\odot\|v_\theta(x_t,t,c)-u\|^2}
{C\sum M_L+\varepsilon}.
\]

This focuses capacity on the region the sampler will actually update.

### 12.2 Outside velocity regularization

Although outside cells are clamped and receive zero sampler step size, a small regularizer encourages their predicted velocity to be zero:

\[
\mathcal L_{outside}
=\operatorname{mean}_{1-M_L}\|v_\theta\|^2.
\]

Its default weight is 0.01. This discourages unstable irrelevant predictions without dominating the pretrained flow objective.

### 12.3 Uncertainty loss

PFT predicts one log-variance channel \(\ell_\theta\). It defines

\[
\sigma_\theta^2=\exp(\ell_\theta).
\]

The target velocity is scored under a diagonal Gaussian whose mean is the detached predicted velocity:

\[
\mathcal L_\sigma
=\operatorname{NLL}
\left(u;\operatorname{stopgrad}(v_\theta),\sigma_\theta^2\right).
\]

This loss is also restricted to the editable region and has default weight 0.01. Detaching the mean prevents the uncertainty objective from changing velocity merely to make variance prediction easier.

### 12.4 Optional high-frequency detail loss

`detail_loss_weight` enables an L1 penalty on first spatial differences of the predicted
clean latent \(\hat z=x_t+(1-t)v_\theta\):

\[
\mathcal L_{detail}
=\operatorname{mean}_{M_L}\left|\Delta_x\hat z-\Delta_x z^*\right|
+\operatorname{mean}_{M_L}\left|\Delta_y\hat z-\Delta_y z^*\right|.
\]

A plain MSE is tolerant of washed-out edges: blurring a logo costs little. Penalising the
gradient field directly penalises that blur. It defaults to `0.0`; set it to about `0.05`
to push high-frequency garment structure harder. It is disabled by default because it
changes the loss balance and should be introduced as a deliberate, separately attributed
change.

### 12.5 Total loss

\[
\mathcal L
=\mathcal L_{flow}
+0.01\mathcal L_{outside}
+0.01\mathcal L_\sigma
+w_{detail}\mathcal L_{detail}.
\]

## 13. Standard inference

Inference begins with noise only in the editable region:

\[
x_0=M_L\odot\epsilon+(1-M_L)\odot z_a.
\]

Token times begin at

\[
t_i=
\begin{cases}
0,&M_{T,i}=1,\\
1,&M_{T,i}=0.
\end{cases}
\]

For uniform Euler sampling, every editable token advances by the same \(\Delta t\):

\[
x\leftarrow x+M_L\odot\Delta t\,v_\theta(x,t,c).
\]

After every update, the outside region is clamped:

\[
x\leftarrow M_L\odot x+(1-M_L)\odot z_a.
\]

Clamping makes preservation an explicit sampler invariant rather than a behavior the model must learn.

## 14. Uncertainty-adaptive inference

Adaptive sampling should be enabled only after the uncertainty head has been fine-tuned on VTON.

At each outer step:

1. Predict velocity and log variance.
2. Average log variance over each \(2\times2\) latent token.
3. Rank only editable tokens.
4. Select the highest-uncertainty fraction, normally 30%.
5. Advance easy tokens by the complete outer step.
6. Advance uncertain tokens through several smaller inner steps, recomputing their velocity after each step.

For `inner_steps = k`, difficult tokens use step size

\[
\Delta t_{inner}=\Delta t/k.
\]

All editable tokens reach the same outer endpoint, but uncertain tokens receive \(k\) local evaluations. Preserved tokens remain fixed at \(t=1\) and are excluded from uncertainty ranking.

This design tests the main VTON-specific Patch Forcing hypothesis: fine garment structures should receive more computation while easy cloth areas and known person context provide progressively cleaner evidence.

## 15. Why this training setup should work

### 15.1 It matches inference states

Training and inference both contain clean agnostic context outside the edit mask and noisy/generated content inside it. The network is not trained with clean paired garment pixels in locations that will contain old-garment context during inference.

### 15.2 It prevents the simplest shortcut

The old or paired garment is removed before VAE encoding and editable agnostic latent cells are zeroed again. Garment forcing (section 6.3) additionally supplies pure noise in the edit region for 10% of examples. The easiest path to low loss is therefore to use the supplied garment condition rather than copy the worn garment.

### 15.3 It retains the pretrained image prior

All new person channels and garment cross-attention outputs are zero-initialized. Before fine-tuning, the VTON model is numerically equal to the original PFT model for the same state, time, and class inputs. This avoids destroying the pretrained natural-image and patchwise-flow behavior at initialization.

### 15.4 Known regions provide useful context

Self-attention can use clean face, hair, body boundary, pose, and background tokens to resolve noisy garment tokens. This is precisely the setting where patchwise heterogeneous time conditioning is useful.

### 15.5 Conditions have distinct responsibilities

- \(x_t\) represents the current generated state.
- \(z_a\) represents person identity and spatial context without the old garment.
- the soft mask identifies the editable boundary.
- the `coarse`, `middle`, and `detail` VAE branches carry copyable garment appearance at three spatial scales, and their attention maps carry garment/body correspondence.
- \(M_g\) removes irrelevant garment background keys.
- per-token time tells the model how reliable each spatial token currently is.

Keeping these roles separate makes it harder for the model to confuse a control signal with appearance content.

### 15.6 The objective matches sampler authority

The main loss is concentrated where the sampler is allowed to update. Outside preservation is guaranteed by clamping and RGB compositing rather than relying solely on learned reconstruction.

## 16. Low-memory fine-tuning

The 16 GB smoke configuration uses:

- batch size 1;
- four-step gradient accumulation;
- gradient checkpointing across transformer blocks;
- adapter, input projection, garment cross-attention, and final-layer training;
- frozen remaining PFT backbone parameters;
- no EMA model copy;
- no FID network;
- eight-step validation sampling.

The `detail` branch is the dominant memory cost: at 512-by-384 it contributes 3072
garment keys per routed block instead of 768. Cross-attention is normally called with
`need_weights=False` so PyTorch can use a memory-efficient attention kernel rather than
materializing the full 768-by-3072 matrix. Correspondence supervision is the exception —
it needs that matrix — which is why `correspondence_scales` exists and why the 512-by-384
configuration restricts it to the 768-key `coarse` and `middle` branches. The
512-by-384 experiment uses batch size 8 with four-step accumulation, holding the global
batch at 32. That is a deliberately conservative starting point — measure before
assuming. If a run does not fit, halve the batch and double the accumulation before
changing the routing; dropping a `detail` route is the change that costs the most
quality.

This mode is intended to validate data flow, checkpoint transfer, gradients, loss reduction, and paired/unpaired visual outputs. It is not a substitute for the later full-data ablation between adapter-only and broader attention fine-tuning.

## 17. Important limitations

1. PFT-XL was pretrained on a square grid. The implementation supports 512-by-384 fine-tuning through positional interpolation, but this remains resolution transfer rather than native rectangular pretraining.
2. One token equals roughly 16 input pixels. This handles small parsing errors but not large category topology changes by itself.
3. The implementation has no DensePose or explicit pose encoder. It relies on agnostic spatial context and visible body boundaries.
4. Exact readable text is bounded by the 1/2-resolution VAE feature grid, the frozen VAE decoder, and the 16-pixel person token.
8. Correspondence targets come from a DINOv3 teacher, so they are only as good as DINOv3's part-level matching on flat-lay-to-worn pairs. `correspondence_min_similarity` gates the obviously bad matches, but a confidently wrong match is supervised as if it were right. Watch `train/correspondence_coverage` and `train/correspondence_similarity`, and prefer `correspondence_mutual: true` if precision matters more than coverage.
7. The `detail` branch quadruples the garment key count in the blocks it is routed to. Memory, not quality, is the binding constraint on how many blocks can use it.
5. Adaptive uncertainty is meaningful only after the head has been trained on the VTON distribution.
6. Unpaired validation has no pixel-aligned ground truth. It should be judged qualitatively or with garment/person-specific metrics, not SSIM against the input person.

## 18. Required invariants and debugging checks

The implementation should maintain these invariants:

- With new condition weights at zero, VTON PFT output equals pretrained PFT output.
- Outside \(M_L\), training state equals the agnostic latent, never the full-person latent.
- Outside token time is exactly 1.
- Outside latent values do not change during either standard or adaptive sampling.
- Outside RGB pixels equal the source person after final composition.
- Garment cross-attention gradients reach its zero-initialized output projection.
- Once that projection is non-zero, gradients reach all four garment branch embedders.
- Person-condition gradients reach the five added input channels.
- Uncertainty ranking considers editable tokens only.
- The editable mask is never larger than the token-grid rounding of the supplied mask.
- The `coarse` garment branch equals `first_stage.encode(garment)` exactly.
- Garment cross-attention is measurably non-uniform at initialization.
- The token-time distribution keeps non-trivial mass both at \(t=0\) and above \(t=0.9\).

Note when debugging gradients locally: PFT's `final_layer.linear` is zero-initialized by
the DiT convention, so **without** a pretrained checkpoint in `PFT_XL_CKPT` every upstream
gradient is exactly zero and only `final_layer` trains. That is expected, not a broken
conditioning path. Perturb `final_layer.linear.weight` before asserting anything about
upstream gradients.

The tests in `tests/test_vton.py` cover the principal architectural and preservation invariants.

## 19. Code map

| Component | File |
|---|---|
| DINOv3 correspondence teacher and attention losses | `patch_flow/correspondence.py` |
| Garment VAE pyramid extraction | `patch_flow/vae_features.py` |
| VTON PFT architecture and conditioning | `patch_flow/models/pf_transformer_vton.py` |
| Patchwise time construction and samplers | `patch_flow/flow_vton.py` |
| Mask conversion and RGB composition | `patch_flow/vton_utils.py` |
| Masked losses and VAE preprocessing | `patch_flow/trainer_vton.py` |
| Paired/unpaired VITON-HD data | `patch_flow/vton_data.py` |
| Full training configuration | `configs/experiment/viton-pft-xl.yaml` |
| 512-by-384 configuration | `configs/experiment/viton-pft-xl-512x384.yaml` |
| 16 GB smoke configuration | `configs/experiment/viton-pft-xl-smoke16gb.yaml` |
| Single-example inference | `scripts/vton_sample.py` |

## 20. Minimal training sequence

1. Run `setup.sh` or provide compatible PFT-XL and SD-VAE checkpoints. Training additionally downloads the DINOv3 correspondence teacher, which is a gated Hugging Face repository — set `HF_TOKEN` for an account with access, or set `correspondence_center_weight` and `correspondence_entropy_weight` to 0 to train without it.
2. Generate or transfer the agnostic masks.
3. Run the 32-sample smoke split and smoke training.
4. Confirm that the paired result reconstructs the held-out garment and the unpaired result follows the new garment.
5. Confirm that background, face, and other outside-mask pixels remain exact after composition.
6. Train adapter-only on the full paired dataset.
7. Compare against broader self-attention/AdaLN fine-tuning.
8. Fine-tune and calibrate the uncertainty head.
9. Compare uniform and adaptive sampling at matched network-evaluation budgets.

## 21. Training steps and periodic validation previews

Training length, validation frequency, checkpoint frequency, and preview frequency are separate settings.

### Optimizer-step settings

The main settings are under `train_params`:

```yaml
train_params:
  max_steps: 50000
  accumulate_grad_batches: 4
  val_check_interval: 1000
  limit_val_batches: 1
  log_every_n_steps: 1
```

- `max_steps` is the number of optimizer updates before training stops.
- `accumulate_grad_batches` is the number of mini-batches used for one optimizer update.
- `val_check_interval` runs validation after this many optimizer updates.
- `limit_val_batches` controls how many validation batches are evaluated at each validation event.
- `log_every_n_steps` controls scalar loss logging.

For batch size 1 and `accumulate_grad_batches: 4`:

\[
50\text{ optimizer steps}\times4\text{ mini-batches}=200\text{ training samples seen},
\]

before accounting for repeated examples or multiple devices.

The approximate number of samples processed is

\[
N_{samples}=N_{steps}\times B_{device}\times N_{devices}\times N_{accumulation}.
\]

### Preview settings

VTON preview settings are under `trainer.params`:

```yaml
trainer:
  params:
    save_validation_previews: true
    preview_every_n_validations: 1
    sample_kwargs:
      num_steps: 30
      adaptive: false
```

- `save_validation_previews` enables PNG output.
- `preview_every_n_validations: 1` saves at every validation event; use 2 to save every second event.
- `sample_kwargs.num_steps` is the number of flow-sampling steps used to generate each preview. It does not change training length. Keep it at 30 or more for full runs: an eight-step preview is visibly under-resolved and will understate what the model has actually learned. Only the smoke configuration uses 8.
- Adaptive sampling should remain disabled for early smoke training.

Preview sheets are written to

```text
logs/<experiment>/<date>/<run>/previews/stepXXXXXX.png
logs/<experiment>/<date>/<run>/previews/latest.png
```

Each row is one validation example. Columns are ordered as target, source person, agnostic person, effective edit mask, garment, and try-on output. In the smoke split, the first row is paired and the second row is unpaired.

### Smoke-test example

The supplied 16 GB configuration uses:

```yaml
train_params:
  max_steps: 1000
  accumulate_grad_batches: 4
  val_check_interval: 5
  limit_val_batches: 1

checkpoint_params:
  every_n_train_steps: 500
```

It therefore produces previews at optimizer steps 5, 10, 15, and so on through step 1000, and checkpoints at steps 500 and 1000.

The smoke dataset contains only 32 training pairs, so the training loop reshuffles and repeats the finite dataloader across epochs until `max_steps` is reached. With batch size 1 and gradient accumulation 4, one pass supplies 8 optimizer updates; 1000 updates therefore require multiple passes over the smoke split.

Settings can be changed without editing YAML:

```bash
python train.py experiment=viton-pft-xl-smoke16gb \
  train_params.max_steps=100 \
  train_params.val_check_interval=5 \
  checkpoint_params.every_n_train_steps=20 \
  trainer.params.preview_every_n_validations=1
```

For full training, a reasonable initial schedule is:

```bash
python train.py experiment=viton-pft-xl \
  train_params.max_steps=50000 \
  train_params.val_check_interval=1000 \
  checkpoint_params.every_n_train_steps=5000
```

Validation sampling is much more expensive than a training update. Very small `val_check_interval` values are useful for a smoke test but should be increased for full training.
