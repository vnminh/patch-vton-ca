I implemented it as **one additional high-resolution refinement block after the DiT backbone**, not by making all 28 DiT blocks high-resolution.

The implementation is in [GarmentLatentRefiner](/home/minh-le-vo-nhat/Documents/Minh-DUT/NCKH/NewAttempt2627/patch-forcing-cross/patch_flow/models/pf_transformer_vton.py:14).

### 1. Build fine person queries

The backbone produces `32×24` person tokens, each with 1152 channels.

```text
Backbone features:        1152 × 32 × 24
        ↓ Linear: 1152 → 4×256
        ↓ PixelShuffle ×2
Fine person features:      256 × 64 × 48
```

PixelShuffle gives each backbone token **four separately learned subqueries**, rather than repeating the same query four times.

I then add:

- Timestep conditioning, expanded to `64×48`.
- A 3×3 convolution of the noisy person latent and agnostic context, already at `64×48`.

So fine queries also receive information directly from individual latent cells—not only expanded coarse tokens.

### 2. Build garment keys and values

```text
Garment image:             3 × 512 × 384
        ↓ Frozen VAE encoder, early feature tap
Garment detail features: 128 × 256 × 192
        ↓ Learned 4×4 convolution, stride 4
Garment detail tokens:  1152 × 64 × 48
        ↓ Separate key/value projections
Keys and values:         256 × 64 × 48
```

These fine tokens are retained **before** pooling garment features for the backbone.

### 3. Cross-attention and direct latent correction

The new attention has:

- **3072 person queries and 3072 garment keys**.
- 256 channels, 8 heads.
- Positional information in queries/keys, not values.
- Garment-background keys masked out.

The transported features pass through a local depthwise 3×3 convolution block and a final projection:

```text
Fine attended features: 256 × 64 × 48
        ↓ Local refinement + 1×1 output projection
Velocity correction:      4 × 64 × 48
        ↓ Apply edit mask
Final velocity = backbone velocity + correction
```

**There is no pooling back to 32×24 in this output path.** The final projection starts at zero to introduce the new branch gradually.

### 4. How it learns

It receives gradients from the combined velocity’s flow loss, latent-edge loss, and decoded RGB/edge losses. Its attention centers also receive garment-masked smoothing.

One important limitation: **DINO/CORAL supervision still directly supervises the backbone attention, not the new fine attention.** The fine branch currently learns through reconstruction and center smoothing. Also, matching grids permits finer routing; it does not itself guarantee readable logos.