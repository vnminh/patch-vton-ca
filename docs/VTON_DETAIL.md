# Equal-grid detail refinement and garment supervision

Experiment: `viton-pft-xl-512x384-detail` (inherits the garment-mask fixes).

Both person and garment images stay **512x384**. All backbone cross-attention
routes now use 32x24 person queries and 32x24 garment keys. The existing detail
features are pooled only for these backbone routes. A new 256-channel refiner
uses **64x48 person queries and unpooled 64x48 garment keys/values**. Learned
subpixel query expansion distinguishes the four latent cells inside each PFT
patch; the refiner writes a 4-channel velocity residual directly at 64x48.
It does not pool the transported fine features back to 32x24. Its output
projection starts at zero. Other existing parameters retain their shapes.

## Losses

- Decoded RGB L1 (`decoded_rgb_weight: 0.2`) and first-difference edge L1
  (`decoded_edge_weight: 0.5`) supervise the estimated clean image through the
  frozen SD-VAE decoder. This bypasses both inference-only `no_grad` wrappers,
  preserving the VAE's exact normalization. Only paired, non-dropped garments
  and pixels with patch time 0.3 through 0.95 are eligible. Pure-noise samples
  retain the existing flow/latent-edge losses, not decoded reconstruction loss.
- Full-resolution garment/edit mask intersection selects supervised pixels;
  both endpoints must be valid for an edge. The decoder still has a receptive
  field, so gradients can propagate outside an eligible pixel's latent cell.
- `attention_tv_weight: 0.01` adds **StableVITON-inspired attention-center total
  variation**, averaging heads and regularizing neighboring query centers inside
  the garment. It applies to each backbone scale and the fine refiner, with
  scale-balanced averaging and the existing 1000-step correspondence warmup.
  Fine centers use a second fused coordinate-value reduction, not a stored
  3072x3072 attention matrix. No center calculation is needed at inference.

This is an adaptation of the [StableVITON ATV objective](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_StableVITON_Learning_Semantic_Correspondence_with_Latent_Diffusion_Model_for_Virtual_CVPR_2024_paper.pdf),
not a verbatim implementation: centers are probability-normalized coordinates
in [-1,1], and **both endpoints are masked** to avoid an artificial pull toward
zero at garment boundaries. It smooths correspondence geometry, not RGB/logos.
TV alone admits constant/collapsed centers; retain correspondence and appearance
losses and inspect previews. These weights are starting points, not tuned claims.

Decoder checkpointing and microbatch 1 / accumulation 32 keep the effective
single-GPU batch at 32 and leave more GPU headroom. Full XL GPU memory and
quality improvement require an actual new training run; CPU checks do not
establish either. DINO targets remain on their original 32x24 grid; the new
fine branch receives flow, decoded reconstruction, and masked center-TV losses.

## Server: warm-start with existing learned weights

Uploaded checkout: `/workspace/patch-forcing-detail-20260907`.
The active old checkout and training process are not modified or stopped.
Stop the old run yourself after a completed checkpoint, then run directly:

```bash
cd /workspace/patch-forcing-detail-20260907
source /venv/ai/bin/activate
export VITONHD_ROOT=/workspace/high-resolution-viton-zalando-dataset
export PFT_XL_CKPT=/workspace/patch-forcing/checkpoints/pft-xl_step400k_ema.ckpt
python train.py experiment=viton-pft-xl-512x384-detail \
  load_weights=/workspace/patch-forcing-garment-fix-20260906/logs/vton/pft-xl-512x384-garment-fix/2026-09-06/T160658/checkpoints/last.ckpt
```

This is a **weight warm-start with fresh optimizer and step counter**, not an
exact resume. New parameters make the old optimizer incompatible. Strict
loading allows a wholly absent refiner only with `allow_new_garment_refiner`;
partially missing or unexpected model weights still fail. Matching the backbone
key grids also changes attention behavior, even though weights are preserved.
Optionally append `+resume_step=19000` only if the selected checkpoint is step
19000; that changes the displayed counter, not optimizer history.

After this architecture saves its own checkpoint, exact optimizer resume uses:

```bash
python train.py experiment=viton-pft-xl-512x384-detail resume_checkpoint=/absolute/path/to/new-detail-run/checkpoints/last.ckpt
```

Do not specify `load_weights` and `resume_checkpoint` together. Do not launch
beside the old process on the same 24-GB GPU. Logs are under a separate
`logs/vton/pft-xl-512x384-detail` directory; no old checkpoints are deleted.
Monitor `decoded_rgb_loss`, `decoded_edge_loss`, `attention_tv/{scale}`,
`garment_grad/refiner/{query,key,value,output}`, and held-out paired previews.
The zero output initialization means value-path gradients start after the first
nonzero-learning-rate update; Q/K can also receive the TV gradient once its
warmup begins.

## Verification

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_vton.py tests/test_vton_supervision.py tests/test_vton_detail.py -q
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python scripts/verify_vton_detail.py \
  --data-root /workspace/high-resolution-viton-zalando-dataset \
  --vae-checkpoint /workspace/patch-forcing/checkpoints/sd_ae.ckpt \
  --height 512 --width 384
```

The real-data smoke test uses a small random DiT with real SD-VAE/DINO; it
checks finite backward/optimizer steps and decoder gradient parity, not quality.
`--checkpoint PATH` additionally audits full XL parameter shapes on CPU/meta.
