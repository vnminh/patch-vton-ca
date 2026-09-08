# Equal-grid detail refinement and garment supervision

2026-09-08 update: use `viton-pft-xl-512x384-detail-stable` for the bounded
fine-attention/RGB-transport revision. See [training audit and restart
instructions](VTON_TRAINING_AUDIT_20260908.md). The supervised experiment and
results below document the previous stage, not a successful logo-quality result.

Experiment: `viton-pft-xl-512x384-detail-supervised` (inherits the garment-mask fixes).

Both person and garment images stay **512x384**. All backbone cross-attention
routes now use 32x24 person queries and 32x24 garment keys. The existing detail
features are pooled only for these backbone routes. A new 256-channel refiner
uses **64x48 person queries and unpooled 64x48 garment keys/values**. Learned
subpixel query expansion distinguishes the four latent cells inside each PFT
patch; the refiner writes a 4-channel velocity residual directly at 64x48.
It does not pool the transported fine features back to 32x24. Person/backbone
features form the queries, but the output path is strictly
`attention @ garment_value -> projection -> local block -> velocity`: the old
direct person-query residual has been removed.

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
- Fine correspondence (`0.2`) directly supervises every refiner head at 64x48
  using reliable DINO targets upsampled from 32x24. The positive set is a 3x3
  garment-key neighborhood. The QxK loss is computed in checkpointed 256-query
  chunks, so the full 3072x3072 attention matrix is never retained.
- Fine transported-value supervision (`0.25`) aligns the actual 256-channel
  `out_proj(A@V)` with unpooled target-person VAE features through frozen EMA
  projectors. It directly trains V/output content transport.
- There is no attention-TV/ATV implementation or configuration. The previous
  loss admitted the observed fixed-key collapse and was removed completely.

Decoder checkpointing, serialized per-image decoding, microbatch 4 and
accumulation 8 keep the effective single-GPU batch at 32. Every eligible sample
is decoded (`decoded_max_samples: 0`), rather than one sample per microbatch.
Full XL GPU memory and
quality improvement require an actual new training run; CPU checks do not
establish either.

## Server: warm-start with existing learned weights

Uploaded checkout: `/workspace/patch-forcing-detail-supervised-20260907`.
The active old checkout and training process are not modified or stopped.
Stop the old run yourself after a completed checkpoint, then run directly:

```bash
cd /workspace/patch-forcing-detail-supervised-20260907
source /venv/ai/bin/activate
export VITONHD_ROOT=/workspace/high-resolution-viton-zalando-dataset
python train.py experiment=viton-pft-xl-512x384-detail-supervised \
  model.params.pretrained_ckpt=null \
  load_weights=/workspace/patch-forcing-detail-20260907/logs/vton/pft-xl-512x384-detail/2026-09-07/T022635/checkpoints/step002000.ckpt \
  +resume_step=2000
```

This is a **weight warm-start with a fresh optimizer**, not an exact resume.
It keeps the displayed step at 2000 while restarting the LR and correspondence
warmups. New EMA target projectors are initialized from the loaded student.
Obsolete person-shortcut tensors are explicitly discarded; other missing or
unexpected weights still fail strict loading.

After this architecture saves its own checkpoint, exact optimizer resume uses:

```bash
python train.py experiment=viton-pft-xl-512x384-detail-supervised resume_checkpoint=/absolute/path/to/new-supervised-run/checkpoints/last.ckpt
```

Do not specify `load_weights` and `resume_checkpoint` together. Do not launch
beside the old process on the same 24-GB GPU. Logs are under a separate
`logs/vton/pft-xl-512x384-detail-supervised` directory; no old checkpoints are deleted.
Monitor `decoded_rgb_loss`, `decoded_edge_loss`, `fine_target_mass`,
`fine_top1_accuracy`, `fine_value_loss`,
`garment_grad/refiner/{query,key,value,output}`, and held-out paired previews.

## Verification

Verified on 2026-09-07: 77 local tests plus 7 subtests passed, including BF16
backward and paired/swap validation. The uploaded code passed two 128x96
real-data CPU optimizer updates with pretrained SD-VAE/DINO, nonzero fine
query/key/value/output gradients, frozen VAE parameters, and exact decoder
output parity. Fine NLL fell 2.410 to 2.351, target mass rose 0.0927 to 0.0980,
and value loss fell 0.236 to 0.217 in that wiring check. The full XL step-2000
checkpoint loaded strictly: 409 model tensors matched, six obsolete shortcut
tensors were discarded, and new EMA targets were initialized from the loaded
student. No full-XL GPU training was started.

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_vton.py tests/test_vton_supervision.py tests/test_vton_detail.py -q
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python scripts/verify_vton_detail.py \
  --data-root /workspace/high-resolution-viton-zalando-dataset \
  --vae-checkpoint checkpoints/sd_ae.ckpt \
  --height 512 --width 384
```

The real-data smoke test uses a small random DiT with real SD-VAE/DINO; it
checks finite backward/optimizer steps and decoder gradient parity, not quality.
`--checkpoint PATH` additionally audits full XL parameter shapes on CPU/meta.
