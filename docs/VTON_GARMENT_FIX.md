# Garment supervision fixes

Use `experiment=viton-pft-xl-512x384-garment-fix`. The existing 512x384
experiment is retained as a baseline. The new experiment preserves the model's
parameter shapes and optimizer groups, so the existing full checkpoint can resume.

## Changes

- Load semantic clothing labels `[5, 6, 7]` from `image-parse-v3`, preserving PNG
  palette indices. Apply exactly the person's flip/shift/scale to this mask.
- Keep the broad agnostic mask for generation. Only garment pixels inside that
  mask supervise correspondence, RGB transport, values, and latent detail edges.
  These ground-truth garment masks are never fed to the model or sampler.
- Supervise person tokens with at least 80% garment coverage, weighted by their
  coverage. Empty masks, dropped garments and unpaired targets get zero auxiliary
  supervision. Normalize pooled person/reference RGB by garment coverage to avoid
  mixing skin or white product background into appearance targets.
- Filter DINO matches by a round trip within 1.5 person tokens. This is Euclidean
  distance on the rectangular person grid, not exact mutual nearest-neighbor.
- Apply detail loss to pure-noise examples and eligible patches at
  `0.3 <= t <= 0.95`; both ends of an edge must be eligible garment pixels.
- Validate 8 fixed test people, their 8 unpaired swaps, and 4 training people every
  500 optimizer steps. Paired and unpaired versions share initial noise seeds.
  All 20 rows are retained in previews, with a JSON sidecar identifying each row.
- Report `val/test_paired/garment_rgb_mae` and `garment_edge_mae`, with separate
  training-example metrics. Scores use generated pixels before composition in
  the garment/edit intersection. Unpaired rows have no reconstruction score.

These losses and metrics are not guarantees of exact logo reproduction. Evaluate
fixed previews and held-out garment errors after a controlled continuation.

## Server resume

The uploaded checkout is `/workspace/patch-forcing-garment-fix-20260906`.
Stop the current training in its terminal with Ctrl+C first. The old loop only
saves at checkpoint intervals, so intervening unsaved steps will not resume.
To avoid losing them, wait for its next completed checkpoint before stopping.

```bash
cd /workspace/patch-forcing-garment-fix-20260906
/venv/ai/bin/python scripts/start_vton_garment_fix.py \
  /workspace/patch-forcing/logs/vton/pft-xl-512x384/2026-09-06/T050434/checkpoints/last.ckpt
supervisorctl status pft-vton-garment-fix
tail -f logs/supervisor-training.log
```

The launcher restores model, optimizer, scheduler, and global step. It refuses to
start while a `train.py` process exists and does not stop the old process itself.
It registers a dedicated Supervisor service; it does not edit the old checkout.
Automatic restarts are disabled to avoid silently repeating training from a stale
checkpoint. For a later restart, stop the service and invoke the launcher with
the new run's latest completed `last.ckpt`.

New runs are under `logs/vton/pft-xl-512x384-garment-fix/`; two step checkpoints
are retained. The correspondence loss warms up over 1,000 optimizer steps after
each launch, so total losses during warmup are not directly comparable.

## Verification

```bash
python -m pytest tests/test_vton.py tests/test_vton_supervision.py -q
VITONHD_ROOT=/workspace/high-resolution-viton-zalando-dataset \
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
/venv/ai/bin/python scripts/verify_vton_garment_fix.py \
  --data-root /workspace/high-resolution-viton-zalando-dataset \
  --vae-checkpoint /workspace/patch-forcing/checkpoints/sd_ae.ckpt
```

The integration checker uses actual data and pretrained encoders with a reduced,
randomly initialized DiT. It verifies finite BF16 gradients, an optimizer update,
and validation generation/metrics without using the active training GPU.
