# Fine-attention stability and garment fidelity audit

## Observed training, not a quality claim

Run: `/workspace/patch-forcing-detail-supervised-20260907/logs/vton/pft-xl-512x384-detail-supervised/2026-09-07/T101546`.
At inspection the main process was PID 585696; the other matching Python
processes were its data-loader children. GPU utilization was 99%, memory
20,574 MiB. No job was stopped or restarted.

TensorBoard scalar windows are means of the logged last microbatch, **not**
means over every example in the accumulated batch:

| Metric | Steps 2501–3000 | Steps 3501–3873 |
|---|---:|---:|
| Fine correspondence NLL | 162.925 | 166.307 |
| Fine target-neighborhood attention mass | 0.00875 | 0.00983 |
| Fine learned-value loss | 0.08760 | 0.05797 |
| Flow loss | 0.46318 | 0.46185 |
| Pre-clipping gradient norm | 29.80 | 48.72 |

No nonfinite entries appeared in these logged metrics. Gradient clipping is
1.0: a large auxiliary gradient can suppress useful reconstruction updates.
Decreasing learned-value loss did not establish successful correspondence.

Held-out paired garment RGB MAE increased from 0.07794 at step 2500 to
0.08351 at step 3500. The small fixed panel is diagnostic, not a benchmark.
The step-3500 VANS sample still omitted dark sleeves and invented lettering;
the training Levi's sample also had incorrect lettering. Thus this is not
only a held-out generalization problem.

A read-only FLOAT32 CPU probe of the full XL step-3000 checkpoint, real VANS
pair at 512x384 and t=0.5, measured raw fine Q RMS 412.615, K RMS 36.097,
maximum absolute valid attention score 120,442.47, and mean top-key mass
0.644 over 91 sampled reliable queries. Every model weight loaded strictly.
This supports an attention-scale failure, not a claim that a smaller token
alone prevents logo reconstruction.

With the new positional/QK normalization on the same saved weights, the
maximum score was 9.284 and fine NLL 4.988. Attention was initially diffuse
(entropy 7.171, target mass 0.00684); the raw target mass had been 0.01275.
**This is stabilization, not an immediate alignment or image-quality gain.**
The model still needs to learn garment routing under the revised objective.

## Changes in `viton-pft-xl-512x384-detail-stable`

1. Normalize the projected positional feature without learned gain so it
   stays comparable to the normalized person/garment content. Normalize Q/K
   per head **after** projection and positional addition. Fixed cosine scale
   10 bounds scores to [-10,10] in exact arithmetic. SDPA and supervision
   receive the same scaled Q/K tensors; this is not a training-only clamp.
   Garment PE goes into K only, never V. The person Q has its own positional
   contribution. The raw garment values remain content-only at every scale.
2. Transfer discrete DINO matches with nearest interpolation, not bilinear UV
   interpolation across rejected matches or sleeve/chest discontinuities.
   The 3x3 positive neighborhood still expresses coarse-teacher uncertainty.
   This does **not** turn the 16-pixel DINO teacher into an 8-pixel teacher.
3. Add per-head fine RGB retrieval supervision on fixed, mask-pooled garment
   image colors versus paired worn colors. White background and skin are
   excluded when pooling. Pair/dropout/empty-garment gates apply independently
   of DINO confidence. This provides a non-learned appearance target; shading
   differences mean it must remain a weak auxiliary cue, not a hard warp.
4. Reduce fine NLL weight 0.2→0.05 and learned-value weight 0.25→0.1; RGB
   retrieval weight is 0.1. Preserve decoded RGB/edge and latent reconstruction.
5. Compute supervised Q/K dot products in FP32, handle empty positive/key
   sets safely, fix fractional-weight top-1 accounting, and log Q/K RMS.

No new learned parameters; checkpoint tensor shapes are unchanged. Old
experiments keep raw attention unless `garment_refiner_qk_norm` is enabled.
Both images remain 512x384, backbone grids 32x24 and fine grids 64x48.
ATV remains removed. No OCR or pixel-copy warp is claimed or implemented.

The normalization rationale follows [Query-Key Normalization for
Transformers](https://arxiv.org/abs/2010.04245); our fixed bounded scale and
positional normalization are implementation choices, not a reproduced VTON
result from that paper.

## Deployment and validation

New isolated checkout: `/workspace/patch-forcing-detail-stable-20260908`.
Use a **weight warm-start with a fresh optimizer** for the changed attention
and objective; do not run a second full training process on the same GPU.
The original code, active process and checkpoint files are preserved.

```bash
cd /workspace/patch-forcing-detail-stable-20260908
source /venv/ai/bin/activate
export VITONHD_ROOT=/workspace/high-resolution-viton-zalando-dataset
python train.py experiment=viton-pft-xl-512x384-detail-stable \
  model.params.pretrained_ckpt=null \
  load_weights=/workspace/patch-forcing-detail-supervised-20260907/logs/vton/pft-xl-512x384-detail-supervised/2026-09-07/T101546/checkpoints/last.ckpt
```

Wait for a completed checkpoint before stopping the old run; unsaved updates
cannot be recovered by this command. `last.ckpt` selects the latest completed
checkpoint when the command runs (step 3000 at inspection).
Weights are retained but displayed steps, optimizer, LR and auxiliary warmups
restart from zero. Afterwards, resume this experiment's own
checkpoint using `resume_checkpoint=PATH` without `load_weights`.

Checks: regression suite (including BF16 backward, fixed RGB gradients,
masked boundaries, empty positives, bounded attention and chunked checkpoint
gradient equivalence); `scripts/verify_vton_detail.py --experiment
viton-pft-xl-512x384-detail-stable` for real-data small-DiT training;
`scripts/audit_vton_fine_attention.py` for the full-checkpoint CPU probe.
These establish wiring/numerical behavior, not improved generated logos or
full-XL GPU memory. Quality must be compared after training using unchanged
paired and swapped previews, fine target mass, decoded edge/RGB errors and
held-out garment metrics—not total loss or learned-value loss alone.

The final revision passed two **512x384** CPU optimizer updates with a small
DiT and real pretrained VAE/DINO: fine NLL 5.463→5.339, fine RGB retrieval L1
0.1973→0.1735, and decoded RGB L1 0.1424→0.1391. All parameter gradients were
finite, VAE parameters remained frozen, and decoding preserved exact output
parity with the inference decoder. This is a wiring smoke test, not evidence
that the full XL checkpoint now generates correct lettering.

Local regression result: **86 tests and 7 subtests passed**, including the
PE-only-routing test. The server environment does not have pytest installed;
its verification used the standalone real-data script and full checkpoint
probe instead. No packages were installed into the running training environment.
