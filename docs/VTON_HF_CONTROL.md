# Target-cloth high-frequency control

Experiment: `viton-pft-xl-512x384-detail-rebalance-hf`.
Logs: `vton/pft-xl-512x384-detail-rebalance-hf-control`.
The stable and rebalance experiments do not enable HF by default.

The main DiT input has 13 channels: noisy latent (4), agnostic latent (4),
person mask (1), DensePose latent (4). HF never enters `x_embedder`.

The dataset computes Canny from the transformed target cloth image and restricts
edges to its cloth mask. The same procedure applies to paired and unpaired data.
The map must remain at full image resolution, eight times the latent dimensions.

The trainer encodes that map with the frozen pretrained SD-VAE, taking the
half-resolution encoder tap (256x192, 128 channels) -- the same features the
`detail` garment branch uses. `garment_high_frequency_channels` must therefore
equal `garment_detail_channels`.

The separate `garment_high_frequency_control` branch performs:

1. A learned 4x4 stride-4 convolution embeds those 128-channel VAE features into
   256-channel HF values on the 64x48 person grid, mirroring
   `garment_detail_embedder`. This convolution is zero-initialized.
2. Cross-attention uses the detail refiner's existing supervised Q/K and garment
   key mask to transport HF values to person coordinates. V contains HF content
   only. No positional embedding or person shortcut is added to these values.
3. A local residual block and a 1x1 convolution produce four velocity channels on
   the 64x48 person grid. The person edit mask bounds output. That convolution
   keeps its standard weight initialization; only its bias is zeroed.

`velocity = backbone_velocity + detail_velocity + hf_velocity`

Uncertainty output is unchanged. This is ControlNet-inspired conditioning, not a
full copied/frozen ControlNet backbone. It trains with the existing flow and
decoded reconstruction losses. Routing continues to use the existing fine
correspondence/RGB supervision; no extra HF loss is introduced.

Where the zero initialization sits is the whole design. The first revision put it
on the velocity head over a randomly initialized pixel encoder, on the assumption
that "the zero head receives gradients immediately; the encoder starts receiving
reconstruction gradients after the head has updated". That assumption is wrong,
and measurably so. Every path from the encoder to the residual passes through the
head, so a zero head makes the encoder gradient identically zero. Across 3500
steps of `detail-rebalance-hf-control`, `garment_grad/hf/encoder` never left
0.00000-0.00001, the checkpointed encoder rms stayed at 0.072657 against its
theoretical initialization value of 0.07217, and the head's own gradient decayed
from 0.0146 to 0.0017 rather than growing, because random features gave it no
consistent direction. The branch contributed nothing.

Zeroing the encoder instead inverts the dependency. The branch still contributes
exactly zero on the first step: `local` is bias-free and GroupNorm's bias starts
at zero, so a zero encoder output propagates as exact zero to the head's zeroed
bias. But the head keeps a usable weight, so gradient reaches the encoder from
step zero. The head's own weight has no gradient until the encoder output is
non-zero, which costs one step. Pretrained VAE features matter for the same
reason: the head now has a meaningful, spatially aligned signal to latch onto.

Existing global LR warmup still applies. The HF encoder and head have adapter
learning rates and gradients logged as `garment_grad/hf/encoder` and
`garment_grad/hf/output`. `garment_grad/hf/encoder` rising above zero within the
first few hundred steps is the check that this branch is alive at all.

Garment dropout zeros HF as well as garment conditions. CFG uses a zero HF map
for its unconditional half. Empty HF maps or cloth masks contribute exactly zero,
even after output biases have trained. The HF branch uses checkpointing and SDPA.
It adds another fine-grid attention operation, so full-model GPU cost must be
measured separately from small-model CPU smoke tests.

Decoded RGB and edge reconstruction use the complete agnostic edit mask. This
directly supervises arms and hands erased by agnostic preprocessing. Garment
correspondence, value, fine RGB, and latent detail losses continue to use the
parsed garment-only mask, so anatomy is never treated as garment appearance.

Warm-start with `load_weights=...` and a fresh optimizer. The experiment enables
`allow_new_garment_high_frequency`: a wholly absent HF branch is allowed with a
zero encoder; partial missing branch weights remain strict errors. A 13-channel
checkpoint keeps its initial function. The explicitly reverted 14-channel version
can also load: only its last input slice is discarded, so any contribution learned
by that removed slice is lost. Original checkpoint files are never changed.
After a checkpoint from this architecture exists, use `resume_checkpoint=...`
to restore its optimizer and step, without also specifying `load_weights`.

This branch cannot recover details absent from the Canny map and depends on the
quality of the existing detail attention, which is its binding constraint: it
transports HF values with the refiner's Q/K, and `fine_top1_accuracy` has sat
between 5.8% and 6.7%. Restoring the refiner's `state` convolution addresses that
separately. RGB garment features still carry colors. Compare held-out paired and
swapped previews before claiming a quality improvement.

Warm-start discards the previous revision's `garment_high_frequency_control`
tensors outright: they are renamed and reshaped, and the measurements above show
they carry no learned information.

Validation (2026-09-09, this revision): 97 tests plus 7 subtests passed on CPU,
including BF16 backward/validation, prediction parity against a nonzero pretrained
output, EMA loading, partial-checkpoint rejection, masked/empty HF residuals, CFG,
and the new gradient ordering (encoder from step 0, head weight from step 1).
Against the real `detail-rebalance-hf-control` step-2000 checkpoint the schema
matched 415 existing model tensors, with 4 new tensors (`garment_refiner.state`
and the rebuilt HF encoder), 8 discarded obsolete tensors, no input migration and
the input still at 13 channels. With real 512x384 paired images and the frozen
SD-VAE and DINO, two CPU optimizer steps ran with finite gradients throughout,
non-zero HF encoder gradients, decoder gradient parity, and finite unpaired CFG
generation. Full-XL GPU peak memory and generated-logo quality are not measured
here; the previous revision's HF branch also passed its own smoke test while
being inert in training, so `garment_grad/hf/encoder` on the real run is the check
that matters.

## Refiner query (2026-09-09)

Unrelated to HF but changed in the same revision. `GarmentLatentRefiner` regained
the `state` convolution over `cat(noisy, agnostic)` at full 64x48 latent
resolution, added to the query before `query_norm`.

Commit `be7102d` removed it together with the person residual around `A @ V`.
Those are different things. Removing the residual from the output path closes a
shortcut that let the branch ignore garment content, and it stays removed. The
query never bypasses attention into the output, so nothing about that argument
applied to `state`, and removing it left the four subpixel queries inside a patch
differing only by a learned constant slice of `query_expand` and by their
positional embedding -- both content-independent. Nothing could tell one subpixel
which garment cell it needed except absolute position, which is the wrong cue for
a logo that moves with the garment. `fine_top1_accuracy` stayed near 6% while
`fine_correspondence_weight` was tripled from 0.05 to 0.15, which is what a
capacity ceiling looks like rather than an under-weighted loss.

`state` is zero-initialized so a loaded refiner keeps its routing on the first
step. Unlike the old HF head, this zero sits on an input branch whose downstream
path is already non-zero, so it receives gradient immediately. Checkpoints
without it load under `allow_new_garment_refiner`; any other partially missing
refiner weight is still a strict error.
