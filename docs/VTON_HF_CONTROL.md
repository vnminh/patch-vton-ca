# Target-cloth high-frequency control

Experiment: `viton-pft-xl-512x384-detail-rebalance-hf`.
Logs: `vton/pft-xl-512x384-detail-rebalance-hf-control`.
The stable and rebalance experiments do not enable HF by default.

The main DiT input has 13 channels: noisy latent (4), agnostic latent (4),
person mask (1), DensePose latent (4). HF never enters `x_embedder`.

The dataset computes Canny from the transformed target cloth image and restricts
edges to its cloth mask. The same procedure applies to paired and unpaired data.
The map must remain at full image resolution, eight times the latent dimensions.

The separate `garment_high_frequency_control` branch performs:

1. Pixel-unshuffle: a 512x384 single-channel map becomes 64x48 with 64 channels.
   This rearrangement preserves within-block edge positions without max pooling.
2. Learned convolutions encode those pixel phases into 256-channel HF values.
3. Cross-attention uses the detail refiner's existing supervised Q/K and garment
   key mask to transport HF values to person coordinates. V contains HF content
   only. No positional embedding or person shortcut is added to these values.
4. A local residual block and a zero-initialized 1x1 convolution produce four
   velocity channels on the 64x48 person grid. The person edit mask bounds output.

`velocity = backbone_velocity + detail_velocity + hf_velocity`

Uncertainty output is unchanged. This is ControlNet-inspired conditioning with
a zero output head, not a full copied/frozen ControlNet backbone. It trains with
the existing flow and decoded reconstruction losses. Routing continues to use
the existing fine correspondence/RGB supervision; no extra HF loss is introduced.
The zero head receives gradients immediately; the encoder starts receiving
reconstruction gradients after the head has updated. Existing global LR warmup
still applies. The HF encoder and head have adapter learning rates and gradients
logged as `garment_grad/hf/encoder` and `garment_grad/hf/output`.

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
zero output; partial missing branch weights remain strict errors. A 13-channel
checkpoint keeps its initial function. The explicitly reverted 14-channel version
can also load: only its last input slice is discarded, so any contribution learned
by that removed slice is lost. Original checkpoint files are never changed.
After a checkpoint from this architecture exists, use `resume_checkpoint=...`
to restore its optimizer and step, without also specifying `load_weights`.

This branch cannot recover details absent from the Canny map and depends on the
quality of the existing detail attention. RGB garment features still carry colors.
Pixel-unshuffle preserves the input before learned encoding, but does not guarantee
that attention and generation preserve exact lettering. Compare held-out paired
and swapped previews before claiming a quality improvement.

Validation (2026-09-09): 97 local tests plus 7 subtests passed, including BF16
backward/validation, original prediction parity with a nonzero pretrained output,
EMA loading, partial-checkpoint rejection, masked/empty HF residuals, and CFG.
On the server, the XL step-1000 checkpoint schema matched all 409 existing model
tensors, with only 14 new HF tensors absent and the input still at 13 channels.
A small DiT with real pretrained SD-VAE/DINO completed two CPU optimizer steps
at 512x384 with finite gradients, nonzero HF output/encoder gradients, decoder
gradient parity, and finite unpaired CFG generation. Full-XL GPU memory and
generated-logo quality have not been measured for this revision.
