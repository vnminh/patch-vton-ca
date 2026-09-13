<p align="center">
 <h2 align="center">Denoising, Fast and Slow: Difficulty-Aware Adaptive Sampling for Image Generation</h2>
 <p align="center">
 <b>
 Johannes Schusterbauer<sup>*</sup> · Ming Gui<sup>*</sup> · Yusong Li · Pingchuan Ma · Felix Krause · Björn Ommer
 </b>
 <p align="center"> 
    CompVis Group @ LMU Munich, Munich Center for Machine Learning (MCML)
 </p>
 <p align="center"> 
    CVPR 2026
 </p>
</p>
 </p>
<div align="center">


[![Website](https://img.shields.io/badge/Project-Page-lightgrey)](https://compvis.github.io/patch-forcing)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2604.19141)

<p align="center"> <sup>*</sup> <i>equal contribution</i> </p>

</div>


<p align="center">
<img src="assets/fpf.png" alt="Patch Forcing overview" width="500px">
</p>

# 🚀 TL;DR


**Patch Forcing turns denoising into a spatially adaptive process.** During training, different image patches receive heterogeneous timesteps. While conceptually straightforward, this only works well with a dedicated timestep sampler that controls how much clean information is exposed per sample, closing the train–test gap where inference starts from pure noise. This framework enables dynamic sampling strategies, where easy regions can be denoised faster and provide cleaner context for harder ones.


🔥 **Contributions**
- Patch-wise timesteps $\rightarrow$ enables heterogeneous denoising
- LTG timestep sampler $\rightarrow$ fixes train-test mismatch
- Patch difficulty-guided sampling $\rightarrow$ allocates compute adaptively


<img src="assets/fpf-inference.png" alt="Patch Forcing inference" width="100%">


# 📖 Overview


Natural images are highly spatially heterogeneous: some regions (e.g. backgrounds) are easy to denoise, while others (e.g. fine structures, text) require more refinement and context.
However, standard diffusion and flow-based models treat all regions equally, applying the same timestep and compute everywhere.


**Key idea**: move from global to patch-wise denoising, where different regions follow different noise trajectories.


### Training

Naively assigning random timesteps per patch does *not work*. When timesteps are sampled independently and uniformly, most training samples contain a mix of noisy and already partially clean regions. As a result, the model learns to rely on this implicit context, even though such states never occur at inference, where generation starts from pure noise. This creates a clear train–test mismatch.

<p align="center">
<img src="assets/srm-comparison.png" alt="Schedule Comparison" width="100%">
</p>

Prior work (SRM) addresses this by controlling the average amount of information per sample. While this partially mitigates the issue, it does not fully resolve it: even if the average is well-behaved, individual patches can still be nearly clean. In practice, this means that almost every training example still contains highly informative regions.


Our key idea is to instead **control the maximum information** available in each sample. Concretely, we first sample a maximum timestep and then restrict all patch-wise timesteps to lie below it. This prevents any region from becoming too clean during training and ensures that the model consistently operates in regimes that match inference.

With this simple change, heterogeneous patch-wise denoising works! Even without any adaptive sampling at inference, this training strategy already improves generation quality over standard diffusion models with uniform timesteps.


### Inference

To fully leverage patch-wise denoising at inference, we need to decide **which regions should be denoised faster** and **which require more refinement**. For this, we augment the model with a lightweight uncertainty (difficulty) head that predicts, for each patch, how reliable the current denoising velocity prediction is.

<p align="center">
<img src="assets/uncertainty.png" alt="Uncertainty" width="400px">
</p>

With heterogeneous denoising and the uncertainty head, we base our adaptive samplers on three key findings:

- **context helps denoising** $\rightarrow$ advancing confident (easy) regions provides cleaner context that improves predictions in harder regions
- **uncertainty reflects patch difficulty** $\rightarrow$ higher uncertainty correlates with higher validation loss
- **more context reduces uncertainty** $\rightarrow$ cleaner neighboring regions make difficult patches easier to denoise

These findings naturally lead to adaptive sampling strategies that allocate compute where it is most useful. Instead of denoising all patches uniformly, we use the predicted uncertainty to guide the process: easy regions are advanced more aggressively, while difficult ones receive additional refinement.


<p align="center">
<img src="assets/denoising-schedule-performance.png" alt="ImageNet Results" width="100%">
</p>

- The **dual-loop** sampler alternates between quickly advancing confident patches and refining uncertain ones with smaller steps.
- The **look-ahead** sampler goes one step further by explicitly advancing confident patches into the future and using their cleaner states as context for denoising harder regions.

**Together, these strategies turn patch-wise heterogeneity into adaptive inference, improving generation quality under the same compute budget by focusing effort where it matters most.**


Please refer to our paper for a more detailed description of our framework. 😉


# 🛠️ Code Setup

This codebase is based on Python `3.12` and the packages listed in `requirements.txt`.

First, clone the repository:

```bash
git clone git@github.com:CompVis/patch-forcing.git
cd patch-forcing
```

Then create the environment and install the dependencies:

```bash
conda create -n pft python=3.12
conda activate pft
pip install -r requirements.txt
```

If the default install fails on your machine, follow the safer install order noted in ?`requirements.txt`: install `torch` and `torchvision` first, then `flash-attn`, then the remaining requirements.

For an automated VTON setup, run:

```bash
bash setup.sh
conda activate pft-vton
export PFT_XL_CKPT="$PWD/checkpoints/pft-xl_step400k_ema.ckpt"
```

The script installs the pinned CUDA 12.8 dependencies, downloads PFT-XL, Stability AI's EMA SD VAE, and the DINOv3 correspondence teacher, verifies the VAE SHA-256 checksum, converts it to the bare state dictionary expected by `jutils`, and strictly loads the PFT/VAE checkpoints. It is resumable and skips existing downloads. Useful overrides include `ENV_NAME`, `CHECKPOINTS_DIR`, `CORRESPONDENCE_TEACHER`, `USE_CURRENT_ENV=1`, `SKIP_INSTALL=1`, `SKIP_VERIFY=1`, and `FORCE_DOWNLOAD=1`. Set `INSTALL_FLASH_ATTN=1` only when it is needed and the machine has a compatible CUDA build toolchain.

We release two [Patch Forcing Transformer](https://ommer-lab.com/files/pft/) checkpoints: [PFT-B](https://ommer-lab.com/files/pft/pft-b_step400k_ema.ckpt) and [PFT-XL](https://ommer-lab.com/files/pft/pft-xl_step400k_ema.ckpt). The checkpoints contain the EMA weights, as well as the model config.

### Class-Conditional Generation

#### Inference

To generate class-conditional samples use:

```bash
python scripts/sample.py \
  --ckpt /path/to/model.ckpt \
  --sample-fn-config configs/sampler/dual-loop.yaml \
  --num-sampling-steps 100 \
  --cfg-scale 4.0
  # ... you can add sampler specific args via dot-notation
```

For FID samples use `scripts/sample_ddp.py`:

```bash
torchrun --standalone --nproc_per_node=8 scripts/sample_ddp.py \
  --ckpt /path/to/model.ckpt \
  --sample-fn-config configs/sampler/euler-pf.yaml \
  --per-proc-batch-size 64 \
  --num-fid-samples 50000 \
  --num-sampling-steps 100 \
  --cfg-scale 1.0
```

If your checkpoint comes from training and does not already contain the compact `config` + `state_dict` format expected by the samplers, convert it first:

```bash
python scripts/convert_ckpt.py /path/to/training.ckpt
```


#### Training

You can train new models via `train.py`. The repository is based on `hydra`, and the base config lives in `configs/config.yaml`. Experiments in `configs/experiment` overwrite this base config. Use CLI overrides to swap configs or change individual fields, for example `python train.py experiment=imnet-pft-b name=imnet/my-run data=dummy256 train_params.max_steps=10000`.

To directly use the ImageNet-256 webdataset file, configure the ImageNet-256 shard locations in `configs/data/imagenet256.yaml`.
For debugging, use you can use `configs/data/dummy256.yaml`.

Train the main class-conditional experiments with:

```bash
python train.py experiment=imnet-pft-b
python train.py experiment=imnet-pft-xl
```

If you want to use your own dataloader, make sure it returns a dictionary with `image` (bchw tensor normalized to $[-1, 1]$) and `label`.

### Text-to-Image

For text-to-image training, first fill in the gaps in `configs/data/t2i-256.yaml` and then you can train them with

```bash
python train.py experiment=t2i-pft1.2b-qwen
```

The batch should contain a dict with `image`, text (set corresponding text key in trainer), and `img_meta` if you want to include crop size conditioning via RoPE (see `patch_flow/data_utils.py` for more info). The default loader uses random caption sampling (as we used multiple caption lengths during training).

You can use `scripts/t2i_sample.py` to sample images based on a text prompt.

### Virtual Try-On

The VTON extension fine-tunes the released PFT-XL checkpoint with mask-constrained
flow, agnostic-person and DensePose conditions, multiscale garment attention, and a
full-latent detail refiner. The current logo experiment is
`viton-pft-xl-512x384-detail-logo-hf`. Start with the ordered
[`docs index`](docs/README.md), then read the current
[`architecture guide`](docs/ARCHITECTURE_VI.md). Older experiment names below are
retained for reproducibility and do not describe the complete fused-HF graph.

The edit mask is applied before VAE encoding, and only the agnostic latent is used
as person-image context; the complete paired image is used as supervised target and
final RGB reference, never as inference conditioning.

Garment appearance travels on three SD-VAE branches, routed one per cross-attention
block: garment latent, encoder 1/4-resolution, and encoder 1/2-resolution detail.
After the DiT, coherent RGB and signed-RGB HF transports are fused at 64x48 before
one shared refiner predicts `fine_velocity`. There is no standalone HF velocity head.
Configure backbone assignment with `model.params.garment_scale_routes`.

Garment/body correspondence is supervised rather than supplied as an inference
condition. A frozen DINOv3 teacher provides sparse geometric anchors during training;
paired RGB/VAE transport, coordinate, mask and smoothness losses refine the coherent
grid below DINO resolution. At inference the teacher and ground-truth person garment
mask are absent; the model uses person agnostic, DensePose, in-shop garment, garment
mask, and HF computed from that same garment.

The barycentre term alone is not enough, and measurably so: a map can be sharp (3.8 effective keys of 768) and barycentre-correct (1.7 tokens off) while placing its peak 4.5 tokens away with 5% of its mass on the target. That straddling map retrieves a *blend* of two fabrics, which is how a navy-and-lavender garment renders as uniform violet. See [`docs/VTON_PFT_DESIGN.md`](docs/VTON_PFT_DESIGN.md) §10.4.

The edit mask is the token-grid rounding of the supplied agnostic mask, with no dilation, and dilation is not configurable. On VITON-HD, one token of dilation grew the editable region from 37.6% to 60.0% of the frame and left ~15% of it regenerated with no pixel conditioning, which cost identity around the jaw, neck, and hair.

See [`docs/ARCHITECTURE_VI.md`](docs/ARCHITECTURE_VI.md) for the current graph and
[`docs/VTON_PFT_DESIGN.md`](docs/VTON_PFT_DESIGN.md) for the baseline flow equations,
leakage analysis, and adaptive sampler design.

Prepare VITON-HD with `image`, `cloth`, `agnostic-mask`, and `cloth-mask` directories under each split, then set the dataset and pretrained checkpoint paths:

```bash
export VITONHD_ROOT=/path/to/VITON-HD
export PFT_XL_CKPT=/path/to/pft-xl_step400k_ema.ckpt
python train.py experiment=viton-pft-xl
```

The pair files default to `train_pairs.txt` and `test_pairs.txt` at the dataset root. Training and metric validation force paired garments by default; set `paired: false` only for unpaired qualitative evaluation. A dependency and data-pipeline smoke run is available with `data=dummy_vton256`.

Run a fine-tuned checkpoint with:

```bash
python scripts/vton_sample.py \
  --ckpt /path/to/vton-training.ckpt \
  --person person.jpg \
  --garment garment.jpg \
  --agnostic-mask person_mask.png \
  --garment-mask garment_mask.png \
  --output result.png
```

Add `--adaptive` after the uncertainty head has been fine-tuned. Pixels outside the edit mask are composited directly from the input person.

#### Metadata transfer and 16 GB smoke run

To deploy the full dataset and this working tree to the configured Vast.ai host,
install the dependencies and checkpoints, and start the smoke run under
Supervisor, run:

```bash
bash scripts/deploy_vton_remote.sh
```

The transfers are resumable. Run `scripts/transfer_code_remote.sh` and
`scripts/transfer_data_remote.sh` separately when only one side changed;
`scripts/transfer_vton_remote.sh` remains as a convenience wrapper for both.
Override `REMOTE_HOST`, `REMOTE_PORT`, `REMOTE_ROOT`, or `LOCAL_DATASET` when
needed. The data transfer uses six parallel rsync workers by default; override
that with `TRANSFER_JOBS`. Use `DRY_RUN=1` to preview. The remote launch stage is
`scripts/start_vton_training_remote.sh`.

Transfer only pair lists and generated agnostic masks to a machine that already contains the original VITON-HD images:

```bash
bash scripts/transfer_vton_metadata.sh \
  /local/path/VITON-HD \
  /remote/path/VITON-HD \
  azr-ai@100.75.140.87
```

Set `DRY_RUN=1` to preview the `rsync` file selection. On the remote machine, create a deterministic split with 32 paired training samples and one held-out person evaluated with paired and unpaired garments:

```bash
python scripts/make_vton_smoke_split.py \
  --dataset-root /remote/path/VITON-HD \
  --output-dir /remote/path/VITON-HD/smoke32
```

Run the low-memory PFT-XL smoke experiment:

```bash
export VITONHD_ROOT=/remote/path/VITON-HD
export VITONHD_SMOKE_DIR=/remote/path/VITON-HD/smoke32
export PFT_XL_CKPT=/remote/path/checkpoints/pft-xl_step400k_ema.ckpt
python train.py experiment=viton-pft-xl-smoke16gb
```

This configuration uses batch size 1, four-step gradient accumulation, gradient checkpointing, adapter-only optimization, no EMA copy, eight-step validation sampling, 1000 optimizer steps, and no statistical validation metrics. Its validation batch still saves both paired and unpaired try-on images.

Validation preview sheets are saved periodically under `logs/<experiment>/<date>/<run>/previews/`. The smoke configuration validates every 5 optimizer steps and saves `stepXXXXXX.png` plus an updated `latest.png`. Configure training and preview periods from the command line:

```bash
python train.py experiment=viton-pft-xl-smoke16gb \
  train_params.max_steps=100 \
  train_params.val_check_interval=5 \
  checkpoint_params.every_n_train_steps=20 \
  trainer.params.preview_every_n_validations=1
```

`max_steps` and `val_check_interval` count optimizer updates. With `accumulate_grad_batches=4`, one optimizer update consumes four mini-batches.

`checkpoint_params.save_top_k` controls checkpoint retention. Its default value of `1` keeps only the newest numbered checkpoint, with `checkpoints/last.ckpt` pointing to it.

The 512-by-384 experiment encodes garments online with the VAE pyramid. Three settings there are worth knowing about:

The current logo-HF variant uses a single detail residual: coherently warped RGB/VAE
features are added to warped signed-RGB HF features, then one shared refiner predicts
`fine_velocity`; the final output is `backbone_velocity + fine_velocity`. HF is neither
an `x_embedder` input channel nor an independent velocity head. See
[`docs/VTON_HF_CONTROL.md`](docs/VTON_HF_CONTROL.md) for the graph, checkpoint migration,
supervision boundaries, and restart command.

- **Batch size 8 with four-step accumulation** (global batch 32). The `detail` branch contributes 3072 garment keys per routed block instead of 768, so activation memory is high. This is a conservative starting point — measure peak memory and raise it if there is headroom. If it does not fit, halve the batch and double the accumulation before changing the routing.
- **`correspondence_scales: [coarse, detail]`.** Correspondence supervision needs the full attention matrix, which disables fused attention for the blocks it touches, so not every branch can be supervised at once. `detail` must be one of them: routed to 6 of 14 blocks and the largest single contributor to garment influence (15.6% of the predicted velocity), it was measured spreading its attention over 500–1060 of 3072 keys with 0.4–0.7% of mass on target — a near-uniform average of the garment, i.e. the mean-colour path itself. `middle` is dropped instead, being the least region-separable of the three branches. `correspondence_warmup_steps: 1000` ramps the losses in so attention is not pinned before the freshly initialised garment embedders produce anything worth pointing at.
- **Timestep mixture.** 10% of examples force every editable token time to zero (garment forcing), 25% pin the time ceiling at `t=1` (`high_time_probability`, detail refinement), and the rest use the logit-normal truncated-Gaussian schedule. The LTG ceiling `sigma(loc+z)` cannot itself reach `t=1`, so without the high-time regime the interval above `t=0.9` receives about 0.25% of the training signal — the interval where logo and printed-text detail is written. With it, that rises to about 7.8%.

`trainer.params.detail_loss_weight` (default `0.0`) adds an L1 penalty on first spatial differences of the predicted clean latent. Set it to about `0.05` to push high-frequency garment structure harder.

The DINOv3 teacher is a gated Hugging Face repository, so `HF_TOKEN` must belong to an account with access. To train without it, set `trainer.params.correspondence_center_weight=0` and `trainer.params.correspondence_entropy_weight=0`; the teacher is then never constructed. While it is on, watch `train/correspondence_coverage` — the fraction of editable tokens the teacher was confident enough to supervise. If it collapses toward zero, `correspondence_min_similarity` (default `0.35`) is too aggressive and the loss is silently doing nothing.

The two metrics that say whether the garment is actually being read are `train/correspondence_appearance_error` (RGB L2 of the retrieved colour: **0.817** is the garment-mean failure mode, **0.119** the best a garment token can do) and `train/correspondence_target_mass` (attention mass on the matched key; it was 0.05 for `coarse` and 0.005 for `detail` before the target-mass term existed).

`scripts/restart_vton_training.sh` launches a from-scratch run with all of this on and prints the resolved settings first. It refuses to start over a live run unless `FORCE_RESTART=1`. A resume is not possible across a `garment_token_norm` or `garment_attention_output_init_std` change: the first adds parameters, the second is an initialisation.

See [`docs/VTON_PFT_DESIGN.md`](docs/VTON_PFT_DESIGN.md) sections 3.1, 6.3, and 10 for the measurements behind these choices.


## 🎓 Citation

If you use our work in your research, please use the following BibTeX entry. 🙂

```bibtex
@InProceedings{schusterbauer2026patchforcing,
      title={Denoising, Fast and Slow: Difficulty-Aware Adaptive Sampling for Image Generation},
      author={Johannes Schusterbauer and Ming Gui and Yusong Li and Pingchuan Ma and Felix Krause and Björn Ommer},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2026}
}
```
