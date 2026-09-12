import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch.utils.checkpoint import checkpoint

from .pf_transformer import PatchForcingDiT, pf_modulate

GARMENT_SCALES = ("coarse", "middle", "detail")


def attention_sampling_grid(query, key, valid, height, width):
    """Return per-head soft-argmax garment coordinates without storing attention."""
    height, width = int(height), int(width)
    if key.shape[-2] != height * width or valid.shape != (query.shape[0], height * width):
        raise ValueError("Attention keys, validity mask and garment grid do not match")
    rows = (torch.arange(height, device=query.device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=query.device, dtype=torch.float32) + 0.5) / width
    y, x = torch.meshgrid(rows, columns, indexing="ij")
    coordinates = torch.stack((x.mul(2).sub(1), y.mul(2).sub(1)), dim=-1)
    coordinates = coordinates.reshape(1, 1, height * width, 2).to(query.dtype)
    return F.scaled_dot_product_attention(
        query,
        key,
        coordinates.expand(query.shape[0], query.shape[1], -1, -1),
        attn_mask=valid[:, None, None, :],
        dropout_p=0.0,
    )


def hard_attention_sampling_grid(query, key, valid, height, width):
    """Straight-through hard garment coordinate for coherent coarse routing.

    A global soft-argmax can average two distant modes and point between garment parts.
    The forward pass therefore uses the winning key while the backward pass follows the
    soft expectation, allowing correspondence losses to keep training Q/K.
    """
    height, width = int(height), int(width)
    if key.shape[-2] != height * width or valid.shape != (query.shape[0], height * width):
        raise ValueError("Attention keys, validity mask and garment grid do not match")
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = query.float() @ key.float().transpose(-1, -2) / math.sqrt(query.shape[-1])
        logits = logits.masked_fill(~valid[:, None, None], float("-inf"))
        probability = logits.softmax(-1)
    rows = (torch.arange(height, device=query.device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=query.device, dtype=torch.float32) + 0.5) / width
    y, x = torch.meshgrid(rows, columns, indexing="ij")
    coordinates = torch.stack((x.mul(2).sub(1), y.mul(2).sub(1)), -1).reshape(-1, 2)
    soft = probability @ coordinates
    hard = coordinates[logits.argmax(-1)]
    return hard + soft - soft.detach()


def upsample_displacement_grid(coarse_grid, coarse_size, fine_size):
    """Upsample coarse displacement rather than UV, preserving the identity warp."""
    coarse_height, coarse_width = (int(value) for value in coarse_size)
    fine_height, fine_width = (int(value) for value in fine_size)
    batch, heads, tokens, coordinates = coarse_grid.shape
    if tokens != coarse_height * coarse_width or coordinates != 2:
        raise ValueError("Coarse sampling grid shape does not match its spatial size")

    def identity(height, width):
        rows = (torch.arange(height, device=coarse_grid.device, dtype=torch.float32) + 0.5) / height
        columns = (torch.arange(width, device=coarse_grid.device, dtype=torch.float32) + 0.5) / width
        y, x = torch.meshgrid(rows, columns, indexing="ij")
        return torch.stack((x.mul(2).sub(1), y.mul(2).sub(1)), -1)

    coarse_identity = identity(coarse_height, coarse_width)
    displacement = coarse_grid.float().reshape(batch, heads, coarse_height, coarse_width, 2)
    displacement = displacement - coarse_identity[None, None]
    displacement = displacement.permute(0, 1, 4, 2, 3).reshape(
        batch * heads, 2, coarse_height, coarse_width
    )
    displacement = F.interpolate(
        displacement, (fine_height, fine_width), mode="bilinear", align_corners=False
    )
    displacement = displacement.reshape(batch, heads, 2, fine_height, fine_width).permute(
        0, 1, 3, 4, 2
    )
    output = displacement + identity(fine_height, fine_width)[None, None]
    limit = output.new_tensor((1 - 1 / fine_width, 1 - 1 / fine_height))
    return output.clamp(-limit, limit).reshape(batch, heads, fine_height * fine_width, 2)


def local_attention_sampling_grid(query, key, valid, base_grid, height, width, radius):
    """Search only a small residual window around a coherent coarse correspondence."""
    height, width, radius = int(height), int(width), int(radius)
    batch, heads, queries, channels = query.shape
    if radius < 0 or key.shape != query.shape or queries != height * width:
        raise ValueError("Local routing expects equal-grid Q/K and a non-negative radius")
    if valid.shape != (batch, queries) or base_grid.shape != (batch, heads, queries, 2):
        raise ValueError("Local routing masks/base grid do not match Q/K")
    dy, dx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=query.device, dtype=torch.float32),
        torch.arange(-radius, radius + 1, device=query.device, dtype=torch.float32),
        indexing="ij",
    )
    offsets = torch.stack((dx.flatten().mul(2 / width), dy.flatten().mul(2 / height)), -1)
    candidate = base_grid.float()[..., None, :] + offsets[None, None, None]
    limit = candidate.new_tensor((1 - 1 / width, 1 - 1 / height))
    inside = (candidate.abs() <= limit).all(-1)
    count = offsets.shape[0]
    sampling = candidate.reshape(batch * heads, height, width * count, 2)

    key_map = key.transpose(-1, -2).reshape(batch * heads, channels, height, width)
    sampled_key = F.grid_sample(
        key_map.float(), sampling, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    sampled_key = sampled_key.reshape(batch, heads, channels, queries, count).permute(0, 1, 3, 4, 2)
    valid_map = valid[:, None].expand(-1, heads, -1).reshape(
        batch * heads, 1, height, width
    ).float()
    sampled_valid = F.grid_sample(
        valid_map, sampling, mode="nearest", padding_mode="zeros", align_corners=False
    ).reshape(batch, heads, queries, count) > 0.5
    candidate_valid = inside & sampled_valid
    empty = ~candidate_valid.any(-1)
    if empty.any():
        candidate_valid = candidate_valid.clone()
        candidate_valid[..., count // 2] |= empty
    logits = torch.einsum("bhqd,bhqcd->bhqc", query.float(), sampled_key) / math.sqrt(channels)
    probability = logits.masked_fill(~candidate_valid, float("-inf")).softmax(-1)
    return (probability[..., None] * candidate).sum(-2).clamp(-limit, limit)


def sample_attention_heads(values, sampling_grid, valid, height, width):
    """Bilinearly sample each value head once at its routed garment coordinate."""
    batch, heads, tokens, head_width = values.shape
    height, width = int(height), int(width)
    if tokens != height * width or sampling_grid.shape != (batch, heads, tokens, 2):
        raise ValueError("Per-head values and sampling grid do not match garment grid")
    source = values.transpose(-1, -2).reshape(batch, heads, head_width, height, width)
    source = source * valid[:, None, None].reshape(batch, 1, 1, height, width).to(source.dtype)
    source = source.reshape(batch * heads, head_width, height, width)
    grid = sampling_grid.reshape(batch * heads, height, width, 2).to(source.dtype)
    sampled = F.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return sampled.reshape(batch, heads, head_width, height, width).permute(
        0, 1, 3, 4, 2
    ).reshape(batch, heads, tokens, head_width)


class GarmentLatentRefiner(nn.Module):
    """Equal-grid cross-attention with a separate output at every VAE latent cell.

    Learned subpixel queries preserve the four phases of a PFT patch. Garment
    detail values and the velocity residual never pass through coarse pooling.
    SDPA avoids materialising a full fine-grid attention matrix during training.
    """

    def __init__(self, backbone_dim, channels=4, width=256, heads=8, patch_size=2,
                 qk_norm=False, cosine_scale=10.0, dense_pose_channels=0,
                 local_radius=2):
        super().__init__()
        if width < 1 or heads < 1 or width % heads or width // heads < 2:
            raise ValueError("Refiner width must be divisible by heads with at least two channels per head")
        self.heads = heads
        self.width = width
        self.patch_size = patch_size
        self.dense_pose_channels = int(dense_pose_channels)
        self.local_radius = int(local_radius)
        if self.local_radius < 0:
            raise ValueError("Fine refiner local radius must be non-negative")
        if self.dense_pose_channels not in (0, channels):
            raise ValueError("Fine DensePose channels must be zero or equal latent channels")
        self.qk_norm = bool(qk_norm)
        self.cosine_scale = float(cosine_scale)
        if not math.isfinite(self.cosine_scale) or not 0 < self.cosine_scale <= 30:
            raise ValueError("Refiner cosine scale must be finite and in (0, 30]")
        self.query_expand = nn.Linear(backbone_dim, width * patch_size ** 2)
        # Fine-resolution person evidence for the query. Without it the four subpixel
        # queries inside a patch differ only by a learned constant slice of
        # ``query_expand`` and by their positional embedding, both content-independent,
        # so nothing can tell one subpixel which garment cell it needs. That is the
        # ceiling behind fine_top1_accuracy sitting near 6% while
        # fine_correspondence_weight was tripled from 0.05 to 0.15.
        # Zero-initialised, so a loaded refiner keeps its routing on the first step.
        # Unlike a zero velocity head, this zero sits on an input branch whose
        # downstream path is already non-zero, so it receives gradient immediately.
        self.state = nn.Conv2d(channels * 2 + self.dense_pose_channels, width, 3, padding=1)
        nn.init.zeros_(self.state.weight)
        nn.init.zeros_(self.state.bias)
        # The fine decoder must correct the velocity that will actually be integrated,
        # not predict an independent residual in parallel. The first half is the
        # backbone+HF preliminary velocity and the second half is its x1 estimate.
        # Zero initialisation makes this architecture checkpoint-compatible: before
        # the adapter learns, the old and new velocity sums are exactly identical.
        self.velocity_condition = nn.Conv2d(channels * 2, width, 3, padding=1)
        self.reset_velocity_condition()
        self.position = nn.Linear(backbone_dim, width, bias=False)
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(backbone_dim, width)
        self.value = nn.Linear(backbone_dim, width)
        self.attention_out = nn.Linear(width, width)
        # Zero keeps the coherent local warp. The adapter can learn to mix global
        # context back in, but washed-out global A@V is no longer the warm-start path.
        self.warp_mix = nn.Linear(width, width)
        self.reset_warp_mix()
        self.local = nn.Sequential(
            nn.GroupNorm(1, width),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.GELU(), nn.Conv2d(width, width, 1, bias=False),
        )
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_velocity_condition(self):
        """Neutralize the cascade adapter for a function-preserving warm start."""
        nn.init.zeros_(self.velocity_condition.weight)
        nn.init.zeros_(self.velocity_condition.bias)

    def reset_warp_mix(self):
        """Neutralize deformable transport for a function-preserving warm start."""
        nn.init.zeros_(self.warp_mix.weight)
        nn.init.zeros_(self.warp_mix.bias)

    def route(self, tokens, noisy, agnostic, values, position, garment_mask, dense_pose=None):
        """Transport garment-detail values and return their supervised routing.

        Person/backbone features construct Q -- including the noisy and agnostic latents
        at full 64x48 resolution, which is what lets one subpixel query differ from its
        three neighbours by content rather than by position alone. This phase is kept
        separate so the HF branch can reuse Q/K before the final correction is decoded.
        """
        batch, _, height, width = noisy.shape
        if values.shape[1] != height * width:
            raise ValueError("Person and garment detail grids must have identical resolution")
        ph, pw = height // self.patch_size, width // self.patch_size
        query = self.query_expand(tokens).transpose(1, 2).reshape(batch, -1, ph, pw)
        query = F.pixel_shuffle(query, self.patch_size)
        state_inputs = (noisy, agnostic.to(noisy.dtype))
        if self.dense_pose_channels:
            if dense_pose is None:
                raise ValueError("Fine refiner requires DensePose latent")
            dense_pose = F.interpolate(
                dense_pose, size=(height, width), mode="bilinear", align_corners=False
            ).to(noisy.dtype)
            if dense_pose.shape[1] != self.dense_pose_channels:
                raise ValueError("Fine refiner DensePose channels do not match configuration")
            state_inputs = (*state_inputs, dense_pose)
        elif dense_pose is not None:
            raise ValueError("Fine refiner received DensePose with dense_pose_channels=0")
        query = query + self.state(torch.cat(state_inputs, dim=1))
        query = query.flatten(2).transpose(1, 2)
        pos = self.position(position)
        if self.qk_norm:
            # A large shared positional component also makes cosine attention nearly
            # constant across garment content. Keep position comparable to normalized
            # person/garment content before combining them, without learned gain.
            pos = F.layer_norm(pos.float(), (self.width,)).to(pos.dtype)
        q = self.query(self.query_norm(query) + pos)
        k = self.key_norm(self.key(values)) + pos
        v = self.value(values)
        def heads(tensor):
            return tensor.reshape(batch, height * width, self.heads, -1).transpose(1, 2)
        if garment_mask is None:
            valid = torch.ones(batch, height * width, device=tokens.device, dtype=torch.bool)
        else:
            valid = F.adaptive_max_pool2d(garment_mask.float(), (height, width)).flatten(1) > 0
        active = valid.any(1)
        # An empty/dropped garment must be finite AND contribute exactly zero.
        valid = valid.clone()
        valid[~active, 0] = True
        q, k, v = heads(q), heads(k), heads(v)
        if self.qk_norm:
            # Normalize AFTER projection AND position, per head. Pre-projection LN
            # cannot bound positional/projection growth. Fold the fixed temperature
            # into Q/K so SDPA and the chunked training loss use identical scores.
            gain = math.sqrt(self.cosine_scale * math.sqrt(q.shape[-1]))
            q = (F.normalize(q.float(), dim=-1, eps=1e-6) * gain).to(q.dtype)
            k = (F.normalize(k.float(), dim=-1, eps=1e-6) * gain).to(k.dtype)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0,
        )
        coarse_height, coarse_width = ph, pw
        def pool_heads(tensor):
            source = tensor.transpose(-1, -2).reshape(
                batch * self.heads, tensor.shape[-1], height, width
            )
            pooled = F.avg_pool2d(source, self.patch_size, self.patch_size)
            return pooled.reshape(
                batch, self.heads, tensor.shape[-1], coarse_height * coarse_width
            ).transpose(-1, -2)

        coarse_q, coarse_k = pool_heads(q), pool_heads(k)
        if self.qk_norm:
            gain = math.sqrt(self.cosine_scale * math.sqrt(coarse_q.shape[-1]))
            coarse_q = (F.normalize(coarse_q.float(), dim=-1, eps=1e-6) * gain).to(q.dtype)
            coarse_k = (F.normalize(coarse_k.float(), dim=-1, eps=1e-6) * gain).to(k.dtype)
        coarse_valid = F.max_pool2d(
            valid.reshape(batch, 1, height, width).float(),
            self.patch_size, self.patch_size,
        ).flatten(1) > 0
        coarse_sampling_grid = hard_attention_sampling_grid(
            coarse_q, coarse_k, coarse_valid, coarse_height, coarse_width
        )
        base_grid = upsample_displacement_grid(
            coarse_sampling_grid, (coarse_height, coarse_width), (height, width)
        )
        sampling_grid = local_attention_sampling_grid(
            q, k, valid, base_grid, height, width, self.local_radius
        )
        warped = sample_attention_heads(v, sampling_grid, valid, height, width)
        attended = attended.transpose(1, 2).reshape(batch, height * width, self.width)
        warped = warped.transpose(1, 2).reshape(batch, height * width, self.width)
        transported = self.attention_out(warped + self.warp_mix(attended - warped))
        features = transported.transpose(1, 2).reshape(
            batch, self.width, height, width
        )
        return features, {
            "scale": "refiner", "query": q, "key": k,
            "output": transported, "sampling_grid": sampling_grid, "key_valid": valid,
            "grid": (height, width), "query_grid": (height, width),
            "coarse_query": coarse_q, "coarse_key": coarse_k,
            "coarse_key_valid": coarse_valid,
            "coarse_sampling_grid": coarse_sampling_grid,
            "coarse_grid": (coarse_height, coarse_width),
        }, active

    def refine(self, features, preliminary_velocity, preliminary_clean, edit, active):
        """Decode a fine correction conditioned on the backbone+HF provisional flow.

        The caller stop-gradients the provisional tensors. The adapter learns how to
        correct the current state without using this conditioning shortcut to rewrite
        the backbone or HF branch. The bounded multiplicative modulation cannot become
        a person-only additive shortcut: garment-transported detail stays the carrier.
        """
        if preliminary_velocity.shape != preliminary_clean.shape:
            raise ValueError("Preliminary velocity and clean estimate must have identical shapes")
        if preliminary_velocity.shape[-2:] != features.shape[-2:]:
            raise ValueError("Preliminary flow and refiner feature grids must be identical")
        condition = self.velocity_condition(
            torch.cat((preliminary_velocity, preliminary_clean), dim=1)
        )
        fused = features * (1 + torch.tanh(condition))
        residual = self.output(fused + self.local(fused))
        height, width = features.shape[-2:]
        gate = F.interpolate(edit.float(), (height, width), mode="area").clamp(0, 1)
        return residual * gate * active[:, None, None, None].to(residual.dtype)

    def forward(self, tokens, noisy, agnostic, values, position, edit, garment_mask,
                return_supervision=False, preliminary_velocity=None, preliminary_clean=None,
                dense_pose=None):
        """Return the garment-only correction, optionally with compact loss tensors."""
        features, supervision, active = self.route(
            tokens, noisy, agnostic, values, position, garment_mask, dense_pose
        )
        if (preliminary_velocity is None) != (preliminary_clean is None):
            raise ValueError("Both preliminary_velocity and preliminary_clean must be provided together")
        if preliminary_velocity is None:
            preliminary_velocity = torch.zeros_like(noisy)
            preliminary_clean = torch.zeros_like(noisy)
        residual = self.refine(
            features, preliminary_velocity, preliminary_clean, edit, active
        )
        if return_supervision:
            return residual, supervision
        return residual


class GarmentHighFrequencyControl(nn.Module):
    """Cloth HF control whose values come from the frozen pretrained SD-VAE encoder.

    Routing comes from the supervised detail refiner. Only HF content enters V; person
    features and position affect routing, never bypassing it into the output.

    Values are centred across keys so the branch can only transport deviations from the
    garment-mean edge response, which is what "high frequency" has to mean for a residual
    driven by attention that is not yet sharp. Unlike the refiner, whose value mean is the
    garment's base colour and must be kept, the HF mean carries no information.

    Two further things differ from the first revision, both forced by measurement. The
    values are frozen pretrained VAE half-resolution features of the cloth detail maps --
    the same basis the refiner's garment keys and values already live in -- instead of a
    randomly initialised pixel encoder. And the zero initialisation sits on the encoder output
    rather than on the velocity head: a zero head makes every upstream gradient
    identically zero, and over 3500 steps that left the old encoder's first convolution
    at exactly its initialisation rms (0.072657 against a theoretical 0.07217) while the
    head's own gradient decayed from 0.0146 to 0.0017 instead of growing. Zeroing the
    encoder keeps the branch's contribution exactly zero on the first step while leaving
    ``output`` at standard initialisation, so gradient reaches the encoder immediately.
    """

    def __init__(self, width, heads, channels=4, source_channels=128, patch_size=4):
        super().__init__()
        self.heads = heads
        self.width = width
        self.source_channels = int(source_channels)
        self.patch_size = int(patch_size)
        self.encoder = nn.Conv2d(
            self.source_channels, width, self.patch_size, stride=self.patch_size
        )
        self.local = nn.Sequential(
            nn.GroupNorm(1, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False), nn.SiLU(),
            nn.Conv2d(width, width, 1, bias=False),
        )
        self.output = nn.Conv2d(width, channels, 1)
        self.warp_mix = nn.Conv2d(width, width, 1)
        self.reset_zero_gate()

    def reset_zero_gate(self):
        """Make the branch contribute exactly zero without disabling its gradient.

        ``local`` is bias-free and GroupNorm's bias starts at zero, so zero encoder
        output propagates as exact zero up to ``output``'s bias, which is zeroed too.
        ``output.weight`` deliberately keeps its standard initialisation.
        """
        nn.init.zeros_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.output.bias)
        self.reset_warp_mix()

    def reset_warp_mix(self):
        """Zero starts from the shared coherent warp; global context is optional."""
        nn.init.zeros_(self.warp_mix.weight)
        nn.init.zeros_(self.warp_mix.bias)

    def forward(self, high_frequency, query, key, valid, edit, garment_mask,
                sampling_grid=None):
        batch, channels, source_height, source_width = high_frequency.shape
        height = source_height // self.patch_size
        width = source_width // self.patch_size
        if (channels != self.source_channels
                or source_height % self.patch_size or source_width % self.patch_size
                or query.shape[-2] != height * width):
            raise ValueError(
                "HF control expects frozen VAE half-resolution features at four source "
                "cells per person latent cell"
            )
        # Garment dropout and CFG zero these features, so an exact-zero test still
        # identifies a dropped sample. An empty cloth mask is caught separately: the VAE
        # of a neutral detail map is not itself zero unless its baseline was subtracted.
        active = high_frequency.flatten(1).ne(0).any(1)
        if garment_mask is not None:
            active = active & garment_mask.flatten(1).ne(0).any(1)
        features = self.encoder(high_frequency)
        values = features.flatten(2).transpose(1, 2)
        # Centre the values across the valid keys. A sparse edge map can otherwise give
        # its VAE features a large global mean: measured 0.5591 for the former Canny input
        # against -0.0102 for the garment features. Attention that is anywhere near
        # diffuse then returns that mean, and the branch injects a spatially constant
        # offset into the latent -- a flat colour shift once decoded. At step 1000 that
        # was 82.7% of the HF residual and 8.5% of the whole velocity.
        #
        # LayerNorm does not fix this: it normalises each token across channels and
        # leaves the component shared by every token untouched, which measured 88.3% of
        # the signal before and 93.3% after. The mean has to be removed across keys, and
        # only over valid ones so background tokens do not drag it.
        key_mask = valid[..., None].to(values.dtype)
        mean = (values * key_mask).sum(1, keepdim=True) / key_mask.sum(1, keepdim=True).clamp_min(1)
        values = values - mean
        values = values.reshape(batch, height * width, self.heads, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, values.to(query.dtype), attn_mask=valid[:, None, None, :], dropout_p=0.0,
        )
        if sampling_grid is None:
            sampling_grid = attention_sampling_grid(query, key, valid, height, width)
        warped = sample_attention_heads(
            values.to(query.dtype), sampling_grid, valid, height, width
        )
        attended = attended.transpose(1, 2).reshape(batch, height * width, -1)
        warped = warped.transpose(1, 2).reshape(batch, height * width, -1)
        attended = attended.transpose(1, 2).reshape(batch, -1, height, width)
        warped = warped.transpose(1, 2).reshape(batch, -1, height, width)
        features = warped + self.warp_mix(attended - warped)
        residual = self.output(features + self.local(features))
        gate = F.interpolate(edit.float(), (height, width), mode="area").clamp(0, 1)
        return residual * gate.to(residual.dtype) * active[:, None, None, None].to(residual.dtype)


class VTONPatchForcingBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        garment_scale=None,
        garment_attention_output_init_std=1e-3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.garment_scale = garment_scale
        self.use_garment_cross_attention = garment_scale is not None
        if self.use_garment_cross_attention:
            if garment_attention_output_init_std <= 0:
                raise ValueError("garment_attention_output_init_std must be positive")
            self.garment_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.garment_cross_attention = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=0.0,
                batch_first=True,
            )
            # A small non-zero residual keeps the pretrained backbone nearly unchanged
            # while allowing Q/K/V and garment embedders to learn from the first step.
            nn.init.normal_(
                self.garment_cross_attention.out_proj.weight,
                std=float(garment_attention_output_init_std),
            )
            nn.init.zeros_(self.garment_cross_attention.out_proj.bias)

    def forward(
        self,
        x,
        c,
        garment_tokens=None,
        garment_values=None,
        garment_padding_mask=None,
        edit_token_mask=None,
        return_attention=False,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(pf_modulate(self.norm1(x), shift_msa, scale_msa))
        attention = None
        if self.use_garment_cross_attention and garment_tokens is not None:
            # Position belongs in K, where it helps routing, but not in V: transporting
            # source coordinates together with garment appearance conflicts with the
            # content-only target used by the value loss and weakens logos/text.
            if garment_values is None:
                garment_values = garment_tokens
            # need_weights forces the unfused attention path, so it is requested only for
            # the blocks the correspondence loss actually supervises.
            cross, attention = self.garment_cross_attention(
                self.garment_norm(x),
                garment_tokens,
                garment_values,
                key_padding_mask=garment_padding_mask,
                need_weights=return_attention,
                # Keep heads separate for CORAL. Averaging here allowed a handful of
                # correctly routed heads to hide the majority attending elsewhere.
                average_attn_weights=False,
            )
            if edit_token_mask is not None:
                cross = cross * edit_token_mask[..., None].to(cross.dtype)
            x = x + cross
        x = x + gate_mlp * self.mlp(pf_modulate(self.norm2(x), shift_mlp, scale_mlp))
        if return_attention:
            if attention is None:
                raise RuntimeError("Attention weights were requested from a block without garment cross-attention")
            # Return the actual residual written by A @ V followed by out_proj.  Losses
            # on ``attention`` alone can train Q/K routing but cannot teach V/out_proj to
            # transport garment content such as logos and texture.
            return x, attention, cross
        return x


class VTONPatchForcingDiT(PatchForcingDiT):
    """PFT-XL with person conditioning and multiscale SD-VAE garment cross-attention.

    Garment appearance reaches the backbone through three branches, all of them VAE:

    ``coarse``  garment VAE latent, embedded by a copy of the pretrained patch projection
    ``middle``  VAE encoder 1/4-resolution feature map
    ``detail``  VAE encoder 1/2-resolution feature map, the finest appearance carrier

    A 4-channel DensePose VAE latent can be appended to the input projection.
    Optional target-cloth HF conditioning uses a separate zero-output velocity branch.

    Every branch's tokens are LayerNormed before the positional embedding is added
    (``garment_token_norm``), so key magnitude is set by the model rather than by the SD
    VAE's internal activation scale.

    There is no semantic (DINO) conditioning branch. Its former job -- establishing which
    garment region belongs at which body location -- is now supervised directly on the
    cross-attention maps by :mod:`patch_flow.correspondence`, using a frozen DINOv3
    teacher that exists only during training. That is strictly cheaper at inference and
    strictly more targeted: the correspondence signal lands on the attention distribution
    itself instead of being one more key set the model may or may not learn to use.
    """

    def __init__(
        self,
        *args,
        person_condition_channels=5,
        dense_pose_channels=0,
        garment_high_frequency_channels=0,
        use_vae_garment=True,
        garment_token_norm=True,
        garment_middle_channels=None,
        garment_detail_channels=None,
        garment_scale_routes=None,
        garment_embed_gain=1.0,
        garment_attention_output_init_std=1e-3,
        cross_attention_every=4,
        gradient_checkpointing=False,
        garment_match_query_grid=False,
        garment_latent_refiner=False,
        garment_refiner_width=256,
        garment_refiner_heads=8,
        garment_refiner_qk_norm=False,
        garment_refiner_cosine_scale=10.0,
        garment_refiner_local_radius=2,
        pretrained_ckpt=None,
        pretrained_use_ema=True,
        **kwargs,
    ):
        kwargs["compile"] = False
        super().__init__(*args, **kwargs)
        if not self.predict_uncertainty:
            raise ValueError("VTONPatchForcingDiT requires predict_uncertainty=True")
        if person_condition_channels != 5:
            raise ValueError("person_condition_channels must be 5: four agnostic latent channels and one mask")
        dense_pose_channels = int(dense_pose_channels)
        if dense_pose_channels not in (0, self.in_channels):
            raise ValueError(f"dense_pose_channels must be 0 or {self.in_channels} latent channels")
        garment_high_frequency_channels = int(garment_high_frequency_channels)
        if garment_high_frequency_channels < 0:
            raise ValueError("garment_high_frequency_channels must be non-negative")
        if cross_attention_every < 1:
            raise ValueError("cross_attention_every must be positive")

        old_embedder = self.x_embedder
        input_size = old_embedder.img_size[0]
        patch_size = old_embedder.patch_size[0]
        self.state_channels = self.in_channels
        self.person_condition_channels = person_condition_channels
        self.dense_pose_channels = dense_pose_channels
        self.garment_high_frequency_channels = garment_high_frequency_channels
        self.use_vae_garment = bool(use_vae_garment)
        self.garment_token_norm = bool(garment_token_norm)
        self.garment_middle_channels = None if garment_middle_channels is None else int(garment_middle_channels)
        self.garment_detail_channels = None if garment_detail_channels is None else int(garment_detail_channels)
        self.garment_embed_gain = float(garment_embed_gain)
        self.garment_attention_output_init_std = float(garment_attention_output_init_std)
        self.cross_attention_every = cross_attention_every
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            self.state_channels + person_condition_channels + dense_pose_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=False,
        )

        self.garment_embedder = None
        if self.use_vae_garment:
            self.garment_embedder = PatchEmbed(
                input_size,
                patch_size,
                self.state_channels,
                self.hidden_size,
                bias=True,
                strict_img_size=False,
            )
        self.garment_middle_embedder = None
        self.garment_detail_embedder = None
        if (self.garment_middle_channels is None) != (self.garment_detail_channels is None):
            raise ValueError("Both garment_middle_channels and garment_detail_channels must be configured")
        self.use_multiscale_garment = self.garment_middle_channels is not None
        if self.use_multiscale_garment:
            self.garment_middle_embedder = nn.Conv2d(
                self.garment_middle_channels, self.hidden_size, kernel_size=4, stride=4
            )
            self.garment_detail_embedder = nn.Conv2d(
                self.garment_detail_channels, self.hidden_size, kernel_size=4, stride=4
            )
            for embedder in (self.garment_middle_embedder, self.garment_detail_embedder):
                nn.init.xavier_uniform_(embedder.weight, gain=self.garment_embed_gain)
                nn.init.zeros_(embedder.bias)

        # Nothing else normalises the garment key/value path. The queries are LayerNormed
        # by ``garment_norm`` inside each block, but the keys and values are raw SD-VAE
        # encoder activations whose scale is set by the VAE's internals -- measured at
        # 512x384, per-token magnitudes were 25 (coarse), 143 (middle) and 68 (detail)
        # against LayerNormed queries at ~34, and darker fabric produced systematically
        # larger keys (navy 1.27-1.41x the garment mean), biasing every query toward it.
        # Normalising the content before the positional embedding is added also fixes the
        # content-to-position ratio, which was 0.86 for coarse (i.e. the coarse keys were
        # more than half positional) against 4.2 for middle.
        self.garment_token_norms = None
        if self.garment_token_norm:
            self.garment_token_norms = nn.ModuleDict(
                {scale: nn.LayerNorm(self.hidden_size, eps=1e-6) for scale in self._enabled_scales()}
            )

        with torch.no_grad():
            self.x_embedder.proj.weight.zero_()
            self.x_embedder.proj.weight[:, : self.state_channels].copy_(old_embedder.proj.weight)
            self.x_embedder.proj.bias.copy_(old_embedder.proj.bias)
            if self.garment_embedder is not None:
                self.garment_embedder.proj.weight.copy_(old_embedder.proj.weight)
                self.garment_embedder.proj.bias.copy_(old_embedder.proj.bias)

        old_blocks = self.blocks
        cross_attention_count = sum((index + 1) % cross_attention_every == 0 for index in range(len(old_blocks)))
        self.garment_scale_routes = self._resolve_routes(garment_scale_routes, cross_attention_count)
        self.blocks = nn.ModuleList()
        route_index = 0
        for index, old_block in enumerate(old_blocks):
            garment_scale = None
            if (index + 1) % cross_attention_every == 0:
                garment_scale = self.garment_scale_routes[route_index]
                route_index += 1
            block = VTONPatchForcingBlock(
                self.hidden_size,
                self.num_heads,
                garment_scale=garment_scale,
                garment_attention_output_init_std=self.garment_attention_output_init_std,
            )
            block.load_state_dict(old_block.state_dict(), strict=False)
            self.blocks.append(block)

        self.garment_match_query_grid = bool(garment_match_query_grid)
        self.garment_refiner = None
        if garment_latent_refiner:
            if not self.use_multiscale_garment:
                raise ValueError("The latent refiner requires multiscale garment features")
            self.garment_refiner = GarmentLatentRefiner(
                self.hidden_size, self.state_channels, garment_refiner_width,
                garment_refiner_heads, self.patch_size,
                qk_norm=garment_refiner_qk_norm, cosine_scale=garment_refiner_cosine_scale,
                dense_pose_channels=self.dense_pose_channels,
                local_radius=garment_refiner_local_radius,
            )

        self.garment_high_frequency_control = None
        if garment_high_frequency_channels:
            if self.garment_refiner is None:
                raise ValueError("HF control requires garment_latent_refiner for supervised spatial routing")
            self.garment_high_frequency_control = GarmentHighFrequencyControl(
                garment_refiner_width, garment_refiner_heads, self.state_channels,
                source_channels=garment_high_frequency_channels,
            )

        if pretrained_ckpt is not None:
            self.load_pretrained_checkpoint(pretrained_ckpt, use_ema=pretrained_use_ema)

    def _enabled_scales(self):
        enabled = []
        if self.use_vae_garment:
            enabled.append("coarse")
        if self.use_multiscale_garment:
            enabled.extend(("middle", "detail"))
        return enabled

    def _resolve_routes(self, garment_scale_routes, cross_attention_count):
        enabled = self._enabled_scales()
        if not enabled:
            raise ValueError("At least one garment conditioning branch must be enabled")
        if garment_scale_routes is None:
            garment_scale_routes = [enabled[0]] * cross_attention_count
        else:
            garment_scale_routes = list(garment_scale_routes)
        if len(garment_scale_routes) != cross_attention_count:
            raise ValueError(
                f"Expected {cross_attention_count} garment scale routes, got {len(garment_scale_routes)}"
            )
        unknown = set(garment_scale_routes) - set(GARMENT_SCALES)
        if unknown:
            raise ValueError(f"Garment scale routes must be one of {list(GARMENT_SCALES)}, got {sorted(unknown)}")
        disabled = set(garment_scale_routes) - set(enabled)
        if disabled:
            raise ValueError(f"Garment scale routes {sorted(disabled)} reference disabled branches")
        return tuple(garment_scale_routes)

    @staticmethod
    def _select_checkpoint_state(checkpoint, use_ema=True):
        state = checkpoint.get("state_dict", checkpoint)
        prefixes = ("ema_model.", "model.") if use_ema else ("model.", "ema_model.")
        for prefix in prefixes:
            selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
            if selected:
                return selected
        return state

    def load_pretrained_checkpoint(self, checkpoint_path, use_ema=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return self.load_pretrained_state_dict(self._select_checkpoint_state(checkpoint, use_ema=use_ema))

    def load_pretrained_state_dict(self, pretrained_state):
        current = self.state_dict()
        loaded = set()
        for key, value in pretrained_state.items():
            if key in current and current[key].shape == value.shape:
                current[key] = value
                loaded.add(key)

        weight_key = "x_embedder.proj.weight"
        bias_key = "x_embedder.proj.bias"
        if weight_key in pretrained_state:
            base_weight = pretrained_state[weight_key]
            if base_weight.shape[1] != self.state_channels:
                raise ValueError(f"Expected {self.state_channels} pretrained input channels, got {base_weight.shape[1]}")
            current[weight_key].zero_()
            current[weight_key][:, : self.state_channels] = base_weight
            if self.garment_embedder is not None:
                current["garment_embedder.proj.weight"] = base_weight.clone()
            loaded.add(weight_key)
        if bias_key in pretrained_state:
            current[bias_key] = pretrained_state[bias_key]
            if self.garment_embedder is not None:
                current["garment_embedder.proj.bias"] = pretrained_state[bias_key].clone()
            loaded.add(bias_key)

        ignored = {key for key in pretrained_state if key not in loaded}
        if ignored:
            raise RuntimeError(f"Could not transfer pretrained PFT parameters: {sorted(ignored)}")
        self.load_state_dict(current, strict=True)
        return self

    def _token_mask(self, mask, spatial_size):
        if mask is None:
            return None
        mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
        mask = F.max_pool2d(mask, kernel_size=self.patch_size, stride=self.patch_size)
        return mask.flatten(2).squeeze(1) > 0.5

    def _position_embedding(self, height, width, dtype, device):
        patch_height, patch_width = self.x_embedder.patch_size
        grid_height = height // patch_height
        grid_width = width // patch_width
        return self._grid_position_embedding(grid_height, grid_width, dtype, device)

    def _grid_position_embedding(self, grid_height, grid_width, dtype, device):
        base_height, base_width = self.x_embedder.grid_size
        if (grid_height, grid_width) == (base_height, base_width):
            return self.pos_embed.to(device=device, dtype=dtype)
        position = self.pos_embed.reshape(1, base_height, base_width, self.hidden_size).permute(0, 3, 1, 2)
        position = F.interpolate(position.float(), size=(grid_height, grid_width), mode="bicubic", align_corners=False)
        return position.permute(0, 2, 3, 1).flatten(1, 2).to(device=device, dtype=dtype)

    def _unpatchify_rectangular(self, tokens, height, width):
        patch_height, patch_width = self.x_embedder.patch_size
        grid_height = height // patch_height
        grid_width = width // patch_width
        if tokens.shape[1] != grid_height * grid_width:
            raise ValueError(f"Expected {grid_height * grid_width} output tokens, got {tokens.shape[1]}")
        tokens = tokens.reshape(
            tokens.shape[0], grid_height, grid_width, patch_height, patch_width, self.out_channels
        )
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(
            tokens.shape[0], self.out_channels, grid_height * patch_height, grid_width * patch_width
        )

    def _normalize_garment_tokens(self, scale, tokens):
        if self.garment_token_norms is None:
            return tokens
        return self.garment_token_norms[scale](tokens)

    def _garment_branches(self, garment, garment_middle, garment_detail, x, position, height, width):
        keys = {}
        values = {}
        grids = {}
        token_height = height // self.patch_size
        token_width = width // self.patch_size

        if garment is not None and self.garment_embedder is not None:
            latent = F.interpolate(garment, size=(height, width), mode="bilinear", align_corners=False)
            values["coarse"] = self._normalize_garment_tokens("coarse", self.garment_embedder(latent))
            keys["coarse"] = values["coarse"] + position
            grids["coarse"] = (token_height, token_width)

        for name, source, embedder in (
            ("middle", garment_middle, self.garment_middle_embedder),
            ("detail", garment_detail, self.garment_detail_embedder),
        ):
            if source is None:
                continue
            if embedder is None:
                raise ValueError(f"garment_{name}_channels must be configured to use the '{name}' branch")
            embedded = embedder(source)
            grid = (embedded.shape[-2], embedded.shape[-1])
            embedded = self._normalize_garment_tokens(name, embedded.flatten(2).transpose(1, 2))
            values[name] = embedded.to(x.dtype)
            keys[name] = values[name] + self._grid_position_embedding(*grid, x.dtype, x.device)
            grids[name] = grid
        return keys, values, grids

    def _garment_padding_masks(self, garment_mask, grids):
        padding = {}
        if garment_mask is None:
            return padding
        for name, (grid_height, grid_width) in grids.items():
            mask_size = (grid_height * self.patch_size, grid_width * self.patch_size)
            keep = self._token_mask(garment_mask, mask_size)
            if keep is None:
                continue
            empty = ~keep.any(dim=1)
            if empty.any():
                keep = keep.clone()
                keep[empty, 0] = True
            padding[name] = ~keep
        return padding

    def forward(
        self,
        x,
        t,
        y=None,
        person_agnostic=None,
        person_mask=None,
        dense_pose=None,
        garment_high_frequency=None,
        edit_mask=None,
        garment=None,
        garment_middle=None,
        garment_detail=None,
        garment_mask=None,
        return_uncertainty=False,
        return_garment_attention=False,
        garment_attention_scales=None,
        return_refiner_supervision=False,
    ):
        batch, _, height, width = x.shape
        if return_refiner_supervision and not return_garment_attention:
            raise ValueError("return_refiner_supervision requires return_garment_attention")
        if person_agnostic is None:
            person_agnostic = torch.zeros_like(x)
        if person_mask is None:
            person_mask = torch.zeros((batch, 1, height, width), device=x.device, dtype=x.dtype)
        person_agnostic = F.interpolate(person_agnostic, size=(height, width), mode="bilinear", align_corners=False)
        person_mask = F.interpolate(person_mask.float(), size=(height, width), mode="area").to(x.dtype)
        if self.dense_pose_channels:
            if dense_pose is None:
                raise ValueError("DensePose latent is required when dense_pose_channels is enabled")
            if dense_pose.ndim != 4 or dense_pose.shape[1] != self.dense_pose_channels:
                raise ValueError(
                    f"Expected DensePose latent with {self.dense_pose_channels} channels, got {tuple(dense_pose.shape)}"
                )
            dense_pose = F.interpolate(dense_pose, size=(height, width), mode="bilinear", align_corners=False)
            dense_pose = dense_pose.to(x.dtype)
        elif dense_pose is not None:
            raise ValueError("DensePose latent was provided to a model with dense_pose_channels=0")
        if self.garment_high_frequency_channels:
            if garment_high_frequency is None:
                raise ValueError(
                    "Target-cloth high-frequency map is required when "
                    "garment_high_frequency_channels is enabled"
                )
            if (garment_high_frequency.ndim != 4
                    or garment_high_frequency.shape[1] != self.garment_high_frequency_channels):
                raise ValueError(
                    f"Expected target-cloth high-frequency features with "
                    f"{self.garment_high_frequency_channels} channels, got "
                    f"{tuple(garment_high_frequency.shape)}"
                )
            if (garment_high_frequency.shape[0] != batch
                    or garment_high_frequency.shape[-2:] != (height * 4, width * 4)):
                raise ValueError(
                    "HF control requires VAE half-resolution features at 4x the latent grid"
                )
            garment_high_frequency = garment_high_frequency.to(x.dtype)
        elif garment_high_frequency is not None:
            raise ValueError(
                "Target-cloth high-frequency map was provided to a model with "
                "garment_high_frequency_channels=0"
            )
        if edit_mask is None:
            edit_mask = person_mask
        noisy_latent = x
        input_parts = (x, person_agnostic, person_mask)
        if self.dense_pose_channels:
            input_parts = (*input_parts, dense_pose)
        x = torch.cat(input_parts, dim=1)
        position = self._position_embedding(height, width, x.dtype, x.device)
        x = self.x_embedder(x) + position

        if t.ndim != 2 or t.shape[1] != x.shape[1]:
            raise ValueError(f"Expected per-token timesteps {(batch, x.shape[1])}, got {tuple(t.shape)}")
        cond = self.t_embedder(t[..., None]).squeeze(1)
        if self.y_embedder is not None:
            if y is None:
                y = torch.full((batch,), self.y_embedder.num_classes, device=x.device, dtype=torch.long)
            cond = cond + self.y_embedder(y, self.training)[:, None, :]

        garment_tokens, garment_values, garment_grids = self._garment_branches(
            garment, garment_middle, garment_detail, x, position, height, width
        )
        fine_values = garment_values.get("detail")
        if self.garment_refiner is not None and fine_values is not None:
            if garment_grids["detail"] != (height, width):
                raise ValueError("Person and garment detail grids must have identical resolution")
        if self.garment_match_query_grid:
            query_grid = (height // self.patch_size, width // self.patch_size)
            for scale, grid in garment_grids.items():
                if grid == query_grid:
                    continue
                content = garment_values[scale].transpose(1, 2).reshape(batch, self.hidden_size, *grid)
                content = F.adaptive_avg_pool2d(content, query_grid).flatten(2).transpose(1, 2)
                garment_values[scale] = content
                garment_tokens[scale] = content + position
                garment_grids[scale] = query_grid
        garment_padding_masks = self._garment_padding_masks(garment_mask, garment_grids)
        if garment_attention_scales is not None:
            garment_attention_scales = set(garment_attention_scales)

        edit_token_mask = self._token_mask(edit_mask, (height, width))
        attention_maps = []
        for index, block in enumerate(self.blocks, start=1):
            block_tokens = garment_tokens.get(block.garment_scale)
            block_values = garment_values.get(block.garment_scale)
            block_padding_mask = garment_padding_masks.get(block.garment_scale)
            want_attention = (
                return_garment_attention
                and block_tokens is not None
                and (garment_attention_scales is None or block.garment_scale in garment_attention_scales)
            )
            if self.gradient_checkpointing and self.training:
                output = checkpoint(
                    block,
                    x,
                    cond,
                    block_tokens,
                    block_values,
                    block_padding_mask,
                    edit_token_mask,
                    want_attention,
                    use_reentrant=False,
                )
            else:
                output = block(
                    x,
                    cond,
                    block_tokens,
                    block_values,
                    block_padding_mask,
                    edit_token_mask,
                    want_attention,
                )
            if want_attention:
                x, weights, transported_value = output
                attention_maps.append(
                    {
                        "block": index,
                        "scale": block.garment_scale,
                        "weights": weights,
                        "output": transported_value,
                        "grid": garment_grids[block.garment_scale],
                        "query_grid": (height // self.patch_size, width // self.patch_size),
                        "key_padding": block_padding_mask,
                    }
                )
            else:
                x = output
        # Decode the backbone first. HF alters that provisional flow, then the fine
        # branch sees and corrects the result. This replaces the old blind parallel sum
        # (backbone + fine + HF) with backbone -> HF -> fine refinement.
        prediction = self.final_layer(x, cond)
        prediction = self._unpatchify_rectangular(prediction, height, width)
        logvar_theta = prediction[:, -1:, :, :]
        velocity = prediction[:, :-1, :, :]
        if self.garment_high_frequency_control is not None and fine_values is None:
            raise ValueError("HF control requires garment detail features for spatial routing")
        if self.garment_refiner is not None and fine_values is not None:
            route_args = (
                x, noisy_latent, person_agnostic, fine_values,
                self._grid_position_embedding(height, width, x.dtype, x.device),
                garment_mask, dense_pose,
            )
            if self.gradient_checkpointing and self.training:
                fine_features, fine_entry, fine_active = checkpoint(
                    self.garment_refiner.route, *route_args, use_reentrant=False
                )
            else:
                fine_features, fine_entry, fine_active = self.garment_refiner.route(*route_args)
            if return_refiner_supervision:
                attention_maps.append(fine_entry)
            pre_hf_velocity = velocity
            if self.garment_high_frequency_control is not None:
                # Reuse the detail refiner's RGB/correspondence-supervised routing, but
                # do not let the auxiliary HF residual rewrite that routing.  The same
                # detached Q/K still produce exactly the same attention probabilities;
                # gradients remain enabled for the HF value encoder and output path.
                hf_args = (garment_high_frequency, fine_entry["query"].detach(),
                           fine_entry["key"].detach(),
                           fine_entry["key_valid"], edit_mask, garment_mask,
                           fine_entry["sampling_grid"].detach())
                if self.gradient_checkpointing and self.training:
                    hf_velocity = checkpoint(self.garment_high_frequency_control, *hf_args, use_reentrant=False)
                else:
                    hf_velocity = self.garment_high_frequency_control(*hf_args)
                velocity = velocity + hf_velocity
                if return_refiner_supervision:
                    # The trainer detaches pre_hf_velocity and gives this component a
                    # latent detail objective of its own. This prevents the backbone or
                    # final refiner from absorbing the logo loss while HF stays idle.
                    fine_entry["pre_hf_velocity"] = pre_hf_velocity
                    fine_entry["hf_velocity"] = hf_velocity

            token_height = height // self.patch_size
            token_width = width // self.patch_size
            time_latent = t.reshape(batch, 1, token_height, token_width)
            time_latent = time_latent.repeat_interleave(
                self.patch_size, -2
            ).repeat_interleave(self.patch_size, -1).to(velocity.dtype)
            preliminary_clean = noisy_latent + (1 - time_latent) * velocity
            refine_args = (
                fine_features, velocity.detach(), preliminary_clean.detach(),
                edit_mask, fine_active,
            )
            if self.gradient_checkpointing and self.training:
                fine_velocity = checkpoint(
                    self.garment_refiner.refine, *refine_args, use_reentrant=False
                )
            else:
                fine_velocity = self.garment_refiner.refine(*refine_args)
            velocity = velocity + fine_velocity
        if return_garment_attention:
            if return_uncertainty:
                return velocity, logvar_theta, attention_maps
            return velocity, attention_maps
        if return_uncertainty:
            return velocity, logvar_theta
        return velocity
