"""StableVITON-inspired garment-masked attention-center total variation.

Reference: https://openaccess.thecvf.com/content/CVPR2024/html/Kim_StableVITON_Learning_Semantic_Correspondence_with_Latent_Diffusion_Model_for_Virtual_CVPR_2024_paper.html
Adaptation: expected (probability-normalized) key coordinates, with both endpoints
masked. This avoids pulling garment boundaries toward the zero/background center.
It regularizes correspondence geometry, not RGB or attention-map sharpness.
"""
import torch

from patch_flow.vton_utils import masked_mean


def key_coordinates(grid, device, dtype=torch.float32):
    height, width = grid
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def attention_centers(attention, key_grid):
    """Average heads first, returning B,Q,2 without an FP32 full-head copy."""
    if attention.ndim == 4:
        attention = attention.mean(1)
    if attention.ndim != 3 or attention.shape[-1] != key_grid[0] * key_grid[1]:
        raise ValueError("Attention shape does not match garment key grid")
    coordinates = key_coordinates(key_grid, attention.device, attention.dtype)
    centers = (attention @ coordinates).float()
    return centers / attention.float().sum(-1, keepdim=True).clamp_min(1e-6)


def masked_center_tv(centers, query_grid, mask):
    """First differences inside garment only, normalized by valid edge count."""
    height, width = query_grid
    if centers.shape[1:] != (height * width, 2) or mask.shape != (centers.shape[0], 1, height, width):
        raise ValueError("Center coordinates and garment mask must match the person query grid")
    field = centers.float().transpose(1, 2).reshape(-1, 2, height, width)
    loss = field.sum() * 0.0
    if width > 1:
        loss = loss + masked_mean((field[:, :, :, 1:] - field[:, :, :, :-1]).abs(),
                                  mask[:, :, :, 1:] * mask[:, :, :, :-1])
    if height > 1:
        loss = loss + masked_mean((field[:, :, 1:, :] - field[:, :, :-1, :]).abs(),
                                  mask[:, :, 1:, :] * mask[:, :, :-1, :])
    return loss
