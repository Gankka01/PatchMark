from __future__ import annotations
from functools import lru_cache
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _activation_checkpoint
from .runtime import assert_tensor_dtype_fp32

@lru_cache(maxsize=32)
def _gaussian_kernel_1d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-coordinates ** 2 / (2 * sigma ** 2))
    return gaussian / gaussian.sum()

def _gaussian_blur(values: torch.Tensor, window_size: int, sigma: float) -> torch.Tensor:
    channels = int(values.shape[1])
    kernel = _gaussian_kernel_1d(window_size, sigma, values.device, values.dtype)
    horizontal = kernel.view(1, 1, 1, window_size).expand(channels, 1, 1, -1).contiguous()
    vertical = kernel.view(1, 1, window_size, 1).expand(channels, 1, -1, 1).contiguous()
    padding = window_size // 2
    values = F.conv2d(values, horizontal, padding=(0, padding), groups=channels)
    return F.conv2d(values, vertical, padding=(padding, 0), groups=channels)

def _x_dependent_moments(x: torch.Tensor, y: torch.Tensor, window_size: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (_gaussian_blur(x, window_size, sigma), _gaussian_blur(x.square(), window_size, sigma), _gaussian_blur(x * y, window_size, sigma))

def _memory_efficient_x_moments(x: torch.Tensor, y: torch.Tensor, window_size: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_grad_enabled() or not x.requires_grad:
        return _x_dependent_moments(x, y, window_size, sigma)

    def moments(trainable: torch.Tensor, reference: torch.Tensor):
        return _x_dependent_moments(trainable, reference, window_size, sigma)
    return _activation_checkpoint(moments, x, y, use_reentrant=False, preserve_rng_state=False)

def ssim_index(x: torch.Tensor, y: torch.Tensor, window_size: int=11, sigma: float=1.5, data_range: float=1.0) -> torch.Tensor:
    assert_tensor_dtype_fp32('ssim.x', x, allow_nonfloating=False)
    assert_tensor_dtype_fp32('ssim.y', y, allow_nonfloating=False)
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError(f'SSIM expects equal NCHW tensors, got {tuple(x.shape)} and {tuple(y.shape)}')
    mu_x, second_x, cross_xy = _memory_efficient_x_moments(x, y, window_size, sigma)
    mu_y = _gaussian_blur(y, window_size, sigma)
    second_y = _gaussian_blur(y.square(), window_size, sigma)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = second_x - mu_x_sq
    sigma_y_sq = second_y - mu_y_sq
    sigma_xy = cross_xy - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = (2 * mu_xy + c1) * (2 * sigma_xy + c2) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return score.mean()

def image_quality_loss(original: torch.Tensor, watermarked: torch.Tensor, alpha_ssim: float=0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert_tensor_dtype_fp32('quality.original', original, allow_nonfloating=False)
    assert_tensor_dtype_fp32('quality.watermarked', watermarked, allow_nonfloating=False)
    mse = F.mse_loss(watermarked, original)
    ssim = ssim_index(watermarked, original)
    loss = mse + alpha_ssim * (1.0 - ssim)
    return (loss, {'mse': mse, 'ssim': ssim, 'image_quality': loss})
