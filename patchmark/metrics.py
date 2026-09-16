from __future__ import annotations
import torch
import torch.nn.functional as F

def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-coordinates ** 2 / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()

def psnr(original: torch.Tensor, reconstruction: torch.Tensor, data_range: float=1.0) -> torch.Tensor:
    if original.shape != reconstruction.shape:
        raise ValueError('PSNR inputs must have equal shapes')
    mse = F.mse_loss(reconstruction, original, reduction='none').flatten(1).mean(1)
    return 10.0 * torch.log10(data_range ** 2 / torch.clamp(mse, min=1e-12))

def ssim(original: torch.Tensor, reconstruction: torch.Tensor, *, window_size: int=11, sigma: float=1.5, data_range: float=1.0) -> torch.Tensor:
    if original.shape != reconstruction.shape or original.ndim != 4:
        raise ValueError(f'SSIM expects equal NCHW tensors, got {tuple(original.shape)} and {tuple(reconstruction.shape)}')
    channels = original.shape[1]
    window = _gaussian_window(window_size, sigma, channels, original.device, original.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(original, window, padding=padding, groups=channels)
    mu_y = F.conv2d(reconstruction, window, padding=padding, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = (mu_x.square(), mu_y.square(), mu_x * mu_y)
    sigma_x_sq = F.conv2d(original.square(), window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(reconstruction.square(), window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(original * reconstruction, window, padding=padding, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = (2 * mu_xy + c1) * (2 * sigma_xy + c2) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return score.flatten(1).mean(1)
