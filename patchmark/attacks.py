from __future__ import annotations
import math
from hashlib import sha256
from dataclasses import asdict, dataclass
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
ATTACK_NAMES = ('jpeg_compression', 'brightness', 'contrast', 'saturation', 'gaussian_blur', 'gaussian_noise', 'posterize', 'color_jiggle', 'rgb_shift', 'horizontal_flip', 'rotation', 'random_erasing')
ATTACK_IMPLEMENTATION_BACKEND = 'custom_differentiable_pytorch'

@dataclass(frozen=True)
class Stage3AttackConfig:
    jpeg_quality: int = 30
    jpeg_subsampling: str = '4:2:0'
    brightness_min: float = 0.75
    brightness_max: float = 1.25
    contrast_min: float = 0.75
    contrast_max: float = 1.25
    saturation_min: float = 0.75
    saturation_max: float = 1.25
    gaussian_blur_kernel: int = 5
    gaussian_blur_sigma_min: float = 0.1
    gaussian_blur_sigma_max: float = 1.5
    gaussian_noise_std: float = 0.04
    posterize_bits: int = 4
    color_jiggle_brightness: float = 0.15
    color_jiggle_contrast: float = 0.15
    color_jiggle_saturation: float = 0.15
    color_jiggle_hue: float = 0.02
    rgb_shift_min: float = -0.05
    rgb_shift_max: float = 0.05
    horizontal_flip_probability: float = 1.0
    rotation_min_degrees: float = -5.0
    rotation_max_degrees: float = 5.0
    random_erasing_scale_min: float = 0.02
    random_erasing_scale_max: float = 0.05
    random_erasing_ratio_min: float = 0.5
    random_erasing_ratio_max: float = 1.5

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> 'Stage3AttackConfig':
        if not values:
            config = cls()
        else:
            unknown = set(values) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(f'Unknown Stage-3 configuration keys: {sorted(unknown)}')
            config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        float_parameters = ('brightness_min', 'brightness_max', 'contrast_min', 'contrast_max', 'saturation_min', 'saturation_max', 'gaussian_blur_sigma_min', 'gaussian_blur_sigma_max', 'gaussian_noise_std', 'color_jiggle_brightness', 'color_jiggle_contrast', 'color_jiggle_saturation', 'color_jiggle_hue', 'rgb_shift_min', 'rgb_shift_max', 'horizontal_flip_probability', 'rotation_min_degrees', 'rotation_max_degrees', 'random_erasing_scale_min', 'random_erasing_scale_max', 'random_erasing_ratio_min', 'random_erasing_ratio_max')
        nonfinite = [name for name in float_parameters if not isinstance(getattr(self, name), (int, float)) or isinstance(getattr(self, name), bool) or (not math.isfinite(float(getattr(self, name))))]
        if nonfinite:
            raise ValueError(f'Stage-3 floating parameters must be finite numbers: {nonfinite}')
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError('jpeg_quality must be in [1,100]')
        if self.jpeg_subsampling not in {'4:2:0', '4:4:4'}:
            raise ValueError("jpeg_subsampling must be '4:2:0' or '4:4:4'")
        for name in ('brightness', 'contrast', 'saturation'):
            low, high = (getattr(self, f'{name}_min'), getattr(self, f'{name}_max'))
            if not 0 <= low <= high:
                raise ValueError(f'Invalid {name} range: {low}, {high}')
        if self.gaussian_blur_kernel <= 0 or self.gaussian_blur_kernel % 2 == 0:
            raise ValueError('gaussian_blur_kernel must be a positive odd integer')
        if not 0 < self.gaussian_blur_sigma_min <= self.gaussian_blur_sigma_max:
            raise ValueError('Invalid Gaussian blur sigma range')
        if self.gaussian_noise_std < 0:
            raise ValueError('gaussian_noise_std must be nonnegative')
        for name in ('color_jiggle_brightness', 'color_jiggle_contrast', 'color_jiggle_saturation', 'color_jiggle_hue'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if not 1 <= self.posterize_bits <= 8:
            raise ValueError('posterize_bits must be in [1,8]')
        if self.horizontal_flip_probability != 1.0:
            raise ValueError('The approved horizontal-flip operator requires probability 1.0')
        if self.rotation_min_degrees > self.rotation_max_degrees:
            raise ValueError('Invalid rotation range')
        if not 0 < self.random_erasing_scale_min <= self.random_erasing_scale_max <= 1:
            raise ValueError('Invalid random-erasing scale range')
        if not 0 < self.random_erasing_ratio_min <= self.random_erasing_ratio_max:
            raise ValueError('Invalid random-erasing ratio range')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()

def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == 'cuda' else 'cpu')
    generator.manual_seed(int(seed))
    return generator

def _rand(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype, generator: torch.Generator, low: float=0.0, high: float=1.0) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype, generator=generator) * (high - low) + low

def _samplewise_rand(seeds: list[int], draws: int, *, device: torch.device, dtype: torch.dtype, low: float=0.0, high: float=1.0) -> torch.Tensor:
    if draws <= 0:
        raise ValueError('draws must be positive')
    rows: list[torch.Tensor] = []
    for seed in seeds:
        generator = _device_generator(device, int(seed))
        rows.append(_rand((draws,), device=device, dtype=dtype, generator=generator, low=low, high=high))
    if not rows:
        return torch.empty((0, draws), device=device, dtype=dtype)
    return torch.stack(rows, dim=0)

def _samplewise_randn_like(images: torch.Tensor, seeds: list[int]) -> torch.Tensor:
    if len(seeds) != images.shape[0]:
        raise ValueError('seeds and images batch dimensions differ')
    noise = torch.empty(tuple(images.shape), device=images.device, dtype=images.dtype)
    for index, seed in enumerate(seeds):
        generator = _device_generator(images.device, int(seed))
        noise[index].normal_(generator=generator)
    return noise

def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image.unbind(1)
    maxc, max_index = image.max(dim=1)
    minc = image.min(dim=1).values
    delta = maxc - minc
    safe_delta = torch.where(delta > 1e-12, delta, torch.ones_like(delta))
    safe_maxc = torch.where(maxc > 1e-12, maxc, torch.ones_like(maxc))
    h_r = torch.remainder((g - b) / safe_delta, 6.0)
    h_g = (b - r) / safe_delta + 2.0
    h_b = (r - g) / safe_delta + 4.0
    h = torch.where(max_index == 0, h_r, torch.where(max_index == 1, h_g, h_b)) / 6.0
    h = torch.where(delta > 1e-12, torch.remainder(h, 1.0), torch.zeros_like(h))
    s = torch.where(maxc > 1e-12, delta / safe_maxc, torch.zeros_like(maxc))
    return torch.stack([h, s, maxc], dim=1)

def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv.unbind(1)
    h6 = torch.remainder(h, 1.0) * 6.0
    index = torch.floor(h6).long() % 6
    f = h6 - torch.floor(h6)
    p, q, t = (v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s))
    sector_0 = index == 0
    sector_1 = index == 1
    sector_2 = index == 2
    sector_3 = index == 3
    sector_4 = index == 4
    sector_5 = index == 5
    red = torch.where(sector_0 | sector_5, v, torch.where(sector_1, q, torch.where(sector_2 | sector_3, p, t)))
    green = torch.where(sector_0, t, torch.where(sector_1 | sector_2, v, torch.where(sector_3, q, p)))
    blue = torch.where(sector_0 | sector_1, p, torch.where(sector_2, t, torch.where(sector_3 | sector_4, v, q)))
    return torch.stack((red, green, blue), dim=1)

def _gaussian_blur_batch(images: torch.Tensor, sigmas: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if sigmas.shape != (images.shape[0],):
        raise ValueError('sigmas must have shape [B]')
    coords = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    sigma = sigmas.clamp_min(0.001).view(-1, 1)
    kernel_1d = torch.exp(-coords.view(1, -1) ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum(1, keepdim=True)
    kernels = kernel_1d[:, :, None] * kernel_1d[:, None, :]
    batch, channels, height, width = images.shape
    weights = kernels[:, None].expand(batch, channels, kernel_size, kernel_size)
    flattened = images.reshape(1, batch * channels, height, width)
    padding = kernel_size // 2
    flattened = F.pad(flattened, (padding, padding, padding, padding), mode='reflect')
    output = F.conv2d(flattened, weights.reshape(batch * channels, 1, kernel_size, kernel_size), padding=0, groups=batch * channels)
    return output.reshape(batch, channels, height, width)

class DifferentiableJPEG(nn.Module):
    _LUMA = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
    _CHROMA = torch.tensor([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], dtype=torch.float32)

    def __init__(self, quality: int=30, subsampling: str='4:2:0'):
        super().__init__()
        if not 1 <= int(quality) <= 100:
            raise ValueError('JPEG quality must be in [1,100]')
        if subsampling not in {'4:2:0', '4:4:4'}:
            raise ValueError("subsampling must be '4:2:0' or '4:4:4'")
        self.quality = int(quality)
        self.subsampling = subsampling
        dct = torch.empty(8, 8, dtype=torch.float32)
        for k in range(8):
            alpha = math.sqrt(1 / 8) if k == 0 else math.sqrt(2 / 8)
            for n in range(8):
                dct[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / 16)
        self.register_buffer('dct', dct, persistent=False)
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        self.register_buffer('luma_quant', torch.floor((self._LUMA * scale + 50) / 100).clamp(1, 255), persistent=False)
        self.register_buffer('chroma_quant', torch.floor((self._CHROMA * scale + 50) / 100).clamp(1, 255), persistent=False)

    @staticmethod
    def _rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
        r, g, b = image.unbind(1)
        return torch.stack([0.299 * r + 0.587 * g + 0.114 * b, -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5, 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5], 1)

    @staticmethod
    def _ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
        y, cb, cr = image.unbind(1)
        cb, cr = (cb - 0.5, cr - 0.5)
        return torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], 1)

    def _block_codec(self, component: torch.Tensor, quant: torch.Tensor) -> torch.Tensor:
        if component.ndim != 4:
            raise ValueError('component must be [B,C,H,W]')
        height, width = component.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError('component dimensions must be divisible by 8')
        x = component * 255.0 - 128.0
        batch, channels = x.shape[:2]
        blocks = x.view(batch, channels, height // 8, 8, width // 8, 8).permute(0, 1, 2, 4, 3, 5)
        dct = self.dct.to(dtype=x.dtype)
        coeff = torch.matmul(torch.matmul(dct, blocks), dct.T)
        q = quant.to(dtype=x.dtype).view(1, 1, 1, 1, 8, 8)
        coeff = _ste_round(coeff / q) * q
        blocks = torch.matmul(torch.matmul(dct.T, coeff), dct)
        return (blocks.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, height, width) + 128.0) / 255.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f'Expected [B,3,H,W], got {tuple(image.shape)}')
        original_h, original_w = image.shape[-2:]
        multiple = 16 if self.subsampling == '4:2:0' else 8
        pad_h, pad_w = (-original_h % multiple, -original_w % multiple)
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
        ycbcr = self._rgb_to_ycbcr(padded)
        y = self._block_codec(ycbcr[:, 0:1], self.luma_quant)
        chroma = ycbcr[:, 1:3]
        if self.subsampling == '4:2:0':
            chroma = F.avg_pool2d(chroma, kernel_size=2, stride=2)
        chroma = self._block_codec(chroma, self.chroma_quant)
        if self.subsampling == '4:2:0':
            chroma = F.interpolate(chroma, size=y.shape[-2:], mode='bilinear', align_corners=False)
        decoded = self._ycbcr_to_rgb(torch.cat([y, chroma], 1))
        return decoded[..., :original_h, :original_w].clamp(0.0, 1.0)

@dataclass
class AttackResult:
    images: torch.Tensor
    names: list[str]
    parameters: list[dict[str, Any]]

def _assemble_attack_groups(reference: torch.Tensor, attacked_groups: list[torch.Tensor], selected_groups: list[torch.Tensor]) -> torch.Tensor:
    if not attacked_groups or len(attacked_groups) != len(selected_groups):
        raise ValueError('attack groups and selected indices must be nonempty and aligned')
    if sum((int(indices.numel()) for indices in selected_groups)) != reference.shape[0]:
        raise ValueError('grouped attack indices do not cover the batch')
    output = torch.empty_like(reference)
    for attacked, selected in zip(attacked_groups, selected_groups, strict=True):
        if attacked.shape[0] != selected.numel() or attacked.shape[1:] != reference.shape[1:]:
            raise ValueError('attack group shape differs from its selected indices')
        output.index_copy_(0, selected, attacked)
    return output

class Stage3AttackSuite(nn.Module):
    names = ATTACK_NAMES
    implementation_backend = ATTACK_IMPLEMENTATION_BACKEND

    def __init__(self, jpeg_quality: int | Stage3AttackConfig | None=None, *, config: Stage3AttackConfig | None=None):
        super().__init__()
        if isinstance(jpeg_quality, Stage3AttackConfig):
            if config is not None:
                raise ValueError('Pass Stage3AttackConfig either positionally or via config=, not both')
            config = jpeg_quality
            jpeg_quality = None
        if config is None:
            config = Stage3AttackConfig(jpeg_quality=30 if jpeg_quality is None else int(jpeg_quality))
        elif jpeg_quality is not None and int(jpeg_quality) != config.jpeg_quality:
            raise ValueError('Pass either config or jpeg_quality, not conflicting values')
        config.validate()
        self.config = config
        self.jpeg = DifferentiableJPEG(config.jpeg_quality, config.jpeg_subsampling)
        self.jpeg_quality = int(config.jpeg_quality)

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> 'Stage3AttackSuite':
        return cls(config=Stage3AttackConfig.from_mapping(values))

    @staticmethod
    def _empty_parameters(batch: int) -> list[dict[str, Any]]:
        return [{} for _ in range(batch)]

    def apply_named(self, images: torch.Tensor, name: str, *, seed: int, record_parameters: bool=False, sample_seeds: list[int] | None=None) -> AttackResult:
        if name not in self.names:
            raise KeyError(f'Unknown attack: {name}')
        batch, _, height, width = images.shape
        if sample_seeds is not None and len(sample_seeds) != batch:
            raise ValueError('sample_seeds and images batch dimensions differ')
        generator = None if sample_seeds is not None else _device_generator(images.device, seed)
        parameters = self._empty_parameters(batch)

        def random_values(draws: int, *, low: float=0.0, high: float=1.0) -> torch.Tensor:
            if sample_seeds is not None:
                return _samplewise_rand(sample_seeds, draws, device=images.device, dtype=images.dtype, low=low, high=high)
            assert generator is not None
            return _rand((batch, draws), device=images.device, dtype=images.dtype, generator=generator, low=low, high=high)
        if name == 'jpeg_compression':
            output = self.jpeg(images)
            if record_parameters:
                parameters = [{'quality': self.jpeg_quality, 'subsampling': self.config.jpeg_subsampling} for _ in range(batch)]
        elif name == 'brightness':
            factors = random_values(1, low=self.config.brightness_min, high=self.config.brightness_max).view(batch, 1, 1, 1)
            output = images * factors
            if record_parameters:
                parameters = [{'factor': float(v)} for v in factors.detach().cpu().flatten()]
        elif name == 'contrast':
            factors = random_values(1, low=self.config.contrast_min, high=self.config.contrast_max).view(batch, 1, 1, 1)
            mean = images.mean((2, 3), keepdim=True)
            output = (images - mean) * factors + mean
            if record_parameters:
                parameters = [{'factor': float(v)} for v in factors.detach().cpu().flatten()]
        elif name == 'saturation':
            factors = random_values(1, low=self.config.saturation_min, high=self.config.saturation_max).view(batch, 1, 1, 1)
            gray = images[:, 0:1] * 0.299 + images[:, 1:2] * 0.587 + images[:, 2:3] * 0.114
            output = gray + factors * (images - gray)
            if record_parameters:
                parameters = [{'factor': float(v)} for v in factors.detach().cpu().flatten()]
        elif name == 'gaussian_blur':
            sigmas = random_values(1, low=self.config.gaussian_blur_sigma_min, high=self.config.gaussian_blur_sigma_max).view(batch)
            output = _gaussian_blur_batch(images, sigmas, self.config.gaussian_blur_kernel)
            if record_parameters:
                parameters = [{'kernel': self.config.gaussian_blur_kernel, 'sigma': float(v)} for v in sigmas.detach().cpu()]
        elif name == 'gaussian_noise':
            if sample_seeds is None:
                assert generator is not None
                noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
            else:
                noise = _samplewise_randn_like(images, sample_seeds)
            noise.mul_(self.config.gaussian_noise_std)
            output = images + noise
            if record_parameters:
                parameters = [{'std': self.config.gaussian_noise_std} for _ in range(batch)]
        elif name == 'posterize':
            levels = 2 ** self.config.posterize_bits - 1
            output = _ste_round(images * levels) / levels
            if record_parameters:
                parameters = [{'bits': self.config.posterize_bits} for _ in range(batch)]
        elif name == 'color_jiggle':
            unit = random_values(4)
            b = (1.0 - self.config.color_jiggle_brightness + unit[:, 0] * (2.0 * self.config.color_jiggle_brightness)).view(batch, 1, 1, 1)
            c = (1.0 - self.config.color_jiggle_contrast + unit[:, 1] * (2.0 * self.config.color_jiggle_contrast)).view(batch, 1, 1, 1)
            s = (1.0 - self.config.color_jiggle_saturation + unit[:, 2] * (2.0 * self.config.color_jiggle_saturation)).view(batch, 1, 1, 1)
            hue = (-self.config.color_jiggle_hue + unit[:, 3] * (2.0 * self.config.color_jiggle_hue)).view(batch)
            output = images * b
            mean = output.mean((2, 3), keepdim=True)
            output = (output - mean) * c + mean
            gray = output[:, 0:1] * 0.299 + output[:, 1:2] * 0.587 + output[:, 2:3] * 0.114
            output = gray + s * (output - gray)
            hsv = _rgb_to_hsv(output.clamp(0, 1))
            hsv = torch.cat([torch.remainder(hsv[:, 0:1] + hue[:, None, None, None], 1.0), hsv[:, 1:]], 1)
            output = _hsv_to_rgb(hsv)
            if record_parameters:
                bv, cv, sv, hv = [x.detach().cpu().flatten() for x in (b, c, s, hue)]
                parameters = [{'brightness': float(bv[i]), 'contrast': float(cv[i]), 'saturation': float(sv[i]), 'hue': float(hv[i])} for i in range(batch)]
        elif name == 'rgb_shift':
            shifts = random_values(3, low=self.config.rgb_shift_min, high=self.config.rgb_shift_max).view(batch, 3, 1, 1)
            output = images + shifts
            if record_parameters:
                parameters = [{'shift': [float(v) for v in row]} for row in shifts.detach().cpu()[:, :, 0, 0]]
        elif name == 'horizontal_flip':
            output = torch.flip(images, (-1,))
            if record_parameters:
                parameters = [{'probability': 1.0} for _ in range(batch)]
        elif name == 'rotation':
            angles = random_values(1, low=self.config.rotation_min_degrees, high=self.config.rotation_max_degrees).view(batch)
            radians = -angles * math.pi / 180.0
            theta = torch.zeros(batch, 2, 3, device=images.device, dtype=images.dtype)
            theta[:, 0, 0], theta[:, 0, 1] = (torch.cos(radians), -torch.sin(radians))
            theta[:, 1, 0], theta[:, 1, 1] = (torch.sin(radians), torch.cos(radians))
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            output = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            if record_parameters:
                parameters = [{'angle_degrees': float(v)} for v in angles.detach().cpu()]
        elif name == 'random_erasing':
            unit = random_values(4)
            scale = self.config.random_erasing_scale_min + unit[:, 0] * (self.config.random_erasing_scale_max - self.config.random_erasing_scale_min)
            ratio = self.config.random_erasing_ratio_min + unit[:, 1] * (self.config.random_erasing_ratio_max - self.config.random_erasing_ratio_min)
            erase_h = torch.sqrt(scale * height * width / ratio).round().clamp(1, height).long()
            erase_w = torch.sqrt(scale * height * width * ratio).round().clamp(1, width).long()
            top_frac, left_frac = (unit[:, 2], unit[:, 3])
            top = (top_frac * (height - erase_h + 1).to(top_frac.dtype)).floor().long()
            left = (left_frac * (width - erase_w + 1).to(left_frac.dtype)).floor().long()
            yy = torch.arange(height, device=images.device)[None, :, None]
            xx = torch.arange(width, device=images.device)[None, None, :]
            erased = (yy >= top[:, None, None]) & (yy < (top + erase_h)[:, None, None]) & (xx >= left[:, None, None]) & (xx < (left + erase_w)[:, None, None])
            output = images * (~erased[:, None]).to(images.dtype)
            if record_parameters:
                tv, lv, hv, wv = [x.detach().cpu().tolist() for x in (top, left, erase_h, erase_w)]
                parameters = [{'top': tv[i], 'left': lv[i], 'height': hv[i], 'width': wv[i]} for i in range(batch)]
        else:
            raise AssertionError(name)
        return AttackResult(output.clamp(0.0, 1.0), [name] * batch, parameters)

    @staticmethod
    def transform_module_targets(module_targets: torch.Tensor, attack_names: list[str]) -> torch.Tensor:
        if module_targets.ndim != 4 or module_targets.shape[1] != 1:
            raise ValueError(f'Expected module targets [B,1,H,W], got {tuple(module_targets.shape)}')
        if len(attack_names) != module_targets.shape[0]:
            raise ValueError('attack_names batch size does not match module targets')
        flip_mask = torch.tensor([name == 'horizontal_flip' for name in attack_names], device=module_targets.device, dtype=torch.bool)
        if not bool(flip_mask.any()):
            return module_targets
        transformed = module_targets.clone()
        transformed[flip_mask] = torch.flip(module_targets[flip_mask], dims=(-1,))
        return transformed

    @staticmethod
    def sample_seed(sample_id: str, *, seed: int, context: str) -> int:
        digest = sha256(f'patchmark-stage3-v1|{sample_id}|{int(seed)}|{context}'.encode('utf-8')).digest()
        return int.from_bytes(digest[:8], 'big') % (2 ** 31 - 1)

    def forward_preplanned(self, images: torch.Tensor, *, assignments: torch.Tensor, parameter_seeds: torch.Tensor, record_parameters: bool=False, record_names: bool=True) -> AttackResult:
        if assignments.device.type != 'cpu' or parameter_seeds.device.type != 'cpu':
            raise TypeError('preplanned attack metadata must remain on CPU')
        assignments = assignments.reshape(-1).to(dtype=torch.long)
        parameter_seeds = parameter_seeds.reshape(-1).to(dtype=torch.long)
        if assignments.numel() != images.shape[0] or parameter_seeds.numel() != images.shape[0]:
            raise ValueError('preplanned attack metadata batch size mismatch')
        if images.shape[0] == 0:
            raise ValueError('Stage-3 preplanned batch must not be empty')
        attacked_groups: list[torch.Tensor] = []
        selected_groups: list[torch.Tensor] = []
        names = [''] * images.shape[0] if record_names else []
        parameters = self._empty_parameters(images.shape[0]) if record_parameters else []
        planned_groups: list[tuple[str, list[int], list[int]]] = []
        for attack_index, attack_name in enumerate(self.names):
            selected_cpu = torch.nonzero(assignments == attack_index, as_tuple=False).flatten().tolist()
            if not selected_cpu:
                continue
            seeds = [int(parameter_seeds[index]) for index in selected_cpu]
            planned_groups.append((attack_name, selected_cpu, seeds))
        flat_indices = [index for _, selected_cpu, _ in planned_groups for index in selected_cpu]
        packed_cpu = torch.tensor(flat_indices, dtype=torch.long, pin_memory=images.device.type == 'cuda')
        packed = packed_cpu.to(device=images.device, non_blocking=images.device.type == 'cuda')
        offset = 0
        for attack_name, selected_cpu, seeds in planned_groups:
            selected = packed[offset:offset + len(selected_cpu)]
            offset += len(selected_cpu)
            attacked = self.apply_named(images.index_select(0, selected), attack_name, seed=0, record_parameters=record_parameters, sample_seeds=seeds)
            attacked_groups.append(attacked.images)
            selected_groups.append(selected)
            for local_index, global_index in enumerate(selected_cpu):
                if record_names:
                    names[global_index] = attack_name
                if record_parameters:
                    parameters[global_index] = attacked.parameters[local_index]
        output = _assemble_attack_groups(images, attacked_groups, selected_groups)
        return AttackResult(output, names, parameters)

    def forward_sample_stable(self, images: torch.Tensor, *, sample_ids: list[str], seed: int, context: str, record_parameters: bool=False, record_names: bool=True) -> AttackResult:
        if images.ndim != 4:
            raise ValueError(f'Expected images [B,C,H,W], got {tuple(images.shape)}')
        if len(sample_ids) != images.shape[0]:
            raise ValueError('sample_ids and images batch dimensions differ')
        if not sample_ids:
            raise ValueError('sample_ids must not be empty')
        sample_seeds = [self.sample_seed(str(sample_id), seed=seed, context=context) for sample_id in sample_ids]
        assignment_values = [value % len(self.names) for value in sample_seeds]
        parameter_seeds = [(value * 1000003 + 104729) % (2 ** 31 - 1) for value in sample_seeds]
        attacked_groups: list[torch.Tensor] = []
        selected_groups: list[torch.Tensor] = []
        names = [''] * images.shape[0] if record_names else []
        parameters = self._empty_parameters(images.shape[0]) if record_parameters else []
        for attack_index, attack_name in enumerate(self.names):
            selected_cpu = [i for i, value in enumerate(assignment_values) if value == attack_index]
            if not selected_cpu:
                continue
            selected = torch.tensor(selected_cpu, device=images.device, dtype=torch.long)
            selected_seeds = [parameter_seeds[index] for index in selected_cpu]
            attacked = self.apply_named(images.index_select(0, selected), attack_name, seed=0, record_parameters=record_parameters, sample_seeds=selected_seeds)
            attacked_groups.append(attacked.images)
            selected_groups.append(selected)
            for local_index, global_index in enumerate(selected_cpu):
                if record_names:
                    names[global_index] = attack_name
                if record_parameters:
                    parameters[global_index] = attacked.parameters[local_index]
        output = _assemble_attack_groups(images, attacked_groups, selected_groups)
        return AttackResult(output, names, parameters)

    def forward(self, images: torch.Tensor, *, seed: int, record_parameters: bool=False, record_names: bool=True) -> AttackResult:
        generator = _device_generator(images.device, seed)
        assignments = torch.randint(0, len(self.names), (images.shape[0],), device=images.device, generator=generator)
        if images.shape[0] == 0:
            raise ValueError('Stage-3 batch must not be empty')
        attacked_groups: list[torch.Tensor] = []
        selected_groups: list[torch.Tensor] = []
        names = [''] * images.shape[0] if record_names else []
        parameters = self._empty_parameters(images.shape[0])
        for attack_index, attack_name in enumerate(self.names):
            selected = torch.nonzero(assignments == attack_index, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            attacked = self.apply_named(images.index_select(0, selected), attack_name, seed=seed + 1009 * (attack_index + 1), record_parameters=record_parameters)
            attacked_groups.append(attacked.images)
            selected_groups.append(selected)
            if record_names or record_parameters:
                selected_cpu = selected.detach().cpu().tolist()
                for local, global_index in enumerate(selected_cpu):
                    if record_names:
                        names[global_index] = attack_name
                    if record_parameters:
                        parameters[global_index] = attacked.parameters[local]
        output = _assemble_attack_groups(images, attacked_groups, selected_groups)
        return AttackResult(output, names, parameters)
