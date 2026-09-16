from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .blocks import ConvNormAct, ResidualConvBlock, SqueezeExcitation

class LocalContextExtractor(nn.Module):

    def __init__(self, context_size: int, feature_dim: int=128):
        super().__init__()
        if context_size not in (8, 12):
            raise ValueError('Supported context sizes are 8 and 12')
        stride = 2 if context_size == 8 else 3
        self.context_size = context_size
        self.net = nn.Sequential(nn.Conv2d(3, 128, 3, padding=1), nn.LayerNorm([128, context_size, context_size]), nn.GELU(), nn.Conv2d(128, 256, 3, stride=stride, padding=1 if context_size == 8 else 0), nn.GELU(), nn.Conv2d(256, 512, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, feature_dim, 1), nn.GELU())

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.net(patches).flatten(1)

class NativeGridDecoder(nn.Module):

    def __init__(self, internal_size: int, grid_size: int, module_patch_size: int=8, context_size: int=12, feature_dim: int=128, patch_chunk_size: int=1024):
        super().__init__()
        self.internal_size = int(internal_size)
        self.grid_size = int(grid_size)
        self.module_patch_size = int(module_patch_size)
        self.context_size = int(context_size)
        self.patch_chunk_size = int(patch_chunk_size)
        self.region_size = self.grid_size * self.module_patch_size
        self.region_offset = (self.internal_size - self.region_size) // 2
        self.context_margin = (self.context_size - self.module_patch_size) // 2
        if self.context_size < self.module_patch_size or (self.context_size - self.module_patch_size) % 2:
            raise ValueError('context_size must exceed module_patch_size by an even amount')
        self.local_extractor = LocalContextExtractor(context_size, feature_dim)
        self.compiled_local_extractor = None
        self.compiled_grid_head = None
        self.channels_last = False
        self.pad_to_full_chunk = False
        self.grid_head = nn.Sequential(ConvNormAct(feature_dim, feature_dim, 3), ResidualConvBlock(feature_dim), SqueezeExcitation(feature_dim), ConvNormAct(feature_dim, 64, 3), nn.Conv2d(64, 1, 1))

    def _extract_context_patches(self, image: torch.Tensor) -> torch.Tensor:
        margin = self.context_margin
        padded = F.pad(image, (margin, margin, margin, margin), mode='reflect') if margin else image
        y0 = self.region_offset
        x0 = self.region_offset
        extent = self.region_size + 2 * margin
        crop = padded[:, :, y0:y0 + extent, x0:x0 + extent]
        unfolded = F.unfold(crop, kernel_size=self.context_size, stride=self.module_patch_size)
        expected = self.grid_size ** 2
        if unfolded.shape[-1] != expected:
            raise RuntimeError(f'Expected {expected} decoder patches, got {unfolded.shape[-1]}')
        batch = image.shape[0]
        return unfolded.transpose(1, 2).reshape(batch * expected, 3, self.context_size, self.context_size)

    def configure_execution_backend(self, *, compiled_local_extractor=None, compiled_grid_head=None, channels_last: bool=False, pad_to_full_chunk: bool=False) -> None:
        self.compiled_local_extractor = compiled_local_extractor
        self.compiled_grid_head = compiled_grid_head
        self.channels_last = bool(channels_last)
        self.pad_to_full_chunk = bool(pad_to_full_chunk)

    def _extract_chunk(self, chunk: torch.Tensor, *, full_chunk: bool) -> torch.Tensor:
        if self.channels_last:
            chunk = chunk.contiguous(memory_format=torch.channels_last)
        if full_chunk and self.compiled_local_extractor is not None:
            return self.compiled_local_extractor(chunk)
        return self.local_extractor(chunk)

    def _extract_features_chunked(self, patches: torch.Tensor) -> torch.Tensor:
        if self.patch_chunk_size <= 0:
            return self._extract_chunk(patches, full_chunk=True)
        original_count = int(patches.shape[0])
        if self.pad_to_full_chunk and original_count % self.patch_chunk_size:
            pad_count = self.patch_chunk_size - original_count % self.patch_chunk_size
            padding = patches.new_zeros((pad_count, *patches.shape[1:]))
            patches = torch.cat((patches, padding), dim=0)
        if patches.shape[0] <= self.patch_chunk_size:
            output = self._extract_chunk(patches, full_chunk=patches.shape[0] == self.patch_chunk_size)
            return output[:original_count]
        chunks = []
        for start in range(0, patches.shape[0], self.patch_chunk_size):
            chunk = patches[start:start + self.patch_chunk_size]
            chunks.append(self._extract_chunk(chunk, full_chunk=chunk.shape[0] == self.patch_chunk_size))
        return torch.cat(chunks, dim=0)[:original_count]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, self.internal_size, self.internal_size):
            raise ValueError(f'Expected [B,3,{self.internal_size},{self.internal_size}], got {tuple(image.shape)}')
        batch = image.shape[0]
        patches = self._extract_context_patches(image)
        features = self._extract_features_chunked(patches)
        feature_dim = features.shape[1]
        grid_features = features.view(batch, self.grid_size, self.grid_size, feature_dim).permute(0, 3, 1, 2)
        if self.channels_last:
            grid_features = grid_features.contiguous(memory_format=torch.channels_last)
        head = self.compiled_grid_head if self.compiled_grid_head is not None else self.grid_head
        return head(grid_features)
