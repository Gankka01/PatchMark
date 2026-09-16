from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .patch_encoder import SharedPatchEncoder
from .patch_ops import unfold_nonoverlap, fold_nonoverlap

class PatchGridEncoder(nn.Module):

    def __init__(self, internal_size: int, grid_size: int, module_patch_size: int=8, residual_scale: float=0.1, active_only: bool=True, patch_chunk_size: int=2048):
        super().__init__()
        self.internal_size = int(internal_size)
        self.grid_size = int(grid_size)
        self.module_patch_size = int(module_patch_size)
        self.region_size = self.grid_size * self.module_patch_size
        self.region_offset = (self.internal_size - self.region_size) // 2
        self.residual_scale = float(residual_scale)
        self.active_only = bool(active_only)
        self.patch_chunk_size = int(patch_chunk_size)
        if self.region_offset < 0:
            raise ValueError('barcode region does not fit inside the internal image')
        self.patch_encoder = SharedPatchEncoder(out_channels=3)
        self.compiled_patch_encoder = None
        self.channels_last = False
        self.pad_to_full_chunk = False

    def _extract_region(self, image: torch.Tensor) -> torch.Tensor:
        o = self.region_offset
        return image[:, :, o:o + self.region_size, o:o + self.region_size]

    def _patchify(self, region: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = region.shape
        unfolded = unfold_nonoverlap(region, self.module_patch_size)
        expected = self.grid_size ** 2
        if unfolded.shape[-1] != expected:
            raise RuntimeError(f'Expected {expected} encoder patches, got {unfolded.shape[-1]}')
        return unfolded.transpose(1, 2).reshape(batch * expected, channels, self.module_patch_size, self.module_patch_size)

    def configure_execution_backend(self, *, compiled_patch_encoder=None, channels_last: bool=False, pad_to_full_chunk: bool=False) -> None:
        self.compiled_patch_encoder = compiled_patch_encoder
        self.channels_last = bool(channels_last)
        self.pad_to_full_chunk = bool(pad_to_full_chunk)

    def _encode_chunk(self, chunk: torch.Tensor, *, full_chunk: bool) -> torch.Tensor:
        if self.channels_last:
            chunk = chunk.contiguous(memory_format=torch.channels_last)
        if full_chunk and self.compiled_patch_encoder is not None:
            return self.compiled_patch_encoder(chunk)
        return self.patch_encoder(chunk)

    def _run_patch_encoder(self, patches: torch.Tensor) -> torch.Tensor:
        if self.patch_chunk_size <= 0:
            return self._encode_chunk(patches, full_chunk=True)
        original_count = int(patches.shape[0])
        if self.pad_to_full_chunk and original_count % self.patch_chunk_size:
            pad_count = self.patch_chunk_size - original_count % self.patch_chunk_size
            padding = patches.new_zeros((pad_count, *patches.shape[1:]))
            patches = torch.cat((patches, padding), dim=0)
        if patches.shape[0] <= self.patch_chunk_size:
            output = self._encode_chunk(patches, full_chunk=patches.shape[0] == self.patch_chunk_size)
            return output[:original_count]
        outputs = []
        for start in range(0, patches.shape[0], self.patch_chunk_size):
            chunk = patches[start:start + self.patch_chunk_size]
            outputs.append(self._encode_chunk(chunk, full_chunk=chunk.shape[0] == self.patch_chunk_size))
        return torch.cat(outputs, dim=0)[:original_count]

    def forward(self, image: torch.Tensor, module_grid: torch.Tensor, *, active_indices: torch.Tensor | None=None) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, self.internal_size, self.internal_size):
            raise ValueError(f'Expected [B,3,{self.internal_size},{self.internal_size}], got {tuple(image.shape)}')
        if module_grid.ndim == 3:
            module_grid = module_grid.unsqueeze(1)
        expected_grid = (image.shape[0], 1, self.grid_size, self.grid_size)
        if tuple(module_grid.shape) != expected_grid:
            raise ValueError(f'Expected module grid {expected_grid}, got {tuple(module_grid.shape)}')
        batch = image.shape[0]
        patch_count = self.grid_size ** 2
        patches = self._patchify(self._extract_region(image))
        mask = module_grid.reshape(batch * patch_count) > 0.5
        if self.active_only:
            if active_indices is None:
                active_indices = torch.nonzero(mask, as_tuple=False).flatten()
            else:
                if active_indices.ndim != 1 or active_indices.dtype != torch.long:
                    raise TypeError('active_indices must be a one-dimensional torch.long tensor')
                if active_indices.device != patches.device:
                    active_indices = active_indices.to(patches.device, non_blocking=True)
                if active_indices.numel() and torch.is_anomaly_enabled():
                    valid = torch.logical_and(active_indices >= 0, active_indices < patches.shape[0]).all()
                    if hasattr(torch, '_assert_async'):
                        torch._assert_async(valid, 'precomputed active index outside patch range')
            if active_indices.numel() > 0:
                active_patches = patches.index_select(0, active_indices)
                active_residuals = self._run_patch_encoder(active_patches)
                residuals = active_residuals.new_zeros(patches.shape)
                residuals = residuals.index_copy(0, active_indices, active_residuals)
            else:
                residuals = patches.new_zeros(patches.shape)
        else:
            residuals = self._run_patch_encoder(patches)
            residuals = residuals * mask.view(-1, 1, 1, 1).to(residuals.dtype)
        residuals = residuals * self.residual_scale
        folded_input = residuals.reshape(batch, patch_count, -1).transpose(1, 2)
        return fold_nonoverlap(folded_input, (self.region_size, self.region_size), self.module_patch_size)

    def pad_region_to_internal(self, residual_region: torch.Tensor) -> torch.Tensor:
        o = self.region_offset
        right = self.internal_size - self.region_size - o
        bottom = self.internal_size - self.region_size - o
        return F.pad(residual_region, (o, right, o, bottom))
