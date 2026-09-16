from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Sequence
import numpy as np
import torch

@dataclass(frozen=True)
class BarcodeMetadata:
    codec_name: str
    grid_height: int
    grid_width: int
    user_payload_bits: int
    total_modules: int
    function_modules: int
    codeword_modules: int
    data_modules: int
    ecc_modules: int
    filler_modules: int
    remainder_modules: int = 0
    valid_modules: int | None = None
    notes: str = ''
    layout_invariant: bool = True

    def __post_init__(self) -> None:
        integer_fields = ('grid_height', 'grid_width', 'user_payload_bits', 'total_modules', 'function_modules', 'codeword_modules', 'data_modules', 'ecc_modules', 'filler_modules', 'remainder_modules')
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'metadata.{name} must be a nonnegative integer')
        if self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError('metadata grid dimensions must be positive')
        expected_total = self.grid_height * self.grid_width
        if self.total_modules != expected_total:
            raise ValueError(f'metadata.total_modules={self.total_modules} does not match grid area {expected_total}')
        if self.valid_modules is None:
            object.__setattr__(self, 'valid_modules', self.total_modules)
        valid = self.valid_modules
        if isinstance(valid, bool) or not isinstance(valid, int) or (not 0 < valid <= self.total_modules):
            raise ValueError('metadata.valid_modules must be a positive integer no larger than total_modules')
        primary_total = self.function_modules + self.codeword_modules + self.filler_modules + self.remainder_modules
        if primary_total != self.total_modules:
            raise ValueError(f'Primary semantic module counts must cover the full canvas: {primary_total} != {self.total_modules}')
        meaningful_valid = self.function_modules + self.codeword_modules + self.remainder_modules
        if meaningful_valid > valid:
            raise ValueError(f'Function/codeword/remainder module counts exceed valid module count: {meaningful_valid} > {valid}')
        if self.data_modules + self.ecc_modules > self.codeword_modules:
            raise ValueError('data_modules + ecc_modules cannot exceed codeword_modules')

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.grid_height, self.grid_width)

@dataclass
class BarcodeBatch:
    module_grid: torch.Tensor
    function_mask: torch.Tensor
    codeword_mask: torch.Tensor
    data_mask: torch.Tensor
    ecc_mask: torch.Tensor
    filler_mask: torch.Tensor
    payloads: list[bytes]
    metadata: BarcodeMetadata
    remainder_mask: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    sample_layouts: list[dict[str, int | list[int]]] | None = None

    def __post_init__(self) -> None:
        reference = self.module_grid
        if reference.ndim != 4 or reference.shape[1] != 1:
            raise ValueError(f'module_grid must have shape [B,1,H,W], got {tuple(reference.shape)}')
        expected = reference.shape
        if expected[-2:] != self.metadata.grid_shape:
            raise ValueError(f'module_grid spatial shape {tuple(expected[-2:])} does not match metadata {self.metadata.grid_shape}')
        if expected[0] != len(self.payloads):
            raise ValueError(f'payload count {len(self.payloads)} does not match batch size {expected[0]}')
        if not torch.isfinite(reference).all():
            raise ValueError('module_grid contains NaN or infinity')
        if not torch.all((reference == 0) | (reference == 1)):
            raise ValueError('Encoded module_grid must contain binary 0/1 values')
        mask_names = ('function_mask', 'codeword_mask', 'data_mask', 'ecc_mask', 'filler_mask')
        for name in mask_names:
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f'{name} shape {tuple(value.shape)} != {tuple(expected)}')
            if value.dtype is not torch.bool:
                raise TypeError(f'{name} must be torch.bool, got {value.dtype}')
        if self.remainder_mask is None:
            self.remainder_mask = torch.zeros_like(self.function_mask)
        if self.valid_mask is None:
            self.valid_mask = torch.ones_like(self.function_mask)
        if self.remainder_mask.shape != expected or self.valid_mask.shape != expected:
            raise ValueError('remainder_mask and valid_mask must match module_grid shape')
        if self.remainder_mask.dtype is not torch.bool or self.valid_mask.dtype is not torch.bool:
            raise TypeError('remainder_mask and valid_mask must be torch.bool')
        if torch.any(self.data_mask & self.ecc_mask):
            raise ValueError('data_mask and ecc_mask overlap')
        if torch.any(self.data_mask & ~self.codeword_mask):
            raise ValueError('data_mask is not a subset of codeword_mask')
        if torch.any(self.ecc_mask & ~self.codeword_mask):
            raise ValueError('ecc_mask is not a subset of codeword_mask')
        primary = {'function': self.function_mask, 'codeword': self.codeword_mask, 'filler': self.filler_mask, 'remainder': self.remainder_mask}
        names = list(primary)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1:]:
                if torch.any(primary[left_name] & primary[right_name]):
                    raise ValueError(f'{left_name}_mask and {right_name}_mask overlap')
        union = torch.zeros_like(self.function_mask)
        for value in primary.values():
            union |= value
        if not torch.all(union):
            missing = int((~union).sum().item())
            raise ValueError(f'Semantic masks do not cover the full canvas ({missing} modules missing)')
        meaningful = self.function_mask | self.codeword_mask | self.remainder_mask
        if torch.any(meaningful & ~self.valid_mask):
            raise ValueError('Function/codeword/remainder modules must be valid modules')
        if torch.any(~self.valid_mask & ~self.filler_mask):
            raise ValueError('Every invalid padded module must be classified as filler')
        if torch.any(reference.bool() & ~self.valid_mask):
            raise ValueError('Padded modules outside valid_mask must remain zero')
        if self.metadata.layout_invariant:
            for name in (*mask_names, 'remainder_mask', 'valid_mask'):
                value = getattr(self, name)
                if value.shape[0] > 1 and (not torch.equal(value, value[:1].expand_as(value))):
                    raise ValueError(f'{name} changes across samples in one BarcodeBatch')
        else:
            if self.sample_layouts is None or len(self.sample_layouts) != expected[0]:
                raise ValueError('Variable-layout BarcodeBatch requires one sample_layouts record per payload')
            for index, layout in enumerate(self.sample_layouts):
                observed_valid = int(self.valid_mask[index].sum().item())
                declared_valid = int(layout.get('valid_modules', -1))
                if declared_valid != observed_valid:
                    raise ValueError(f'sample_layouts[{index}].valid_modules={declared_valid} does not match valid_mask count {observed_valid}')
        if self.metadata.layout_invariant:
            expected_counts = {'function_modules': int(self.function_mask[0].sum().item()), 'codeword_modules': int(self.codeword_mask[0].sum().item()), 'data_modules': int(self.data_mask[0].sum().item()), 'ecc_modules': int(self.ecc_mask[0].sum().item()), 'filler_modules': int(self.filler_mask[0].sum().item()), 'remainder_modules': int(self.remainder_mask[0].sum().item()), 'valid_modules': int(self.valid_mask[0].sum().item())}
            for field, observed in expected_counts.items():
                declared = int(getattr(self.metadata, field))
                if declared != observed:
                    raise ValueError(f'metadata.{field}={declared} does not match mask count {observed}')
        if self.metadata.total_modules != expected[-2] * expected[-1]:
            raise ValueError('metadata.total_modules does not match grid dimensions')
        for payload in self.payloads:
            if len(payload) * 8 != self.metadata.user_payload_bits:
                raise ValueError(f'Payload has {len(payload) * 8} bits but metadata declares {self.metadata.user_payload_bits}')

    def to(self, device: torch.device | str) -> 'BarcodeBatch':
        return BarcodeBatch(module_grid=self.module_grid.to(device), function_mask=self.function_mask.to(device), codeword_mask=self.codeword_mask.to(device), data_mask=self.data_mask.to(device), ecc_mask=self.ecc_mask.to(device), filler_mask=self.filler_mask.to(device), remainder_mask=self.remainder_mask.to(device), valid_mask=self.valid_mask.to(device), payloads=self.payloads, metadata=self.metadata, sample_layouts=self.sample_layouts)

    def with_module_grid(self, module_grid: torch.Tensor) -> 'BarcodeBatch':
        return replace(self, module_grid=module_grid)

@dataclass(frozen=True)
class DecodePolicy:
    threshold: float = 0.5
    restore_invariant_patterns: bool = True

class BarcodeCodec(ABC):

    @property
    @abstractmethod
    def metadata(self) -> BarcodeMetadata:
        raise NotImplementedError

    @abstractmethod
    def encode(self, payloads: Sequence[bytes], device: torch.device | str='cpu') -> BarcodeBatch:
        raise NotImplementedError

    @abstractmethod
    def decode(self, module_grids: torch.Tensor, *, threshold: float=0.5, restore_invariant_patterns: bool=True) -> list[bytes | None]:
        raise NotImplementedError

    def decode_with_policy(self, module_grids: torch.Tensor, policy: DecodePolicy | None=None) -> list[bytes | None]:
        policy = policy or DecodePolicy()
        return self.decode(module_grids, threshold=policy.threshold, restore_invariant_patterns=policy.restore_invariant_patterns)

    @staticmethod
    def ensure_binary_grid(module_grids: torch.Tensor, threshold: float=0.5) -> torch.Tensor:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f'threshold must be in [0,1], got {threshold}')
        if module_grids.ndim == 3:
            module_grids = module_grids.unsqueeze(1)
        if module_grids.ndim != 4 or module_grids.shape[1] != 1:
            raise ValueError(f'Expected [B,1,H,W] or [B,H,W], got {tuple(module_grids.shape)}')
        if module_grids.is_floating_point():
            module_grids = (module_grids >= threshold).to(torch.uint8)
        else:
            module_grids = (module_grids != 0).to(torch.uint8)
        return module_grids

def numpy_mask_to_tensor(mask: np.ndarray, batch_size: int, device: torch.device | str) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.bool_)).view(1, 1, *mask.shape)
    return tensor.expand(batch_size, -1, -1, -1).to(device)
