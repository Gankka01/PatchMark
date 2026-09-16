from __future__ import annotations
import torch

def unfold_nonoverlap(input: torch.Tensor, patch: int) -> torch.Tensor:
    b, c, h, w = input.shape
    if h % patch or w % patch:
        raise ValueError('spatial dimensions must be divisible by patch')
    gh, gw = (h // patch, w // patch)
    return input.reshape(b, c, gh, patch, gw, patch).permute(0, 1, 3, 5, 2, 4).reshape(b, c * patch * patch, gh * gw)

def fold_nonoverlap(columns: torch.Tensor, output_hw: tuple[int, int], patch: int) -> torch.Tensor:
    b, cp, l = columns.shape
    h, w = output_hw
    area = patch * patch
    if cp % area:
        raise ValueError('invalid columns')
    c = cp // area
    gh, gw = (h // patch, w // patch)
    if l != gh * gw:
        raise ValueError('invalid locations')
    return columns.reshape(b, c, patch, patch, gh, gw).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h, w)
