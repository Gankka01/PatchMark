from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class QRFunctionPattern:
    full_function_mask: np.ndarray
    teachable_mask: np.ndarray
    teachable_values: np.ndarray

def alignment_pattern_positions(version: int) -> list[int]:
    if not 1 <= version <= 40:
        raise ValueError(f'QR version must be in [1, 40], got {version}')
    if version == 1:
        return []
    num_align = version // 7 + 2
    step = 26 if version == 32 else (version * 4 + num_align * 2 + 1) // (num_align * 2 - 2) * 2
    result = [6]
    pos = version * 4 + 10
    for _ in range(num_align - 1):
        result.insert(1, pos)
        pos -= step
    return result

def _mark_rect(mask: np.ndarray, y0: int, x0: int, height: int, width: int) -> None:
    h, w = mask.shape
    y1, x1 = (max(0, y0), max(0, x0))
    y2, x2 = (min(h, y0 + height), min(w, x0 + width))
    if y1 < y2 and x1 < x2:
        mask[y1:y2, x1:x2] = True

def qr_function_mask(version: int) -> np.ndarray:
    size = version * 4 + 17
    mask = np.zeros((size, size), dtype=np.bool_)
    _mark_rect(mask, -1, -1, 9, 9)
    _mark_rect(mask, -1, size - 8, 9, 9)
    _mark_rect(mask, size - 8, -1, 9, 9)
    mask[6, :] = True
    mask[:, 6] = True
    positions = alignment_pattern_positions(version)
    for cy in positions:
        for cx in positions:
            if (cx, cy) in {(6, 6), (size - 7, 6), (6, size - 7)}:
                continue
            _mark_rect(mask, cy - 2, cx - 2, 5, 5)
    for i in range(6):
        mask[i, 8] = True
        mask[8, i] = True
    mask[7, 8] = mask[8, 8] = mask[8, 7] = True
    for i in range(8):
        mask[8, size - 1 - i] = True
    for i in range(7):
        mask[size - 1 - i, 8] = True
    mask[size - 8, 8] = True
    if version >= 7:
        _mark_rect(mask, 0, size - 11, 6, 3)
        _mark_rect(mask, size - 11, 0, 3, 6)
    return mask

def _draw_finder(values: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    size = values.shape[0]
    _mark_rect(mask, top - 1, left - 1, 9, 9)
    for y in range(7):
        for x in range(7):
            yy, xx = (top + y, left + x)
            if 0 <= yy < size and 0 <= xx < size:
                mask[yy, xx] = True
                values[yy, xx] = int(y in {0, 6} or x in {0, 6} or (2 <= y <= 4 and 2 <= x <= 4))

def _draw_alignment(values: np.ndarray, mask: np.ndarray, cy: int, cx: int) -> None:
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            mask[cy + dy, cx + dx] = True
            radius = max(abs(dy), abs(dx))
            values[cy + dy, cx + dx] = int(radius in {0, 2})

def qr_function_pattern(version: int) -> QRFunctionPattern:
    size = version * 4 + 17
    values = np.zeros((size, size), dtype=np.uint8)
    teachable = np.zeros((size, size), dtype=np.bool_)
    _draw_finder(values, teachable, 0, 0)
    _draw_finder(values, teachable, 0, size - 7)
    _draw_finder(values, teachable, size - 7, 0)
    for index in range(8, size - 8):
        teachable[6, index] = True
        teachable[index, 6] = True
        values[6, index] = int(index % 2 == 0)
        values[index, 6] = int(index % 2 == 0)
    for cy in alignment_pattern_positions(version):
        for cx in alignment_pattern_positions(version):
            if (cx, cy) in {(6, 6), (size - 7, 6), (6, size - 7)}:
                continue
            _draw_alignment(values, teachable, cy, cx)
    teachable[size - 8, 8] = True
    values[size - 8, 8] = 1
    return QRFunctionPattern(qr_function_mask(version), teachable, values)

def apply_teach_qr(module_grids: torch.Tensor, version: int) -> torch.Tensor:
    if module_grids.ndim == 3:
        module_grids = module_grids.unsqueeze(1)
    expected = version * 4 + 17
    if tuple(module_grids.shape[1:]) != (1, expected, expected):
        raise ValueError(f'Expected [B,1,{expected},{expected}], got {tuple(module_grids.shape)}')
    pattern = qr_function_pattern(version)
    mask = torch.from_numpy(pattern.teachable_mask).to(module_grids.device).view(1, 1, expected, expected)
    values = torch.from_numpy(pattern.teachable_values).to(device=module_grids.device, dtype=module_grids.dtype).view(1, 1, expected, expected)
    return torch.where(mask, values, module_grids)
