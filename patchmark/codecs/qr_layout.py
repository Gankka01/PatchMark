from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import qrcode.base
import qrcode.constants
from .qr_function_mask import qr_function_mask
_EC_MAP = {'L': qrcode.constants.ERROR_CORRECT_L, 'M': qrcode.constants.ERROR_CORRECT_M, 'Q': qrcode.constants.ERROR_CORRECT_Q, 'H': qrcode.constants.ERROR_CORRECT_H}

@dataclass(frozen=True)
class QRLayoutMasks:
    function_mask: np.ndarray
    data_mask: np.ndarray
    ecc_mask: np.ndarray
    codeword_mask: np.ndarray
    remainder_mask: np.ndarray
    data_codeword_bits: int
    ecc_codeword_bits: int

    @property
    def remainder_bits(self) -> int:
        return int(self.remainder_mask.sum())

def _placement_order(version: int, function_mask: np.ndarray) -> list[tuple[int, int]]:
    size = version * 4 + 17
    if function_mask.shape != (size, size):
        raise ValueError('function mask shape mismatch')
    order: list[tuple[int, int]] = []
    row = size - 1
    inc = -1
    for base_col in range(size - 1, 0, -2):
        col = base_col - 1 if base_col <= 6 else base_col
        while True:
            for current_col in (col, col - 1):
                if not function_mask[row, current_col]:
                    order.append((row, current_col))
            row += inc
            if row < 0 or row >= size:
                row -= inc
                inc = -inc
                break
    return order

def qr_layout_masks(version: int, error_correction: str='M') -> QRLayoutMasks:
    error_correction = error_correction.upper()
    if error_correction not in _EC_MAP:
        raise ValueError(f'Unsupported QR error-correction level: {error_correction}')
    function = qr_function_mask(version)
    blocks = qrcode.base.rs_blocks(version, _EC_MAP[error_correction])
    data_bits = sum((block.data_count for block in blocks)) * 8
    total_bits = sum((block.total_count for block in blocks)) * 8
    ecc_bits = total_bits - data_bits
    order = _placement_order(version, function)
    if len(order) < total_bits:
        raise RuntimeError(f'QR placement capacity {len(order)} is smaller than codeword bits {total_bits}')
    shape = function.shape
    data_mask = np.zeros(shape, dtype=np.bool_)
    ecc_mask = np.zeros(shape, dtype=np.bool_)
    remainder_mask = np.zeros(shape, dtype=np.bool_)
    for index, (row, col) in enumerate(order):
        if index < data_bits:
            data_mask[row, col] = True
        elif index < total_bits:
            ecc_mask[row, col] = True
        else:
            remainder_mask[row, col] = True
    codeword = data_mask | ecc_mask
    semantic_union = function | codeword | remainder_mask
    if not semantic_union.all():
        missing = int((~semantic_union).sum())
        raise RuntimeError(f'QR layout left {missing} unclassified modules')
    if np.any(function & (codeword | remainder_mask)):
        raise RuntimeError('QR semantic masks overlap')
    return QRLayoutMasks(function_mask=function, data_mask=data_mask, ecc_mask=ecc_mask, codeword_mask=codeword, remainder_mask=remainder_mask, data_codeword_bits=data_bits, ecc_codeword_bits=ecc_bits)
