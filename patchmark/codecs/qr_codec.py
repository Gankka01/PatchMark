from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np
import qrcode
import qrcode.constants
import qrcode.util
import torch
from .base import BarcodeBatch, BarcodeCodec, BarcodeMetadata, numpy_mask_to_tensor
from .qr_function_mask import apply_teach_qr, qr_function_pattern
from .qr_layout import qr_layout_masks
_EC_MAP = {'L': qrcode.constants.ERROR_CORRECT_L, 'M': qrcode.constants.ERROR_CORRECT_M, 'Q': qrcode.constants.ERROR_CORRECT_Q, 'H': qrcode.constants.ERROR_CORRECT_H}
_MAX_BYTE_PAYLOAD = {(3, 'M'): 42, (11, 'M'): 251}

@dataclass
class QRDecodeTrace:
    raw_binary_grid: torch.Tensor
    taught_binary_grid: torch.Tensor
    decoded_payloads: list[bytes | None]
    teach_qr_applied: bool
    threshold: float

class QRCodec(BarcodeCodec):

    def __init__(self, version: int, error_correction: str='M', payload_bytes: int | None=None, *, teach_qr_default: bool=True):
        error_correction = error_correction.upper()
        if error_correction not in _EC_MAP:
            raise ValueError(f'Unsupported error correction level: {error_correction}')
        self.version = int(version)
        self.error_correction = error_correction
        self.grid_size = 17 + 4 * self.version
        max_bytes = _MAX_BYTE_PAYLOAD.get((self.version, self.error_correction))
        if payload_bytes is None:
            if max_bytes is None:
                raise ValueError('payload_bytes is required for unlisted QR version/EC combinations')
            payload_bytes = max_bytes
        self.payload_bytes = int(payload_bytes)
        if max_bytes is not None and self.payload_bytes > max_bytes:
            raise ValueError(f'QR v{self.version}-{self.error_correction} supports at most {max_bytes} bytes; got {self.payload_bytes}')
        self.teach_qr_default = bool(teach_qr_default)
        self._patterns = qr_function_pattern(self.version)
        self._layout = qr_layout_masks(self.version, self.error_correction)
        total = self.grid_size ** 2
        function = int(self._layout.function_mask.sum())
        codeword = int(self._layout.codeword_mask.sum())
        data_modules = int(self._layout.data_mask.sum())
        ecc_modules = int(self._layout.ecc_mask.sum())
        remainder = int(self._layout.remainder_mask.sum())
        self._metadata = BarcodeMetadata(codec_name=f'QRv{self.version}-{self.error_correction}', grid_height=self.grid_size, grid_width=self.grid_size, user_payload_bits=self.payload_bytes * 8, total_modules=total, function_modules=function, codeword_modules=codeword, data_modules=data_modules, ecc_modules=ecc_modules, filler_modules=0, remainder_modules=remainder, valid_modules=total, notes='Spatial masks separate QR data-codeword, ECC-codeword, function, and remainder modules. User payload bits are reported independently from the QR data-codeword capacity, which also contains mode/length/padding.')

    @property
    def metadata(self) -> BarcodeMetadata:
        return self._metadata

    def _encode_one(self, payload: bytes) -> np.ndarray:
        if len(payload) != self.payload_bytes:
            raise ValueError(f'Expected exactly {self.payload_bytes} payload bytes, got {len(payload)}')
        qr = qrcode.QRCode(version=self.version, error_correction=_EC_MAP[self.error_correction], box_size=1, border=0)
        qr.add_data(qrcode.util.QRData(payload, mode=qrcode.util.MODE_8BIT_BYTE), optimize=0)
        qr.make(fit=False)
        matrix = np.asarray(qr.get_matrix(), dtype=np.uint8)
        if matrix.shape != (self.grid_size, self.grid_size):
            raise RuntimeError(f'Unexpected QR shape: {matrix.shape}')
        return matrix

    def encode(self, payloads: Sequence[bytes], device: torch.device | str='cpu') -> BarcodeBatch:
        payloads = [bytes(payload) for payload in payloads]
        matrices = np.stack([self._encode_one(payload) for payload in payloads], axis=0)
        module_grid = torch.from_numpy(matrices).unsqueeze(1).to(device=device, dtype=torch.float32)
        batch = len(payloads)
        function_mask = numpy_mask_to_tensor(self._layout.function_mask, batch, device)
        codeword_mask = numpy_mask_to_tensor(self._layout.codeword_mask, batch, device)
        data_mask = numpy_mask_to_tensor(self._layout.data_mask, batch, device)
        ecc_mask = numpy_mask_to_tensor(self._layout.ecc_mask, batch, device)
        remainder_mask = numpy_mask_to_tensor(self._layout.remainder_mask, batch, device)
        zero = torch.zeros_like(function_mask)
        valid = torch.ones_like(function_mask)
        return BarcodeBatch(module_grid=module_grid, function_mask=function_mask, codeword_mask=codeword_mask, data_mask=data_mask, ecc_mask=ecc_mask, filler_mask=zero, remainder_mask=remainder_mask, valid_mask=valid, payloads=payloads, metadata=self.metadata)

    def apply_teach(self, module_grids: torch.Tensor) -> torch.Tensor:
        binary = self.ensure_binary_grid(module_grids)
        return apply_teach_qr(binary, self.version)

    @staticmethod
    def _grid_to_reader_image(grid: np.ndarray, scale: int=10, quiet_zone: int=4) -> np.ndarray:
        grid = (grid != 0).astype(np.uint8)
        padded = np.pad(grid, quiet_zone, mode='constant', constant_values=0)
        image = (1 - padded) * 255
        return cv2.resize(image.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def decode_with_trace(self, module_grids: torch.Tensor, *, teach_qr: bool | None=None, threshold: float=0.5) -> QRDecodeTrace:
        if teach_qr is None:
            teach_qr = self.teach_qr_default
        raw = self.ensure_binary_grid(module_grids, threshold=threshold)
        taught = apply_teach_qr(raw, self.version) if teach_qr else raw.clone()
        detector = cv2.QRCodeDetector()
        outputs: list[bytes | None] = []
        for grid in taught.cpu().numpy()[:, 0]:
            image = self._grid_to_reader_image(grid)
            try:
                text, _, _ = detector.detectAndDecode(image)
                outputs.append(text.encode('utf-8') if text else None)
            except (cv2.error, UnicodeError):
                outputs.append(None)
        return QRDecodeTrace(raw_binary_grid=raw, taught_binary_grid=taught, decoded_payloads=outputs, teach_qr_applied=bool(teach_qr), threshold=float(threshold))

    def decode(self, module_grids: torch.Tensor, *, threshold: float=0.5, restore_invariant_patterns: bool=True, teach_qr: bool | None=None) -> list[bytes | None]:
        if teach_qr is not None:
            restore_invariant_patterns = bool(teach_qr)
        return self.decode_with_trace(module_grids, teach_qr=restore_invariant_patterns, threshold=threshold).decoded_payloads

    def with_payload_bytes(self, payload_bytes: int) -> 'QRCodec':
        return QRCodec(self.version, self.error_correction, payload_bytes=payload_bytes, teach_qr_default=self.teach_qr_default)
