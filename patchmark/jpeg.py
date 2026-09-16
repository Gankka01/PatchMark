from __future__ import annotations
import io
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

def _pillow_subsampling(subsampling: str) -> int:
    mapping = {'4:4:4': 0, '4:2:2': 1, '4:2:0': 2}
    try:
        return mapping[subsampling]
    except KeyError as exc:
        raise ValueError(f'Unsupported JPEG subsampling: {subsampling}') from exc

def real_jpeg_roundtrip(images: torch.Tensor, quality: int=30, *, subsampling: str='4:2:0') -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f'Expected [B,3,H,W], got {tuple(images.shape)}')
    if not 1 <= int(quality) <= 100:
        raise ValueError('quality must be in [1,100]')
    device, dtype = (images.device, images.dtype)
    cpu_uint8 = images.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def roundtrip(array: np.ndarray) -> torch.Tensor:
        buffer = io.BytesIO()
        Image.fromarray(array, mode='RGB').save(buffer, format='JPEG', quality=int(quality), subsampling=_pillow_subsampling(subsampling), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded_image:
            decoded = np.asarray(decoded_image.convert('RGB'), dtype=np.float32) / 255.0
        return torch.from_numpy(decoded.copy()).permute(2, 0, 1)
    workers = max(1, min(int(os.environ.get('PATCHMARK_JPEG_WORKERS', '4')), len(cpu_uint8)))
    if workers == 1:
        outputs = [roundtrip(array) for array in cpu_uint8]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='patchmark-jpeg') as pool:
            outputs = list(pool.map(roundtrip, cpu_uint8))
    return torch.stack(outputs).to(device=device, dtype=dtype, non_blocking=True)
