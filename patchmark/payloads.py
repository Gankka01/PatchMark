from __future__ import annotations
from hashlib import sha256
import random
import string
from typing import Sequence
import torch
_ALPHABET = string.ascii_letters + string.digits

def _sample_seed(sample_id: str, seed: int, context: str, length: int) -> int:
    digest = sha256(f'patchmark-payload-v1|{sample_id}|{seed}|{context}|{length}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:16], 'big')

def alphanumeric_payload_for_sample(sample_id: str, byte_length: int, seed: int, *, context: str='default') -> bytes:
    if byte_length <= 0:
        raise ValueError('byte_length must be positive')
    rng = random.Random(_sample_seed(str(sample_id), seed, context, byte_length))
    return ''.join((rng.choice(_ALPHABET) for _ in range(byte_length))).encode('ascii')

def alphanumeric_payloads_for_samples(sample_ids: Sequence[str], byte_length: int, seed: int, *, context: str='default') -> list[bytes]:
    return [alphanumeric_payload_for_sample(str(sample_id), byte_length, seed, context=context) for sample_id in sample_ids]
