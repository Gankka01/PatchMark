from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch

from .models import PatchMark
from .runtime import atomic_save


MODEL_KEYS = ("version", "residual_scale", "context_size", "patch_chunk_size", "active_patch_encoding")


def make_model(config):
    return PatchMark(**{key: config["model"][key] for key in MODEL_KEYS})


def export_weights(model, path):
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid FP32 model tensor: {key}")
    atomic_save(state, path)


def load_weights(path, config, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint is missing: {path}")
    expected_variant = {"patchmark-s.pth": "S", "patchmark-b.pth": "B"}.get(path.name)
    if expected_variant is not None and config["model"]["variant"] != expected_variant:
        raise ValueError("Checkpoint filename does not match the selected model variant")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Use a PatchMark model state dictionary")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint entry is not a tensor: {key}")
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid model tensor: {key}")
    model = make_model(config)
    model.load_state_dict(state, strict=True)
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return model.to(device).eval(), {"sha256": digest.hexdigest()}
