from __future__ import annotations

import json
import os
import random
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    model = config["model"]
    if (model["variant"], model["version"]) not in (("S", 3), ("B", 11)):
        raise ValueError("Model variant and QR version disagree")
    max_bits = 336 if model["version"] == 3 else 2008
    if config["payload_bits"] != max_bits:
        raise ValueError("Training uses the full payload of the selected QR configuration")
    if config["data"]["image_size"] != 1024:
        raise ValueError("The supplied protocol requires 1024-pixel images")
    training = config["training"]
    if training["precision"] != "fp32" or training["optimizer"] != "adam":
        raise ValueError("The supplied protocol uses FP32 and Adam")
    for name in ("stage_epochs", "learning_rates", "alpha_q", "alpha_r"):
        if len(training[name]) != 3:
            raise ValueError(f"training.{name} must contain three entries")
    if any(int(v) <= 0 for v in training["stage_epochs"]):
        raise ValueError("Stage lengths must be positive")
    if any(training[k] <= 0 for k in ("micro_batch_size", "gradient_accumulation_steps")):
        raise ValueError("Batch size and gradient accumulation must be positive")
    if config["evaluation"]["threshold"] != 0.5:
        raise ValueError("The module threshold is fixed at 0.5")
    return config


def configure_runtime(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_tensor_dtype_fp32(name, tensor, *, allow_nonfloating=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not tensor.is_floating_point() and allow_nonfloating:
        return
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu for CPU execution")
    return device


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
