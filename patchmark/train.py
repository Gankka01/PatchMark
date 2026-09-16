from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import Stage3AttackSuite
from .checkpoints import export_weights, make_model
from .codecs import QRCodec
from .data import add_data_arguments, make_dataset, to_float
from .losses import image_quality_loss
from .payloads import alphanumeric_payloads_for_samples
from .runtime import atomic_save, configure_runtime, load_config, resolve_device
from .sampler import DeterministicEpochSampler


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def stage_at(epoch, lengths):
    boundary = 0
    for index, length in enumerate(lengths):
        boundary += int(length)
        if epoch < boundary:
            return index
    raise ValueError("Epoch exceeds the configured schedule")


def main():
    parser = argparse.ArgumentParser(description="Train PatchMark-S or PatchMark-B")
    parser.add_argument("--config", required=True)
    add_data_arguments(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    seed = int(training["seed"])
    configure_runtime(seed)
    device = resolve_device(args.device)
    epochs = args.epochs if args.epochs is not None else training["checkpoint_epoch"]
    if not 1 <= epochs <= sum(training["stage_epochs"]):
        parser.error("--epochs must lie within the configured three-stage schedule")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last.pt").exists():
        parser.error("An existing run is present; use --resume or select a new output directory")
    dataset = make_dataset(args, "train")
    sampler = DeterministicEpochSampler(dataset, base_seed=seed)
    loader = DataLoader(dataset, batch_size=training["micro_batch_size"], sampler=sampler,
                        num_workers=training["workers"], pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rates"][0],
                                 betas=tuple(training["betas"]), eps=training["eps"],
                                 weight_decay=training["weight_decay"])
    codec = QRCodec(config["model"]["version"], payload_bytes=config["payload_bits"] // 8)
    attacks = Stage3AttackSuite.from_mapping(config["attacks"]).to(device)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("format") != "patchmark-training" or state["config"] != config:
            raise ValueError("Resume requires a matching PatchMark training checkpoint")
        if state["sample_ids"] != [row["sample_id"] for row in dataset.rows]:
            raise ValueError("Resume dataset sample identity or ordering changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["epoch"])
        restore_rng(state["rng"])
        if start >= epochs:
            raise ValueError("The requested epoch count has already been reached")
    try:
        for epoch in range(start, epochs):
            stage = stage_at(epoch, training["stage_epochs"])
            for group in optimizer.param_groups:
                group["lr"] = training["learning_rates"][stage]
            sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            accumulation = training["gradient_accumulation_steps"]
            micro_batch = training["micro_batch_size"]
            for batch_index, batch in enumerate(loader):
                images = to_float(batch["image"], device)
                ids = list(batch["sample_id"])
                context = f"train-epoch-{epoch}"
                payloads = alphanumeric_payloads_for_samples(ids, config["payload_bits"] // 8,
                                                            seed, context=context)
                targets = codec.encode(payloads).module_grid.to(device)
                watermarked, _ = model.encode_training(images, targets)
                if stage == 2:
                    result = attacks.forward_sample_stable(watermarked, sample_ids=ids,
                                                          seed=seed, context=context)
                    attacked = result.images
                    recovery_targets = attacks.transform_module_targets(targets, result.names)
                else:
                    attacked = watermarked
                    recovery_targets = targets
                logits, _ = model.decode(attacked)
                recovery = F.binary_cross_entropy_with_logits(logits, recovery_targets)
                quality, _ = image_quality_loss(images, watermarked, training["alpha_ssim"])
                loss = training["alpha_q"][stage] * quality + training["alpha_r"][stage] * recovery
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                group_start = (batch_index // accumulation) * accumulation * micro_batch
                group_count = min(accumulation * micro_batch, len(dataset) - group_start)
                (loss * len(ids) / group_count).backward()
                epoch_loss += float(loss.detach()) * len(ids)
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Non-finite model gradient")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            state = {"format": "patchmark-training", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                     "config": config, "rng": rng_state(),
                     "sample_ids": [row["sample_id"] for row in dataset.rows]}
            atomic_save(state, output / "last.pt")
            if epoch + 1 == training["checkpoint_epoch"] or epoch + 1 == epochs:
                filename = "patchmark-s.pth" if config["model"]["variant"] == "S" else "patchmark-b.pth"
                export_weights(model, output / filename)
            print(f"Epoch {epoch + 1}/{epochs}; stage {stage + 1}; loss {epoch_loss / len(dataset):.6f}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
