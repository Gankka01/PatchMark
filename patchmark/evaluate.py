from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch
from torch.utils.data import DataLoader

from .attacks import ATTACK_NAMES, Stage3AttackSuite
from .checkpoints import load_weights
from .codecs import QRCodec
from .data import add_data_arguments, make_dataset, to_float
from .jpeg import real_jpeg_roundtrip
from .metrics import psnr, ssim
from .payloads import alphanumeric_payloads_for_samples
from .runtime import configure_runtime, load_config, resolve_device, write_json


METRICS = ("module_accuracy", "full_message_success", "psnr", "ssim", "lpips")


def attack_seed(sample_id, seed, name):
    digest = sha256(f"{sample_id}|{int(seed)}|{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def apply_attack(images, name, ids, seed, suite):
    if name == "clean":
        return images
    if name == "jpeg_compression":
        return real_jpeg_roundtrip(images, suite.config.jpeg_quality,
                                   subsampling=suite.config.jpeg_subsampling)
    seeds = [attack_seed(identity, seed, name) for identity in ids]
    return suite.apply_named(images, name, seed=0, sample_seeds=seeds).images


def summarize(rows):
    seed_groups = defaultdict(list)
    for row in rows:
        for domain in ("all", row["domain"]):
            seed_groups[(domain, row["attack"], row["seed"])].append(row)
    conditions = defaultdict(list)
    for (domain, attack, seed), group in seed_groups.items():
        means = {metric: float(np.mean([r[metric] for r in group]))
                 if group[0][metric] is not None else None for metric in METRICS}
        conditions[(domain, attack)].append((seed, len(group), means))
    summaries = []
    for (domain, attack), records in sorted(conditions.items()):
        item = {"domain": domain, "attack": attack, "samples_per_seed": records[0][1],
                "seeds": sorted(r[0] for r in records)}
        for metric in METRICS:
            if records[0][2][metric] is None:
                item[metric] = None
                continue
            values = np.asarray([r[2][metric] for r in records])
            mean = float(values.mean())
            ci = None
            if len(values) > 1:
                half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                ci = [mean - half, mean + half]
            item[metric] = {"mean": mean, "ci95_across_evaluation_seeds": ci}
        summaries.append(item)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PatchMark checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    add_data_arguments(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--payload-bits", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--attacks", nargs="+", choices=("clean",) + ATTACK_NAMES)
    parser.add_argument("--without-lpips", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["evaluation"]
    seeds = args.seeds or settings["seeds"]
    if len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        parser.error("Evaluation seeds must be distinct positive integers")
    bits = args.payload_bits if args.payload_bits is not None else config["payload_bits"]
    if bits <= 0 or bits % 8 or bits > config["payload_bits"]:
        parser.error("Payload bits must be byte-aligned and within the selected QR capacity")
    batch_size = args.batch_size if args.batch_size is not None else settings["batch_size"]
    if batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and workers nonnegative")
    configure_runtime(config["training"]["seed"])
    device = resolve_device(args.device)
    model, checkpoint = load_weights(args.checkpoint, config, device)
    codec = QRCodec(config["model"]["version"], payload_bytes=bits // 8)
    suite = Stage3AttackSuite.from_mapping(config["attacks"]).to(device)
    dataset = make_dataset(args, "test")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    metric = None
    if not args.without_lpips:
        import lpips
        metric = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
        metric.requires_grad_(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "samples.csv").exists() or (output / "summary.json").exists():
        parser.error("Select a new output directory to preserve an existing evaluation")
    names = args.attacks or ["clean", *ATTACK_NAMES]
    if len(set(names)) != len(names):
        parser.error("Attack names must be unique")
    rows = []
    columns = ["sample_id", "domain", "seed", "attack", "payload_bits", *METRICS]
    try:
        with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for index, batch in enumerate(loader):
                images = to_float(batch["image"], device)
                ids = list(batch["sample_id"])
                payloads = alphanumeric_payloads_for_samples(ids, bits // 8, settings["payload_seed"],
                                                            context="evaluation")
                encoded = codec.encode(payloads)
                targets = encoded.module_grid.to(device)
                mask = encoded.codeword_mask.to(device)
                watermarked, _, _ = model.encode(images, targets)
                quality = {"psnr": psnr(images, watermarked).cpu().tolist(),
                           "ssim": ssim(images, watermarked).cpu().tolist(),
                           "lpips": metric(images * 2 - 1, watermarked * 2 - 1).flatten().cpu().tolist()
                           if metric is not None else [None] * len(ids)}
                for seed in seeds:
                    for name in names:
                        attacked = apply_attack(watermarked, name, ids, seed, suite)
                        logits, _ = model.decode(attacked)
                        if name == "horizontal_flip":
                            logits = torch.flip(logits, dims=(-1,))
                        binary = logits.sigmoid() >= settings["threshold"]
                        accuracy = ((binary == targets.bool()) & mask).flatten(1).sum(1) / mask.flatten(1).sum(1)
                        decoded = codec.decode(binary.float().cpu(), restore_invariant_patterns=settings["restore_invariant_patterns"])
                        for j, identity in enumerate(ids):
                            row = {"sample_id": identity, "domain": batch["domain"][j], "seed": seed,
                                   "attack": name, "payload_bits": bits,
                                   "module_accuracy": float(accuracy[j]) * 100.0,
                                   "full_message_success": float(decoded[j] == payloads[j]) * 100.0,
                                   **{key: values[j] for key, values in quality.items()}}
                            writer.writerow(row)
                            rows.append(row)
                handle.flush()
                print(f"Evaluated {min((index + 1) * batch_size, len(dataset))}/{len(dataset)} images", flush=True)
        write_json({"model": config["model"]["variant"], "payload_bits": bits,
                    "checkpoint": checkpoint, "payload_seed": settings["payload_seed"],
                    "evaluation_seeds": seeds, "settings": config,
                    "ci_unit": "evaluation-seed means with a fixed trained checkpoint",
                    "quality_reference": "clean floating-point watermarked images before distortion",
                    "results": summarize(rows)}, output / "summary.json")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
