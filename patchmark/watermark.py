from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from .checkpoints import load_weights
from .codecs import QRCodec
from .data import read_image, to_float
from .runtime import configure_runtime, load_config, resolve_device


def embed_image(model, config, image, message):
    payload = message.encode("ascii")
    if not payload or len(payload) * 8 > config["payload_bits"]:
        raise ValueError(f"Provide 1 to {config['payload_bits'] // 8} ASCII characters")
    codec = QRCodec(config["model"]["version"], payload_bytes=len(payload))
    grid = codec.encode([payload], device=image.device).module_grid
    with torch.inference_mode():
        return model.encode(image, grid)[0]


def decode_image(model, config, image):
    codec = QRCodec(config["model"]["version"])
    with torch.inference_mode():
        logits, _ = model.decode(image)
        return codec.decode(logits.sigmoid().cpu(),
                            restore_invariant_patterns=config["evaluation"]["restore_invariant_patterns"])[0]


def save_png(image, path):
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Use a .png output path")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image[0].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(description="Watermark an image or recover an embedded message")
    parser.add_argument("operation", choices=("embed", "decode"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--message")
    parser.add_argument("--resize-input", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.operation == "embed" and (args.message is None or args.output is None):
        parser.error("embed requires --message and --output")
    if args.operation == "decode" and args.resize_input:
        parser.error("decode expects the saved 1024x1024 image")
    config = load_config(args.config)
    configure_runtime(config["training"]["seed"])
    device = resolve_device(args.device)
    model, _ = load_weights(args.checkpoint, config, device)
    image = to_float(read_image(args.input, resize=args.resize_input).unsqueeze(0), device)
    if args.operation == "embed":
        watermarked = embed_image(model, config, image, args.message)
        save_png(watermarked, args.output)
        saved = to_float(read_image(args.output).unsqueeze(0), device)
        recovered = decode_image(model, config, saved)
        print(f"Saved: {args.output}")
        print(f"Saved-image recovery: {recovered == args.message.encode('ascii')}")
    else:
        recovered = decode_image(model, config, image)
        if recovered is None:
            raise SystemExit("QR recovery failed")
        print(recovered.decode("ascii"))


if __name__ == "__main__":
    main()
