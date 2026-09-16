import argparse

from patchmark.checkpoints import load_weights
from patchmark.data import read_image, to_float
from patchmark.runtime import configure_runtime, load_config, resolve_device
from patchmark.watermark import decode_image, embed_image, save_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchmark_s.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/patchmark-s.pth")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/watermarked.png")
    parser.add_argument("--message", default="PatchMark example")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config["training"]["seed"])
    device = resolve_device(args.device)
    model, _ = load_weights(args.checkpoint, config, device)
    image = to_float(read_image(args.input, resize=True).unsqueeze(0), device)
    save_png(embed_image(model, config, image, args.message), args.output)
    saved = to_float(read_image(args.output).unsqueeze(0), device)
    recovered = decode_image(model, config, saved)
    print(f"Saved-image recovery: {recovered == args.message.encode('ascii')}")


if __name__ == "__main__":
    main()
