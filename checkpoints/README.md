# Trained checkpoints

| File | Model | Net payload | Training seed | Checkpoint epoch |
|---|---|---:|---:|---:|
| `patchmark-s.pth` | PatchMark-S / QR version 3-M | 336 bits | 1 | 550 |
| `patchmark-b.pth` | PatchMark-B / QR version 11-M | 2,008 bits | 1 | 225 |

Each file is a plain PyTorch model state dictionary containing 76 FP32 tensors with 7,852,652 parameter elements. The tensor storage bytes are preserved from the supplied trained checkpoints. Optimizer state, RNG state, configuration objects, epoch records, private paths, and experiment metadata have been removed from the checkpoint files. Model and training settings are provided in `configs/` and the table above.

Use `configs/patchmark_s.yaml` with `patchmark-s.pth` and `configs/patchmark_b.yaml` with `patchmark-b.pth`. Both variants share the same neural architecture, so tensor shapes alone cannot distinguish them. The loader checks the standard filename against the selected variant, loads with `weights_only=True`, and checks state-dictionary keys and shapes with `strict=True`.

These weights support inference and evaluation. Training creates a separate local `last.pt` for epoch-boundary resume and exports a weights-only file under the matching standard filename. Generated resume states and results should remain outside the published repository.

## SHA-256

- `patchmark-s.pth`: `ca804bcc36f10bc6b75e2ebdaff658d5124aaae47060aca64ba135c9cbcf6184`
- `patchmark-b.pth`: `91009296ad443545d44e157b1a6e0acff61c2d0181dfdcac8cf920599939c797`
