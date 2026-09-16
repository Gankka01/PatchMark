# PatchMark

Patch-based invisible image watermarking with a 2D barcode mechanism.

This repository contains the PatchMark implementation, training configurations, evaluation code, and an image watermarking example. GPT (OpenAI) was used to remove personal data and identifying information from the experimental code and reorganize it for public release.

## Models

| Configuration | QR symbol | Net message capacity | Internal resolution | Published checkpoint epoch |
|---|---|---:|---:|---:|
| PatchMark-S | Version 3, level M, byte mode | 336 bits / 42 bytes | 256 × 256 | 550 |
| PatchMark-B | Version 11, level M, byte mode | 2,008 bits / 251 bytes | 512 × 512 | 225 |

Both models operate on 1024 × 1024 RGB cover images. The encoder uses 8 × 8 patches and the decoder uses 12 × 12 context patches with stride 8. Message lengths count application bytes before barcode framing and error correction. The neural architecture, QR layout, distortion operators, and scientific loss definitions follow the experimental implementation.

The two trained checkpoints are included in `checkpoints/`. Each contains only the trained model state dictionary. Model settings, training seeds, and checkpoint epochs are documented in the YAML configurations and checkpoint README. See [`checkpoints/README.md`](checkpoints/README.md) for file details and SHA-256 checksums.

## Installation

Use Python 3.10 or newer. Install a compatible PyTorch/torchvision pair for your hardware using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install this package from the repository root:

```bash
python -m pip install -e .
```

The experiment used CUDA 12.1 and NVIDIA RTX A5000 GPUs. The dependency ranges in `requirements.txt` specify the intended dependencies; they are not an exported lockfile of the experiment environment. LPIPS uses the AlexNet backbone and the 0.1 calibration weights. Its first use may download the official backbone weights through torchvision. CPU inference is available but slower than CUDA inference.

## Watermark an image

Run from the repository root:

```bash
python examples/watermark_image.py --input input.png --output outputs/watermarked.png --message "PatchMark example"
```

This example applies EXIF orientation, converts to RGB, and uses a centered aspect-preserving resize/crop when the input is not 1024 × 1024. It embeds the message, saves an 8-bit PNG, and checks message recovery from the saved PNG. This convenience preprocessing is separate from the dataset preparation used in the paper.

The full interface accepts prepared 1024 × 1024 images:

```bash
python -m patchmark.watermark embed --config configs/patchmark_s.yaml --checkpoint checkpoints/patchmark-s.pth --input input.png --output outputs/watermarked.png --message "PatchMark example" --device cuda
python -m patchmark.watermark decode --config configs/patchmark_s.yaml --checkpoint checkpoints/patchmark-s.pth --input outputs/watermarked.png --device cuda
```

Use `--resize-input` during embedding to enable the same convenience preprocessing as the example. Use `configs/patchmark_b.yaml` and `checkpoints/patchmark-b.pth` for PatchMark-B. Messages are nonempty ASCII strings up to the byte capacity shown above. The experimental message generator samples ASCII letters and digits and encodes them in QR byte mode. Recovery depends on image content and distortion; the example prints the actual recovery outcome.

## Datasets and input preparation

Download data from its original provider and follow the applicable usage terms:

- [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- [Landscape / LHQ 1024](https://www.kaggle.com/datasets/dimensi0n/lhq-1024)
- [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)

Dataset images are obtained separately. The paper used 2,000 FFHQ, 2,000 Landscape, and 1,000 Artwork training images, and 400, 400, and 200 test images respectively. Artwork images were prepared at 1024 × 1024 through tiling and cropping. The original selected-image manifest and prepared HDF5 are external inputs. Retain the original image preparation, identities, and split assignments when reproducing the reported numbers.

Two data interfaces are available:

1. **Prepared image folders.** Place 1024 × 1024 images under `data/train/<domain>/` and `data/test/<domain>/`. Paths relative to `data/` become sample IDs. These folders support new training and evaluation runs. A newly selected subset is a new evaluation dataset.
2. **Original prepared HDF5 plus matching manifest.** Supply `--h5 data/images.h5 --manifest data/manifest.csv`. The `images` dataset must be uint8 NHWC with shape `[N,1024,1024,3]`. The CSV requires `sample_id`, `split`, `domain`, and `h5_index`. The reader preserves IDs and row ordering. It checks the stored manifest fingerprint when present.

For image folders with externally assigned sample identities, supply `--data-root data --manifest data/manifest.csv`. The manifest then requires `sample_id,split,domain,path`, where `path` is relative to `data/`. An optional `source_id` column enables source-level checks against split overlap. Training and evaluation readers require prepared 1024 × 1024 inputs and scale uint8 RGB values to `[0,1]`.

Sample IDs determine messages and distortion draws. Changing an ID changes those draws. The package does not recreate the paper's exact image selection from dataset counts alone.

## Training

```bash
python -m patchmark.train --config configs/patchmark_s.yaml --data-root data --output outputs/train_s --device cuda
python -m patchmark.train --config configs/patchmark_b.yaml --data-root data --output outputs/train_b --device cuda
```

Substitute the HDF5/manifest arguments described above to use the original prepared dataset.

| Setting | PatchMark-S | PatchMark-B |
|---|---:|---:|
| Training seed | 1 | 1 |
| Micro-batch size | 16 | 8 |
| Gradient accumulation | 1 | 2 |
| Effective batch size | 16 | 16 |
| Scheduled stage lengths | 10 / 20 / 520 | 10 / 20 / 520 |
| Default stopping epoch | 550 | 225 |

Both configurations use FP32 Adam with learning rates `0.0001 / 0.0003 / 0.0003`. Image-quality weights are `0.5 / 20 / 40`; recovery weights are `1.5 / 1 / 1`. The image loss is MSE plus `0.1 × (1 − SSIM)`. Recovery uses binary cross-entropy across the QR grid. The third stage applies one of the twelve configured distortions per image. Horizontal flips also transform the spatial training target.

The complete schedule contains 550 epochs. PatchMark-B stops at epoch 225 by default to match the reported checkpoint. `--epochs 550` continues a new B training run through the full schedule. Messages change by sample and epoch. Distortion assignments and sample ordering use deterministic seeded rules. CUDA rotation backward may retain nondeterminism, so seeds do not guarantee bit-identical retraining across hardware and library versions.

The training command saves an epoch-boundary `last.pt` containing local resume state and a portable `patchmark-s.pth` or `patchmark-b.pth` containing model weights. Resume with the same configuration and data:

```bash
python -m patchmark.train --config configs/patchmark_s.yaml --data-root data --output outputs/train_s --resume outputs/train_s/last.pt --device cuda
```

The distributed checkpoints provide pretrained model weights. New training runs use `last.pt` for epoch-boundary resume. Keep local resume states and generated outputs outside the published repository.

## Evaluation

```bash
python -m patchmark.evaluate --config configs/patchmark_s.yaml --checkpoint checkpoints/patchmark-s.pth --data-root data --output outputs/eval_s --device cuda
python -m patchmark.evaluate --config configs/patchmark_b.yaml --checkpoint checkpoints/patchmark-b.pth --data-root data --output outputs/eval_b --device cuda
```

The default evaluation covers clean images and the twelve distortions specified in the YAML files. JPEG evaluation uses a real Pillow/libjpeg round trip at quality 30 with 4:2:0 subsampling. Training uses the preserved differentiable JPEG approximation.

Evaluation seeds are `11, 22, 33, 44, 55`. A fixed payload seed of `20260805` preserves messages across distortion seeds. The evaluator reports:

- Module recovery accuracy over QR data and error-correction codeword modules.
- Full-message success based on exact recovery of the input message.
- PSNR, SSIM, and LPIPS on the clean watermarked image before distortion.
- Aggregate and per-domain results with confidence intervals across evaluation-seed means.

The five evaluation seeds use one fixed trained checkpoint. Their confidence intervals describe variation in this evaluation protocol, not variation across independent training runs. A deterministic condition can therefore have a zero-width interval. A single selected seed yields no confidence interval.

QR recovery restores invariant function patterns after the raw module metric has been computed. Format and version fields remain predicted. Horizontal-flip evaluation uses the known flip condition to canonicalize predicted grids before scoring and message recovery. This convention does not imply general geometric synchronization.

The evaluator writes `samples.csv` and `summary.json` under the chosen output directory. Recovery values are percentages. Image-quality metrics use floating-point watermarked images; the image example additionally tests PNG quantization. `--without-lpips` omits LPIPS and records it as null. `--attacks clean` limits evaluation to clean images. The paper's shared-checkpoint payload settings are 128, 256, and 336 bits for S and 1,024 and 2,008 bits for B; select each with `--payload-bits` and a separate output directory.

This package covers the two main PatchMark configurations. Comparative models and the additional ablation studies are outside its scope.

## Code organization

- `patchmark/models/`: patch encoder and native-grid decoder.
- `patchmark/codecs/`: QR message encoding, module masks, and message recovery.
- `patchmark/attacks.py`, `patchmark/jpeg.py`: training distortions and evaluation JPEG.
- `patchmark/train.py`: training and epoch-boundary resume.
- `patchmark/evaluate.py`: image-quality and recovery evaluation.
- `patchmark/watermark.py`: image watermarking and decoding interface.
- `configs/`: training and evaluation settings for S and B.
- `examples/`: minimal image watermarking example.
- `checkpoints/`: one portable trained checkpoint per configuration.

The project retains the MIT license supplied with the implementation. External packages retain their respective licenses.
