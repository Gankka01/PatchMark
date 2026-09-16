from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path, *, resize=False):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (1024, 1024):
            if not resize:
                raise ValueError(f"Expected a prepared 1024x1024 image: {path}")
            image = ImageOps.fit(image, (1024, 1024), method=Image.Resampling.LANCZOS,
                                centering=(0.5, 0.5))
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)


def to_float(images, device):
    return images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)


class ImageDataset(Dataset):
    def __init__(self, *, root=None, h5=None, manifest=None, split="test"):
        self.root = Path(root).resolve() if root is not None else None
        self.h5_path = str(Path(h5).resolve()) if h5 is not None else None
        self.handle = None
        self.pid = None
        if (self.root is None) == (self.h5_path is None):
            raise ValueError("Select exactly one input: an image root or an HDF5 file")
        if manifest:
            with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or not {"sample_id", "split", "domain"}.issubset(rows[0]):
                raise ValueError("Manifest requires sample_id, split, and domain columns")
            if len({r["sample_id"] for r in rows}) != len(rows):
                raise ValueError("Manifest sample IDs must be globally unique")
            if "source_id" in rows[0]:
                sources = {}
                for row in rows:
                    old = sources.setdefault(row["source_id"], row["split"])
                    if old != row["split"]:
                        raise ValueError("A source image occurs in multiple splits")
            if self.h5_path:
                with h5py.File(self.h5_path, "r") as handle:
                    images = handle["images"]
                    if images.dtype != np.uint8 or images.shape[1:] != (1024, 1024, 3):
                        raise ValueError("HDF5 images must be uint8 [N,1024,1024,3]")
                    indices = [int(r["h5_index"]) for r in rows]
                    if len(set(indices)) != len(indices) or set(indices) != set(range(len(images))):
                        raise ValueError("Manifest must identify every HDF5 row exactly once")
                    expected = handle.attrs.get("manifest_sha256")
                    if expected:
                        actual = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
                        expected = expected.decode("ascii") if isinstance(expected, bytes) else str(expected)
                        if expected != actual:
                            raise ValueError("HDF5 and manifest fingerprints differ")
            self.rows = [r for r in rows if r["split"] == split]
        elif self.h5_path:
            raise ValueError("HDF5 input requires its matching manifest")
        else:
            base = self.root / split
            if not base.is_dir():
                raise FileNotFoundError(f"Split directory is missing: {base}")
            self.rows = []
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    relative = path.relative_to(self.root)
                    domain = path.relative_to(base).parts[0] if len(path.relative_to(base).parts) > 1 else "custom"
                    self.rows.append({"sample_id": relative.as_posix(), "path": relative.as_posix(),
                                      "domain": domain, "split": split})
        if any(not row.get("sample_id") or not row.get("domain") or row["domain"] == "all" for row in self.rows):
            raise ValueError("Sample IDs and domains must be nonempty; domain all is reserved")
        if not self.rows:
            raise ValueError(f"No images found for split {split}")
        if self.root:
            for row in self.rows:
                relative = Path(row["path"])
                full = (self.root / relative).resolve()
                if relative.is_absolute() or not full.is_relative_to(self.root):
                    raise ValueError("Manifest image paths must stay inside the image root")
                if not full.is_file():
                    raise FileNotFoundError(full)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.h5_path:
            if self.handle is None or self.pid != os.getpid():
                if self.handle is not None:
                    self.handle.close()
                self.handle = h5py.File(self.h5_path, "r")
                self.pid = os.getpid()
            array = np.asarray(self.handle["images"][int(row["h5_index"])], dtype=np.uint8)
            image = torch.from_numpy(array.copy()).permute(2, 0, 1)
        else:
            image = read_image(self.root / row["path"])
        return {"image": image, "sample_id": row["sample_id"], "domain": row["domain"]}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handle"] = None
        state["pid"] = None
        return state

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def add_data_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-root")
    group.add_argument("--h5")
    parser.add_argument("--manifest")


def make_dataset(args, split):
    return ImageDataset(root=args.data_root, h5=args.h5, manifest=args.manifest, split=split)
