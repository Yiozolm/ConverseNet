"""Read-only real HR images with deterministic, declared synthetic LR degradation.

Training uses a manifest split, random crops/dihedral image augmentation and the
repository's five measured 7x7 kernels. Validation uses one fixed center crop per
held-out image. Neither path writes, resizes or relabels the source photographs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(*values):
    text = json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def degrade(hr, kernel, scale, noise_std, rng):
    """Centered circular convolution (not correlation), phase-zero decimation."""
    blurred = ndimage.convolve(hr, kernel[:, :, None], mode="wrap", output=np.float32)
    lr = np.ascontiguousarray(blurred[::scale, ::scale, :])
    if noise_std:
        lr = lr + rng.normal(0, noise_std, lr.shape).astype(np.float32)
    # Additive noise remains unclipped; HR targets remain in [0, 1].
    return np.ascontiguousarray(lr, dtype=np.float32)


def tensor_image(array):
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class DatasetProtocol:
    def __init__(self, manifest, patch_size=96, scale=3, seed=20260917, noise_std=0.01):
        self.manifest = Path(manifest).resolve()
        document = json.loads(self.manifest.read_text(encoding="utf-8-sig"))
        self.root = Path(document["image_root"])
        if not self.root.is_absolute():
            self.root = ROOT / self.root
        self.root = self.root.resolve()
        self.patch_size, self.scale, self.seed = int(patch_size), int(scale), int(seed)
        self.noise_std = float(noise_std)
        if patch_size % scale or patch_size - 2 * scale < 11 or scale not in (1, 2, 3, 4):
            raise ValueError("Patch must be divisible by scale and support cropped SSIM")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("Noise standard deviation must be finite and nonnegative")
        self.records = document["images"]
        self.train = [row for row in self.records if row["split"] == "train"]
        self.validation = [row for row in self.records if row["split"] in ("validation", "val", "holdout")]
        if not self.train or not self.validation or len(self.train) + len(self.validation) != len(self.records):
            raise ValueError("Manifest must assign every image to train or validation")
        names = [row["relative_path"] for row in self.records]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate manifest path")
        for key in ("file_sha256", "rgb_sha256"):
            train_hashes = {row[key] for row in self.train}
            if train_hashes.intersection(row[key] for row in self.validation):
                raise ValueError(f"Train/validation leakage via {key}")
        for row in self.records:
            path = (self.root / row["relative_path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Manifest path escapes image root")
            if min(row["width"], row["height"]) < patch_size:
                raise ValueError(f"Image too small for requested crop: {row['relative_path']}")
            if sha256(path) != row["file_sha256"]:
                raise ValueError(f"Image changed since audit: {row['relative_path']}")
        self.kernels, kernel_hashes = [], {}
        for path in sorted((ROOT / "blur_kernels").glob("kernel_*.npy")):
            array = np.load(path, allow_pickle=False)
            if array.shape != (7, 7) or np.iscomplexobj(array) or not np.isfinite(array).all():
                raise ValueError(f"Expected finite real 7x7 kernel: {path}")
            if not np.isclose(array.sum(), 1, atol=1e-6, rtol=0):
                raise ValueError(f"Kernel sum differs from one; no implicit normalization: {path}")
            self.kernels.append(np.array(array, dtype=np.float32, order="C"))
            kernel_hashes[str(path.relative_to(ROOT))] = sha256(path)
        if len(self.kernels) != 5:
            raise ValueError("This declared protocol requires exactly five repository kernels")
        self._permutations = {}
        self._validation = None
        self.metadata = dict(
            manifest=str(self.manifest), manifest_sha256=sha256(self.manifest),
            image_root=str(self.root), train_images=len(self.train), validation_images=len(self.validation),
            patch_size=self.patch_size, scale=self.scale, seed=self.seed, noise_std=self.noise_std,
            kernels_sha256=kernel_hashes, image_hashes_verified=True,
            source_sha256=sha256(__file__),
            decode="PIL EXIF transpose then uint8 RGB; divide by 255 before blur",
            train_sampling="Deterministic shuffled full epochs; per-occurrence crop, dihedral image augmentation, kernel and noise",
            validation_sampling="Fixed center crop per held-out image; kernel/noise seeded by image ID independent of training seed",
            degradation="scipy.ndimage.convolve(float32 HR, 7x7 kernel, mode=wrap), [0::scale,0::scale], AWGN; no clipping/quantization of LR",
            scope="Real photographs, synthetic declared degradation, patch holdout; not an external official test benchmark")

    def _image(self, row):
        with Image.open(self.root / row["relative_path"]) as image:
            rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (row["height"], row["width"]):
            raise ValueError(f"Decoded size differs from manifest: {row['relative_path']}")
        return rgb

    def _example(self, row, rng, training):
        rgb = self._image(row)
        height, width = rgb.shape[:2]
        patch = self.patch_size
        if training:
            top, left = int(rng.integers(height - patch + 1)), int(rng.integers(width - patch + 1))
        else:
            top, left = (height - patch) // 2, (width - patch) // 2
        crop = rgb[top:top + patch, left:left + patch]
        if training:
            crop = np.rot90(crop, int(rng.integers(4)))
            if rng.integers(2):
                crop = crop[:, ::-1]
        hr = np.ascontiguousarray(crop, dtype=np.float32) / np.float32(255)
        kernel = self.kernels[int(rng.integers(len(self.kernels)))]
        lr = degrade(hr, kernel, self.scale, self.noise_std, rng)
        return tensor_image(lr), torch.from_numpy(kernel.copy())[None], tensor_image(hr)

    def train_batch(self, step, batch_size):
        if step < 0 or batch_size < 1:
            raise ValueError("Invalid training step or batch size")
        examples = []
        for occurrence in range(step * batch_size, (step + 1) * batch_size):
            epoch, offset = divmod(occurrence, len(self.train))
            if epoch not in self._permutations:
                self._permutations[epoch] = np.random.default_rng(
                    stable_seed("train-order-v1", self.seed, epoch)).permutation(len(self.train))
            row = self.train[int(self._permutations[epoch][offset])]
            rng = np.random.default_rng(stable_seed("train-example-v1", self.seed, occurrence))
            examples.append(self._example(row, rng, training=True))
        return tuple(torch.stack(values) for values in zip(*examples))

    def validation_batches(self, batch_size=1):
        if batch_size < 1:
            raise ValueError("Validation batch size must be positive")
        if self._validation is None:
            self._validation = [
                (row["relative_path"], *self._example(row, np.random.default_rng(
                    stable_seed("validation-example-v1", row["relative_path"])), training=False))
                for row in self.validation]
        for start in range(0, len(self._validation), batch_size):
            rows = self._validation[start:start + batch_size]
            yield ([row[0] for row in rows], *(torch.stack(values) for values in zip(*(row[1:] for row in rows))))
