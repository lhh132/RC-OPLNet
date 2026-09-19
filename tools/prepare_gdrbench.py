"""Prepare the downloaded GDRBench bundle for the official GDRNet loader.

The source bundle stores domain directories directly below its root, while the
training code expects ``images/``, ``masks/`` and ``splits/``. This script builds
that layout for redistributable/download-bundle domains and creates fundus
masks for four training sources. Licensed FGADR is added separately with
``tools/import_fgadr.py``; Messidor-2 remains unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


TRAINABLE_DOMAINS = ("APTOS", "DEEPDR", "IDRID", "RLDR")
TARGET_ONLY_DOMAINS = ("DDR", "EYEPACS")
INCLUDED_DOMAINS = TRAINABLE_DOMAINS + TARGET_ONLY_DOMAINS
SEPARATELY_MANAGED_DOMAINS = ("FGADR", "MESSIDOR")


def parse_split(split_path: Path) -> list[tuple[Path, int]]:
    records: list[tuple[Path, int]] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                relative_path, raw_label = stripped.rsplit(" ", 1)
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(f"Invalid split row {split_path}:{line_number}: {stripped}") from error
            if label not in range(5):
                raise ValueError(f"Invalid label {label} at {split_path}:{line_number}")
            records.append((Path(relative_path), label))
    return records


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def fundus_mask(image_path: Path) -> Image.Image:
    """Create a smooth binary mask from an already preprocessed 512x512 image."""
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

    gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    threshold = max(0.0, float(gray.mean()) / 3.0 - 5.0)
    foreground = gray > threshold
    labels, count = ndimage.label(foreground)
    if count == 0:
        raise ValueError(f"Unable to find fundus foreground in {image_path}")

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    foreground = labels == int(component_sizes.argmax())
    foreground = ndimage.binary_fill_holes(foreground)
    foreground = ndimage.binary_closing(foreground, structure=np.ones((7, 7), dtype=bool))
    mask = np.where(foreground, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def create_mask(image_path: Path, mask_path: Path) -> None:
    if mask_path.exists():
        return
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = fundus_mask(image_path)
    temporary_path = mask_path.with_name(mask_path.name + ".tmp")
    mask.save(temporary_path, format="PNG", optimize=True)
    temporary_path.replace(mask_path)


def unique_records(split_root: Path, domains: Iterable[str]) -> dict[Path, int]:
    records: dict[Path, int] = {}
    for domain in domains:
        for split in ("train", "crossval"):
            for relative_path, label in parse_split(split_root / f"{domain}_{split}.txt"):
                existing = records.get(relative_path)
                if existing is not None and existing != label:
                    raise ValueError(f"Conflicting labels for {relative_path}: {existing} and {label}")
                records[relative_path] = label
    return records


def prepare(source_root: Path, output_root: Path, image_mode: str, workers: int) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_splits = source_root / "splits"

    for domain in INCLUDED_DOMAINS:
        if not (source_root / domain).is_dir():
            raise FileNotFoundError(f"Missing source domain directory: {source_root / domain}")
        for split in ("train", "crossval"):
            if not (source_splits / f"{domain}_{split}.txt").is_file():
                raise FileNotFoundError(source_splits / f"{domain}_{split}.txt")

    images_root = output_root / "images"
    masks_root = output_root / "masks"
    splits_root = output_root / "splits"
    for directory in (images_root, masks_root, splits_root):
        directory.mkdir(parents=True, exist_ok=True)

    for domain in SEPARATELY_MANAGED_DOMAINS:
        if (images_root / domain).exists() or (masks_root / domain).exists():
            raise RuntimeError(f"Excluded domain already exists in output: {domain}")

    for domain in INCLUDED_DOMAINS:
        for split in ("train", "crossval"):
            source_split = source_splits / f"{domain}_{split}.txt"
            shutil.copy2(source_split, splits_root / source_split.name)

    all_records = unique_records(source_splits, INCLUDED_DOMAINS)
    for relative_path in all_records:
        source_image = source_root / relative_path
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        link_or_copy(source_image, images_root / relative_path, image_mode)

    training_records = unique_records(source_splits, TRAINABLE_DOMAINS)

    def mask_job(relative_path: Path) -> None:
        create_mask(images_root / relative_path, masks_root / relative_path)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(mask_job, training_records))

    domain_statistics = {}
    for domain in INCLUDED_DOMAINS:
        train = parse_split(splits_root / f"{domain}_train.txt")
        crossval = parse_split(splits_root / f"{domain}_crossval.txt")
        class_counts = [0] * 5
        for _, label in train + crossval:
            class_counts[label] += 1
        domain_statistics[domain] = {
            "train": len(train),
            "crossval": len(crossval),
            "total": len(train) + len(crossval),
            "class_counts_0_to_4": class_counts,
            "role": "trainable" if domain in TRAINABLE_DOMAINS else "target_only",
        }

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "image_mode": image_mode,
        "trainable_domains": list(TRAINABLE_DOMAINS),
        "target_only_domains": list(TARGET_ONLY_DOMAINS),
        "separately_managed_domains": list(SEPARATELY_MANAGED_DOMAINS),
        "unique_images": len(all_records),
        "generated_masks": len(training_records),
        "domains": domain_statistics,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def get_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\dataset\GDRBench\FundusDG_mini"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "GDRBench",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hard links avoid duplicating the approximately 8 GB image bundle.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    result = prepare(args.source_root, args.output_root, args.image_mode, args.workers)
    print(json.dumps(result, indent=2, ensure_ascii=False))
