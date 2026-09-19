"""Import the licensed FGADR Seg-set into an existing local GDRBench tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True, order=True)
class FGADRRecord:
    filename: str
    label: int


def parse_grading_csv(path: Path) -> list[FGADRRecord]:
    records = []
    labels_by_name = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 2:
                raise ValueError(f"row {row_number}: expected two columns")
            filename = row[0].strip()
            candidate = Path(filename)
            if not filename or candidate.name != filename or candidate.is_absolute():
                raise ValueError(f"row {row_number}: expected a plain filename")
            try:
                label = int(row[1].strip())
            except ValueError as error:
                raise ValueError(f"row {row_number}: label must be an integer") from error
            if label not in range(5):
                raise ValueError(f"row {row_number}: label must be in 0..4")
            previous = labels_by_name.get(filename)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Conflicting labels for {filename}: {previous} and {label}"
                )
            if previous is not None:
                raise ValueError(f"Duplicate filename: {filename}")
            labels_by_name[filename] = label
            records.append(FGADRRecord(filename, label))
    if not records:
        raise ValueError(f"No grading records in {path}")
    return records


def stratified_split(records, seed=42, val_fraction=0.2):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    by_grade = defaultdict(list)
    for record in records:
        by_grade[record.label].append(record)
    if set(by_grade) != set(range(5)):
        raise ValueError("All five DR grades 0..4 are required")
    generator = random.Random(seed)
    validation = set()
    for grade in range(5):
        grade_records = sorted(
            by_grade[grade], key=lambda item: item.filename.lower()
        )
        generator.shuffle(grade_records)
        validation_count = round(len(grade_records) * val_fraction)
        validation.update(grade_records[:validation_count])
    ordered = sorted(records, key=lambda item: item.filename.lower())
    return (
        [item for item in ordered if item not in validation],
        [item for item in ordered if item in validation],
    )


def preprocess_fundus(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = rgb.astype(np.float32) @ np.asarray(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    threshold = max(0.0, float(gray.mean()) / 3.0 - 5.0)
    foreground = gray > threshold
    labels, count = ndimage.label(foreground)
    if count == 0:
        raise ValueError("Unable to find fundus foreground")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    foreground = labels == int(sizes.argmax())
    foreground = ndimage.binary_fill_holes(foreground)
    foreground = ndimage.binary_closing(
        foreground, structure=np.ones((7, 7), dtype=bool)
    )
    rows, columns = np.where(foreground)
    if not len(rows):
        raise ValueError("Unable to find fundus foreground")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    cropped_rgb = rgb[top:bottom, left:right]
    cropped_mask = foreground[top:bottom, left:right]
    side = max(cropped_rgb.shape[:2])
    image_square = np.zeros((side, side, 3), dtype=np.uint8)
    mask_square = np.zeros((side, side), dtype=np.uint8)
    y = (side - cropped_rgb.shape[0]) // 2
    x = (side - cropped_rgb.shape[1]) // 2
    image_square[y : y + cropped_rgb.shape[0], x : x + cropped_rgb.shape[1]] = (
        cropped_rgb
    )
    mask_square[y : y + cropped_mask.shape[0], x : x + cropped_mask.shape[1]] = (
        cropped_mask * 255
    )
    processed = Image.fromarray(image_square, mode="RGB").resize(
        (512, 512), Image.Resampling.BILINEAR
    )
    mask = Image.fromarray(mask_square, mode="L").resize(
        (512, 512), Image.Resampling.NEAREST
    )
    mask = Image.fromarray(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8), mode="L"
    )
    return processed, mask


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_hashes(split_root: Path, include_fgadr: bool = False) -> dict[str, str]:
    hashes = {}
    for path in sorted(Path(split_root).glob("*.txt"), key=lambda item: item.name):
        if include_fgadr or not path.name.startswith("FGADR_"):
            hashes[path.name] = _sha256(path)
    return hashes


def validate_source(
    source_root: Path, expected_count: int = 1842
) -> list[FGADRRecord]:
    source_root = Path(source_root).resolve()
    seg_root = source_root / "Seg-set"
    image_root = seg_root / "Original_Images"
    csv_path = seg_root / "DR_Seg_Grading_Label.csv"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    records = parse_grading_csv(csv_path)
    image_paths = {path.name: path for path in image_root.iterdir() if path.is_file()}
    record_names = {record.filename for record in records}
    if record_names != set(image_paths):
        missing = sorted(record_names - set(image_paths))
        extra = sorted(set(image_paths) - record_names)
        raise ValueError(
            f"CSV/image filename sets differ; missing={missing[:3]}, extra={extra[:3]}"
        )
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} FGADR records, found {len(records)}")
    for record in records:
        path = image_paths[record.filename]
        with Image.open(path) as image:
            if image.size != (1280, 1280):
                raise ValueError(f"Expected 1280x1280 image: {path} has {image.size}")
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB")
    counts = [Counter(record.label for record in records)[grade] for grade in range(5)]
    if expected_count == 1842 and counts != [101, 212, 595, 647, 287]:
        raise ValueError(f"Unexpected production class counts: {counts}")
    return records


def _write_split(path: Path, records: list[FGADRRecord]) -> None:
    text = "".join(f"FGADR/{record.filename} {record.label}\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_targets(
    output_root: Path, staging: Path, overwrite: bool
) -> None:
    pairs = [
        (staging / "images" / "FGADR", output_root / "images" / "FGADR"),
        (staging / "masks" / "FGADR", output_root / "masks" / "FGADR"),
        (
            staging / "splits" / "FGADR_train.txt",
            output_root / "splits" / "FGADR_train.txt",
        ),
        (
            staging / "splits" / "FGADR_crossval.txt",
            output_root / "splits" / "FGADR_crossval.txt",
        ),
        (staging / "manifest.json", output_root / "manifest.json"),
        (
            staging / "fgadr_import_audit.json",
            output_root / "fgadr_import_audit.json",
        ),
    ]
    protected_targets = [target for _, target in pairs if target.name != "manifest.json"]
    existing = [target for target in protected_targets if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "FGADR outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    backup = Path(tempfile.mkdtemp(prefix=".fgadr-backup-", dir=output_root))
    moved_backups = []
    published = []
    try:
        for index, (_, target) in enumerate(pairs):
            if target.exists():
                backup_target = backup / str(index)
                shutil.move(str(target), str(backup_target))
                moved_backups.append((backup_target, target))
        for source, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            published.append(target)
    except Exception:
        for target in reversed(published):
            _remove_path(target)
        for backup_source, target in moved_backups:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_source), str(target))
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def import_fgadr(
    source_root: Path,
    output_root: Path,
    seed: int = 42,
    val_fraction: float = 0.2,
    expected_count: int = 1842,
    workers: int = 4,
    overwrite: bool = False,
) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    protected_targets = [
        output_root / "images" / "FGADR",
        output_root / "masks" / "FGADR",
        output_root / "splits" / "FGADR_train.txt",
        output_root / "splits" / "FGADR_crossval.txt",
        output_root / "fgadr_import_audit.json",
    ]
    existing_targets = [path for path in protected_targets if path.exists()]
    if existing_targets and not overwrite:
        raise FileExistsError(
            "FGADR outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing_targets)
        )
    records = validate_source(source_root, expected_count=expected_count)
    train, crossval = stratified_split(records, seed=seed, val_fraction=val_fraction)
    split_root = output_root / "splits"
    existing_hashes_before = _split_hashes(split_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staging = Path(tempfile.mkdtemp(prefix=".fgadr-import-", dir=output_root))
    stage_images = staging / "images" / "FGADR"
    stage_masks = staging / "masks" / "FGADR"
    stage_images.mkdir(parents=True)
    stage_masks.mkdir(parents=True)
    image_root = source_root / "Seg-set" / "Original_Images"

    def process(record: FGADRRecord) -> None:
        with Image.open(image_root / record.filename) as image:
            processed, mask = preprocess_fundus(image)
        processed.save(stage_images / record.filename, format="PNG")
        mask.save(stage_masks / record.filename, format="PNG")

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            list(executor.map(process, records))
        _write_split(staging / "splits" / "FGADR_train.txt", train)
        _write_split(staging / "splits" / "FGADR_crossval.txt", crossval)

        class_counts = [Counter(record.label for record in records)[grade] for grade in range(5)]
        previous_fgadr = manifest.get("domains", {}).get("FGADR", {})
        previous_total = int(previous_fgadr.get("total", 0))
        trainable = [name for name in manifest.get("trainable_domains", []) if name != "FGADR"]
        trainable.append("FGADR")
        manifest["trainable_domains"] = trainable
        manifest["excluded_domains"] = [
            name for name in manifest.get("excluded_domains", []) if name != "FGADR"
        ]
        manifest["unique_images"] = int(manifest.get("unique_images", 0)) - previous_total + len(records)
        manifest["generated_masks"] = int(manifest.get("generated_masks", 0)) - previous_total + len(records)
        manifest.setdefault("domains", {})["FGADR"] = {
            "train": len(train),
            "crossval": len(crossval),
            "total": len(records),
            "class_counts_0_to_4": class_counts,
            "role": "trainable",
            "split_origin": "local fixed-seed stratified split",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        existing_hashes_after = _split_hashes(split_root)
        if existing_hashes_before != existing_hashes_after:
            raise RuntimeError("Existing non-FGADR split hashes changed during import")
        fgadr_split_hashes = _split_hashes(staging / "splits", include_fgadr=True)
        audit = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "seed": seed,
            "val_fraction": val_fraction,
            "preprocessor": "fgadr-eyeq-style-v1",
            "input_records": len(records),
            "input_class_counts_0_to_4": class_counts,
            "output_images": len(list(stage_images.glob("*.png"))),
            "output_masks": len(list(stage_masks.glob("*.png"))),
            "train_images": len(train),
            "crossval_images": len(crossval),
            "patient_mapping_limitation": (
                "public patient mapping unavailable; image ID treated as sample ID"
            ),
            "source_label_manifest_sha256": _sha256(
                source_root / "Seg-set" / "DR_Seg_Grading_Label.csv"
            ),
            "existing_split_sha256_before": existing_hashes_before,
            "existing_split_sha256_after": existing_hashes_after,
            "fgadr_split_sha256": fgadr_split_hashes,
        }
        if audit["output_images"] != len(records) or audit["output_masks"] != len(records):
            raise RuntimeError("Incomplete staged FGADR output")
        (staging / "fgadr_import_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        _publish_targets(output_root, staging, overwrite=overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if staging.exists():
        shutil.rmtree(staging)
    return audit


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\dataset\FGADR"))
    parser.add_argument(
        "--output-root", type=Path, default=project_root / "data" / "GDRBench"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = import_fgadr(
        args.source_root,
        args.output_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
