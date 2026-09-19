"""Build GDRBench-StrictPatient-v1 without modifying official split files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.strict_splits import (  # noqa: E402
    SplitRecord,
    filter_patients,
    patient_stratified_split,
    read_split,
    sha256_file,
    write_split,
)


DOMAINS = ("APTOS", "DEEPDR", "FGADR", "IDRID", "RLDR", "DDR", "EYEPACS")
COPY_DOMAINS = ("APTOS", "FGADR", "IDRID", "DDR")
SPLIT_NAMES = ("train", "crossval")
BENCHMARK_NAME = "GDRBench-StrictPatient-v1"

PATIENT_ID_RULES = {
    "APTOS": "lower-case image stem; public patient mapping unavailable",
    "DEEPDR": "remove terminal _l1, _l2, _r1 or _r2 from image stem",
    "FGADR": "lower-case image stem; public patient mapping unavailable",
    "IDRID": "lower-case full image stem; cross-split patient identity unavailable",
    "RLDR": "remove terminal _left or _right from image stem",
    "DDR": "lower-case image stem; target-only domain",
    "EYEPACS": "remove terminal _left or _right from image stem",
}

LIMITATIONS = {
    "APTOS": "public patient mapping unavailable; image ID treated as patient ID",
    "FGADR": "public patient mapping unavailable; image ID treated as sample ID",
    "IDRID": "filename treated as sample ID; cross-split patient identity is unavailable",
}


def _patient_ids(records: list[SplitRecord]) -> set[str]:
    return {record.patient_id for record in records}


def _validate_records(
    root: Path, domain: str, records_by_split: dict[str, list[SplitRecord]]
) -> None:
    labels_by_path: dict[Path, int] = {}
    for split_name in SPLIT_NAMES:
        records = records_by_split[split_name]
        if not records:
            raise ValueError(f"Strict split is empty: {domain}_{split_name}.txt")
        for record in records:
            previous = labels_by_path.get(record.relative_path)
            if previous is not None and previous != record.label:
                raise ValueError(
                    f"Conflicting labels for {record.relative_path}: "
                    f"{previous} and {record.label}"
                )
            labels_by_path[record.relative_path] = record.label
            image_path = root / "images" / record.relative_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)


def _domain_statistics(
    records_by_split: dict[str, list[SplitRecord]],
) -> dict:
    combined = records_by_split["train"] + records_by_split["crossval"]
    class_counts = Counter(record.label for record in combined)
    return {
        "train_images": len(records_by_split["train"]),
        "crossval_images": len(records_by_split["crossval"]),
        "total_images": len(combined),
        "unique_patients": len(_patient_ids(combined)),
        "class_counts_0_to_4": [class_counts.get(index, 0) for index in range(5)],
    }


def _hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.glob("*.txt"), key=lambda item: item.name)
    }


def _load_official(root: Path) -> dict[str, dict[str, list[SplitRecord]]]:
    split_root = root / "splits"
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    loaded: dict[str, dict[str, list[SplitRecord]]] = {}
    for domain in DOMAINS:
        loaded[domain] = {}
        for split_name in SPLIT_NAMES:
            path = split_root / f"{domain}_{split_name}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded[domain][split_name] = read_split(path, domain)
        _validate_records(root, domain, loaded[domain])
    return loaded


def _construct_strict_records(
    official: dict[str, dict[str, list[SplitRecord]]],
    seed: int,
    val_fraction: float,
) -> tuple[dict[str, dict[str, list[SplitRecord]]], dict]:
    strict: dict[str, dict[str, list[SplitRecord]]] = {
        domain: {
            split_name: list(official[domain][split_name])
            for split_name in SPLIT_NAMES
        }
        for domain in COPY_DOMAINS
    }

    deepdr_train_patients = _patient_ids(official["DEEPDR"]["train"])
    deepdr_val_patients = _patient_ids(official["DEEPDR"]["crossval"])
    deepdr_overlap = deepdr_train_patients & deepdr_val_patients
    if deepdr_overlap:
        deepdr_train, deepdr_val = patient_stratified_split(
            official["DEEPDR"]["train"] + official["DEEPDR"]["crossval"],
            val_fraction,
            seed,
        )
        strict["DEEPDR"] = {"train": deepdr_train, "crossval": deepdr_val}
    else:
        strict["DEEPDR"] = {
            split_name: list(official["DEEPDR"][split_name])
            for split_name in SPLIT_NAMES
        }

    rldr_all = official["RLDR"]["train"] + official["RLDR"]["crossval"]
    rldr_train, rldr_val = patient_stratified_split(
        rldr_all, val_fraction=val_fraction, seed=seed
    )
    strict["RLDR"] = {"train": rldr_train, "crossval": rldr_val}

    rldr_patients = _patient_ids(rldr_all)
    strict["EYEPACS"] = {}
    removed_eyepacs: list[SplitRecord] = []
    for split_name in SPLIT_NAMES:
        kept, removed = filter_patients(
            official["EYEPACS"][split_name], rldr_patients
        )
        strict["EYEPACS"][split_name] = kept
        removed_eyepacs.extend(removed)

    rldr_stems = {
        record.relative_path.stem.lower()
        for record in rldr_all
    }
    eyepacs_all = (
        official["EYEPACS"]["train"] + official["EYEPACS"]["crossval"]
    )
    eyepacs_stems = {
        record.relative_path.stem.lower()
        for record in eyepacs_all
    }
    filter_audit = {
        "rldr_patients": len(rldr_patients),
        "exact_image_overlap": len(rldr_stems & eyepacs_stems),
        "removed_images": len(removed_eyepacs),
        "remaining_images": sum(
            len(strict["EYEPACS"][split_name]) for split_name in SPLIT_NAMES
        ),
    }
    construction_audit = {
        "deepdr_official_train_crossval_patient_overlap": len(deepdr_overlap),
        "eyepacs_filter": filter_audit,
    }
    return strict, construction_audit


def _publish(staging: Path, output_dir: Path) -> None:
    backup = output_dir.with_name(output_dir.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    had_previous = output_dir.exists()
    if had_previous:
        output_dir.rename(backup)
    try:
        staging.rename(output_dir)
    except Exception:
        if had_previous and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_strict_splits(
    root: Path,
    output_dir: Path | None = None,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> dict:
    root = Path(root).resolve()
    official_dir = (root / "splits").resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (root / "splits_strict").resolve()
    )
    if official_dir == output_dir:
        raise ValueError("Official input and strict output directories must differ")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    official_hashes_before = _hashes(official_dir)
    official = _load_official(root)
    strict, construction_audit = _construct_strict_records(
        official, seed=seed, val_fraction=val_fraction
    )

    for domain in DOMAINS:
        _validate_records(root, domain, strict[domain])

    rldr_train_patients = _patient_ids(strict["RLDR"]["train"])
    rldr_val_patients = _patient_ids(strict["RLDR"]["crossval"])
    rldr_patients = rldr_train_patients | rldr_val_patients
    eyepacs_patients = _patient_ids(
        strict["EYEPACS"]["train"] + strict["EYEPACS"]["crossval"]
    )
    rldr_internal_overlap = rldr_train_patients & rldr_val_patients
    cross_domain_overlap = rldr_patients & eyepacs_patients
    if rldr_internal_overlap:
        raise ValueError("RLDR strict train/crossval patient overlap is not zero")
    if cross_domain_overlap:
        raise ValueError("RLDR/strict EyePACS patient overlap is not zero")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        for domain in DOMAINS:
            for split_name in SPLIT_NAMES:
                destination = temporary_path / f"{domain}_{split_name}.txt"
                if domain in COPY_DOMAINS or (
                    domain == "DEEPDR"
                    and construction_audit[
                        "deepdr_official_train_crossval_patient_overlap"
                    ]
                    == 0
                ):
                    shutil.copy2(
                        official_dir / f"{domain}_{split_name}.txt", destination
                    )
                else:
                    write_split(destination, strict[domain][split_name])

        strict_hashes = _hashes(temporary_path)
        official_hashes_after = _hashes(official_dir)
        if official_hashes_before != official_hashes_after:
            raise RuntimeError("Official split hashes changed during strict generation")

        audit = {
            "schema_version": 1,
            "benchmark": BENCHMARK_NAME,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": (
                "python tools/build_strict_splits.py "
                f"--root {root} --output-dir {output_dir} "
                f"--seed {seed} --val-fraction {val_fraction}"
            ),
            "seed": seed,
            "val_fraction": val_fraction,
            "patient_id_rules": PATIENT_ID_RULES,
            "limitations": LIMITATIONS,
            "domains": {
                domain: {
                    "official": _domain_statistics(official[domain]),
                    "strict": _domain_statistics(strict[domain]),
                }
                for domain in DOMAINS
            },
            "eyepacs_filter": construction_audit["eyepacs_filter"],
            "leakage": {
                "deepdr_official_train_crossval_patient_overlap": construction_audit[
                    "deepdr_official_train_crossval_patient_overlap"
                ],
                "rldr_train_crossval_patient_overlap": len(rldr_internal_overlap),
                "rldr_eyepacs_patient_overlap": len(cross_domain_overlap),
            },
            "sha256": {
                "official": official_hashes_before,
                "strict": strict_hashes,
            },
        }
        (temporary_path / "audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        _publish(temporary_path, output_dir)
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    return audit


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=project_root / "data" / "GDRBench"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    audit = build_strict_splits(
        root=args.root,
        output_dir=args.output_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
