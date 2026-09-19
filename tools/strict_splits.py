"""Core utilities for deterministic patient-level GDRBench split construction."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class SplitRecord:
    relative_path: Path
    label: int
    patient_id: str


_EYE_PATTERN = re.compile(r"^(?P<patient>.+)_(?:left|right)$", re.IGNORECASE)
_DEEPDR_PATTERN = re.compile(r"^(?P<patient>.+?)_(?:l|r)[12]$", re.IGNORECASE)


def parse_patient_id(domain: str, relative_path: Path) -> str:
    stem = Path(relative_path).stem.lower()
    domain = domain.upper()
    if domain in {"RLDR", "EYEPACS"}:
        match = _EYE_PATTERN.fullmatch(stem)
        if match is None:
            raise ValueError(
                f"{domain} filename must end with left or right: {relative_path}"
            )
        return match.group("patient")
    if domain == "DEEPDR":
        match = _DEEPDR_PATTERN.fullmatch(stem)
        return match.group("patient") if match else stem
    if domain in {"APTOS", "FGADR", "IDRID", "DDR"}:
        return stem
    raise ValueError(f"Unsupported GDRBench domain: {domain}")


def read_split(path: Path, domain: str) -> list[SplitRecord]:
    path = Path(path)
    records: list[SplitRecord] = []
    labels_by_path: dict[Path, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                relative, raw_label = stripped.rsplit(maxsplit=1)
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(
                    f"Invalid split row {path}:{line_number}: {stripped}"
                ) from error
            if label not in range(5):
                raise ValueError(f"Invalid label {label} at {path}:{line_number}")
            relative_path = Path(relative)
            previous = labels_by_path.get(relative_path)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Conflicting labels for {relative_path}: {previous} and {label}"
                )
            labels_by_path[relative_path] = label
            records.append(
                SplitRecord(
                    relative_path=relative_path,
                    label=label,
                    patient_id=parse_patient_id(domain, relative_path),
                )
            )
    if not records:
        raise ValueError(f"Split is empty: {path}")
    return records


def write_split(path: Path, records: list[SplitRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{record.relative_path.as_posix()} {record.label}\n" for record in records
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_stratified_split(
    records: list[SplitRecord], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[SplitRecord], list[SplitRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    by_patient: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)

    by_grade: dict[int, list[str]] = defaultdict(list)
    for patient_id, patient_records in by_patient.items():
        by_grade[max(record.label for record in patient_records)].append(patient_id)

    random_generator = random.Random(seed)
    validation_patients: set[str] = set()
    for grade in sorted(by_grade):
        patients = sorted(by_grade[grade])
        random_generator.shuffle(patients)
        validation_count = int(round(len(patients) * val_fraction))
        if len(patients) >= 2:
            validation_count = min(len(patients) - 1, max(1, validation_count))
        else:
            validation_count = 0
        validation_patients.update(patients[:validation_count])

    ordered = sorted(
        records, key=lambda record: record.relative_path.as_posix().lower()
    )
    train = [
        record for record in ordered if record.patient_id not in validation_patients
    ]
    validation = [
        record for record in ordered if record.patient_id in validation_patients
    ]
    train_patients = {record.patient_id for record in train}
    validation_patient_ids = {record.patient_id for record in validation}
    if train_patients & validation_patient_ids:
        raise AssertionError("Patient leakage after stratified split")
    if not train or not validation:
        raise ValueError("Patient split produced an empty train or validation split")
    return train, validation


def filter_patients(
    records: list[SplitRecord], excluded_patient_ids: set[str]
) -> tuple[list[SplitRecord], list[SplitRecord]]:
    excluded = set(excluded_patient_ids)
    kept = [record for record in records if record.patient_id not in excluded]
    removed = [record for record in records if record.patient_id in excluded]
    return kept, removed
