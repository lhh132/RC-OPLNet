import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DOMAINS = ("APTOS", "DEEPDR", "FGADR", "IDRID", "RLDR")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient_id(domain, relative_path):
    stem = Path(relative_path).stem.lower()
    if domain == "DEEPDR":
        return re.sub(r"_(?:l1|l2|r1|r2)$", "", stem)
    if domain == "RLDR":
        return re.sub(r"_(?:left|right)$", "", stem)
    return stem


def _read_split(data_root, split_dir, domain, split):
    path = split_dir / f"{domain}_{split}.txt"
    entries = []
    invalid_rows = []
    if not path.is_file():
        return {
            "path": path,
            "entries": entries,
            "invalid_rows": [{"line": None, "reason": "missing split file"}],
        }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 2:
                invalid_rows.append(
                    {
                        "line": line_number,
                        "reason": "expected path and label",
                    }
                )
                continue
            relative_path, raw_label = fields
            try:
                label = int(raw_label)
            except ValueError:
                label = None
            image_path = data_root / "images" / relative_path
            entries.append(
                {
                    "relative_path": relative_path,
                    "image_path": image_path,
                    "label": label,
                }
            )
    return {"path": path, "entries": entries, "invalid_rows": invalid_rows}


def audit_lodo(data_root, profile="strict"):
    data_root = Path(data_root)
    split_dir = data_root / (
        "splits_strict" if profile == "strict" else "splits"
    )
    split_files = []
    by_domain = {}
    missing_images = 0
    invalid_labels = 0
    malformed_rows = 0
    exact_overlap = {}
    patient_overlap = {}
    class_counts = {}

    for domain in DOMAINS:
        by_domain[domain] = {}
        combined_labels = Counter()
        for split in ("train", "crossval"):
            parsed = _read_split(data_root, split_dir, domain, split)
            by_domain[domain][split] = parsed["entries"]
            malformed_rows += len(parsed["invalid_rows"])
            for entry in parsed["entries"]:
                if not entry["image_path"].is_file():
                    missing_images += 1
                label = entry["label"]
                if label not in range(5):
                    invalid_labels += 1
                else:
                    combined_labels[label] += 1
            item = {
                "domain": domain,
                "split": split,
                "path": str(parsed["path"]),
                "exists": parsed["path"].is_file(),
                "rows": len(parsed["entries"]),
                "malformed_rows": parsed["invalid_rows"],
            }
            if parsed["path"].is_file():
                item["sha256"] = _sha256(parsed["path"])
            split_files.append(item)

        train_paths = {
            entry["relative_path"]
            for entry in by_domain[domain]["train"]
        }
        val_paths = {
            entry["relative_path"]
            for entry in by_domain[domain]["crossval"]
        }
        exact_overlap[domain] = len(train_paths & val_paths)
        train_patients = {
            _patient_id(domain, path) for path in train_paths
        }
        val_patients = {_patient_id(domain, path) for path in val_paths}
        patient_overlap[domain] = len(train_patients & val_patients)
        class_counts[domain] = [combined_labels[index] for index in range(5)]

    folds = {}
    for target in DOMAINS:
        sources = [domain for domain in DOMAINS if domain != target]
        folds[target] = {
            "sources": sources,
            "train": sum(
                len(by_domain[domain]["train"]) for domain in sources
            ),
            "validation": sum(
                len(by_domain[domain]["crossval"]) for domain in sources
            ),
            "test": len(by_domain[target]["train"])
            + len(by_domain[target]["crossval"]),
        }

    audit_path = split_dir / "audit.json"
    audit_metadata = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {}
    )
    supported_patient_domains = ("DEEPDR", "RLDR")
    patient_leakage = sum(
        patient_overlap[domain] for domain in supported_patient_domains
    )
    valid = (
        missing_images == 0
        and invalid_labels == 0
        and malformed_rows == 0
        and sum(exact_overlap.values()) == 0
        and patient_leakage == 0
        and all(item["exists"] for item in split_files)
    )
    return {
        "data_root": str(data_root),
        "profile": profile,
        "split_files": split_files,
        "folds": folds,
        "class_counts": class_counts,
        "exact_train_crossval_overlap": exact_overlap,
        "patient_train_crossval_overlap": patient_overlap,
        "patient_overlap_fully_checkable_domains": list(
            supported_patient_domains
        ),
        "audit_metadata": audit_metadata,
        "summary": {
            "missing_images": missing_images,
            "invalid_labels": invalid_labels,
            "malformed_rows": malformed_rows,
            "exact_overlap": sum(exact_overlap.values()),
            "supported_patient_overlap": patient_leakage,
            "valid": valid,
        },
    }


def _print_report(report):
    print(f"Data root: {report['data_root']}")
    print(f"Split profile: {report['profile']}")
    print("")
    print("Target   Train  Val   Test  Sources")
    for target, fold in report["folds"].items():
        print(
            f"{target:<8} {fold['train']:>5} "
            f"{fold['validation']:>5} {fold['test']:>6}  "
            f"{' '.join(fold['sources'])}"
        )
    print("")
    for key, value in report["summary"].items():
        print(f"{key}={value}")
    limitations = report["audit_metadata"].get("limitations", {})
    if limitations:
        print("")
        print("Patient-identity limitations:")
        for domain, message in limitations.items():
            print(f"- {domain}: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--split-profile",
        choices=("official", "strict"),
        default="strict",
    )
    parser.add_argument("--json-output")
    args = parser.parse_args()
    report = audit_lodo(args.root, args.split_profile)
    _print_report(report)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    raise SystemExit(0 if report["summary"]["valid"] else 1)


if __name__ == "__main__":
    main()
