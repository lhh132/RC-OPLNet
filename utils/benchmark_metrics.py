import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def compute_classification_metrics(labels, probabilities, num_classes=5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(
            f"Expected probability matrix with {num_classes} columns, got "
            f"{probabilities.shape}"
        )
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same row count")
    predictions = probabilities.argmax(axis=1)
    class_ids = list(range(num_classes))
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)
        ),
        "qwk": _finite_or_none(
            cohen_kappa_score(labels, predictions, labels=class_ids, weights="quadratic")
        ),
        "per_class_recall": recall_score(
            labels, predictions, labels=class_ids, average=None, zero_division=0
        ).astype(float).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_ids
        ).astype(int).tolist(),
    }
    present = set(np.unique(labels).tolist())
    if present == set(class_ids):
        metrics["auc_ovo"] = float(
            roc_auc_score(labels, probabilities, average="macro", multi_class="ovo")
        )
    else:
        metrics["auc_ovo"] = None
        metrics["auc_unavailable_reason"] = (
            f"AUC requires all {num_classes} classes; present classes: {sorted(present)}"
        )
    return metrics


def compute_domain_metrics(
    labels, probabilities, domain_ids, domain_names, num_classes=5
):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    domain_ids = np.asarray(domain_ids, dtype=int)
    if not (len(labels) == len(probabilities) == len(domain_ids)):
        raise ValueError(
            "labels, probabilities and domain_ids must have equal length"
        )
    unknown_ids = set(np.unique(domain_ids).tolist()) - set(range(len(domain_names)))
    if unknown_ids:
        raise ValueError(
            "domain id has no matching domain name: {}".format(sorted(unknown_ids))
        )

    pooled = compute_classification_metrics(labels, probabilities, num_classes)
    per_domain = {}
    for domain_id, domain_name in enumerate(domain_names):
        selected = domain_ids == domain_id
        if selected.any():
            per_domain[domain_name] = compute_classification_metrics(
                labels[selected], probabilities[selected], num_classes
            )

    scalar_keys = ("accuracy", "macro_f1", "qwk", "auc_ovo")
    domain_macro = {}
    worst_domain = {}
    for key in scalar_keys:
        values = [
            metrics[key]
            for metrics in per_domain.values()
            if metrics.get(key) is not None
        ]
        domain_macro[key] = float(np.mean(values)) if values else None
        worst_domain[key] = float(np.min(values)) if values else None
    return {
        "pooled": pooled,
        "per_domain": per_domain,
        "domain_macro": domain_macro,
        "worst_domain": worst_domain,
    }


def _without_confusion_matrices(value):
    if isinstance(value, dict):
        return {
            key: _without_confusion_matrices(item)
            for key, item in value.items()
            if key != "confusion_matrix"
        }
    return value


def _write_confusion_matrix(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(matrix)


def write_metrics(output_dir, metrics, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_metrics = _without_confusion_matrices(metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if "pooled" in metrics:
        _write_confusion_matrix(
            output_dir / "confusion_matrix.csv",
            metrics["pooled"]["confusion_matrix"],
        )
        for domain_name, domain_metrics in metrics["per_domain"].items():
            safe_name = Path(domain_name).name
            if safe_name != domain_name:
                raise ValueError(f"Invalid domain name for metrics path: {domain_name}")
            _write_confusion_matrix(
                output_dir / "per_domain" / safe_name / "confusion_matrix.csv",
                domain_metrics["confusion_matrix"],
            )
    else:
        _write_confusion_matrix(
            output_dir / "confusion_matrix.csv", metrics["confusion_matrix"]
        )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata or {}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
