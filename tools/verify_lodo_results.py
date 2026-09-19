import argparse
import json
import math
import re
from pathlib import Path


def _metric_errors(value, path, allow_null_metrics):
    errors = []
    if value is None:
        if not allow_null_metrics:
            errors.append(f"null metric at {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"non-finite metric at {path}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(
                _metric_errors(
                    item,
                    f"{path}.{key}",
                    allow_null_metrics,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(
                _metric_errors(
                    item,
                    f"{path}[{index}]",
                    allow_null_metrics,
                )
            )
    return errors


def _final_metric_dir(run_dir):
    candidates = []
    for path in (run_dir / "test").glob("epoch_*/metrics.json"):
        match = re.fullmatch(r"epoch_(\d+)", path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path.parent))
    return max(candidates, default=(None, None), key=lambda item: item[0])


def verify_run(run_dir, allow_null_metrics=False):
    run_dir = Path(run_dir)
    errors = []
    if not run_dir.is_dir():
        return {
            "run_dir": str(run_dir),
            "valid": False,
            "errors": ["run directory does not exist"],
            "final_epoch": None,
        }
    if not (run_dir / "done").is_file():
        errors.append("missing done")
    if (run_dir / "running").exists():
        errors.append("running marker still exists")
    if (run_dir / "failed.json").exists():
        errors.append("failed.json exists")
    if not (run_dir / "log.txt").is_file():
        errors.append("missing log.txt")
    if not list(run_dir.glob("best*.pth")):
        errors.append("missing best checkpoint")

    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid run_metadata.json: {error}")
            metadata = {}
        if metadata.get("algorithm_class") in {"RC-OPLNet", "RC-OPLNet-Ablation"}:
            required = (
                "config_resolved.yaml",
                "command.txt",
                "environment.txt",
                "data_manifest.json",
                "run_metadata.json",
                "checkpoints/last.pth",
            )
            for relative in required:
                if not (run_dir / relative).is_file():
                    errors.append(f"missing {relative}")

    final_epoch, metric_dir = _final_metric_dir(run_dir)
    if metric_dir is None:
        errors.append("missing final test metrics.json")
    else:
        metrics_path = metric_dir / "metrics.json"
        confusion_path = metric_dir / "confusion_matrix.csv"
        if not confusion_path.is_file():
            errors.append(
                f"missing {confusion_path.relative_to(run_dir)}"
            )
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            errors.extend(
                _metric_errors(
                    metrics,
                    "metrics",
                    allow_null_metrics,
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid metrics.json: {error}")

    return {
        "run_dir": str(run_dir),
        "valid": not errors,
        "errors": errors,
        "final_epoch": final_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--allow-null-metrics", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    reports = [
        verify_run(path, allow_null_metrics=args.allow_null_metrics)
        for path in args.run_dir
    ]
    for report in reports:
        status = "PASS" if report["valid"] else "FAIL"
        print(f"[{status}] {report['run_dir']}")
        for error in report["errors"]:
            print(f"  - {error}")
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    raise SystemExit(0 if all(report["valid"] for report in reports) else 1)


if __name__ == "__main__":
    main()
