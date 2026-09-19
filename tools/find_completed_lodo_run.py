import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = (
    "status",
    "method",
    "source_domains",
    "target_domains",
    "seed",
    "split_profile",
    "dg_mode",
    "epochs",
    "batch_size",
    "num_workers",
    "validation_interval",
    "max_train_batches",
    "max_eval_batches",
)


def metadata_matches(metadata, expected):
    return (
        isinstance(metadata, dict)
        and all(field in metadata for field in REQUIRED_FIELDS)
        and all(
            metadata[field] == expected[field]
            for field in REQUIRED_FIELDS
        )
    )


def find_completed_run(result_root, expected):
    result_root = Path(result_root)
    if not result_root.is_dir():
        return None

    for metadata_path in sorted(result_root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        if not (run_dir / "done").is_file():
            continue
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if metadata_matches(metadata, expected):
            return run_dir.resolve()
    return None


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-domains", nargs="+", required=True)
    parser.add_argument("--target-domains", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-profile", required=True)
    parser.add_argument("--dg-mode", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--validation-interval", type=int, required=True)
    parser.add_argument("--max-train-batches", type=int, required=True)
    parser.add_argument("--max-eval-batches", type=int, required=True)
    return parser


def main():
    args = _build_parser().parse_args()
    expected = {
        "status": "completed",
        "method": args.method,
        "source_domains": args.source_domains,
        "target_domains": args.target_domains,
        "seed": args.seed,
        "split_profile": args.split_profile,
        "dg_mode": args.dg_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "validation_interval": args.validation_interval,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
    }
    match = find_completed_run(args.result_root, expected)
    if match is None:
        raise SystemExit(1)
    print(match)


if __name__ == "__main__":
    main()
