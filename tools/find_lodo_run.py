import argparse
import json
from pathlib import Path


MATCH_FIELDS = (
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
        and metadata.get("algorithm_class") == expected["algorithm_class"]
        and all(metadata.get(field) == expected[field] for field in MATCH_FIELDS)
    )


def _candidates(result_root, expected):
    for metadata_path in Path(result_root).rglob("run_metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata_matches(metadata, expected):
            yield metadata_path.parent.resolve(), metadata


def find_preferred_run(result_root, expected):
    completed = []
    resumable = []
    if not Path(result_root).is_dir():
        return None
    for run_dir, metadata in _candidates(result_root, expected):
        started = float(metadata.get("start_time_epoch", 0.0))
        if (run_dir / "done").is_file() and metadata.get("status") == "completed":
            completed.append((started, run_dir))
        elif not (run_dir / "done").exists() and (run_dir / "checkpoints" / "last.pth").is_file():
            resumable.append((started, run_dir))
    if completed:
        return "completed", max(completed)[1]
    if resumable:
        return "resumable", max(resumable)[1]
    return None


def _parser():
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
    args = _parser().parse_args()
    expected = {
        "algorithm_class": args.method,
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
    match = find_preferred_run(args.result_root, expected)
    if match is None:
        raise SystemExit(1)
    print(f"{match[0]}\t{match[1]}")


if __name__ == "__main__":
    main()
