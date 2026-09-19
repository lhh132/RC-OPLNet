import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml

from utils.method_names import (
    enabled_gdrnet_components,
    method_display_name,
)
from utils.training_state import config_fingerprint



def _now():
    return datetime.now().astimezone()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line_count(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _git_state(cwd):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _environment_text():
    lines = [
        f"timestamp={_now().isoformat()}",
        f"hostname={socket.gethostname()}",
        f"platform={platform.platform()}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"executable={sys.executable}",
        f"cwd={Path.cwd()}",
        f"torch={torch.__version__}",
        f"cuda_runtime={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cudnn={torch.backends.cudnn.version()}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    if torch.cuda.is_available():
        lines.extend(
            f"gpu_{index}={torch.cuda.get_device_name(index)}"
            for index in range(torch.cuda.device_count())
        )
    lines.append("")
    lines.append("[python-packages]")
    packages = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.lower,
    )
    lines.extend(packages)
    return "\n".join(lines) + "\n"


def _data_manifest(cfg, dataset_size):
    data_root = Path(cfg.DATASET.ROOT)
    split_dir_name = (
        "splits_strict"
        if cfg.DATASET.SPLIT_PROFILE == "strict"
        else "splits"
    )
    split_dir = data_root / split_dir_name
    domains = list(
        dict.fromkeys(
            list(cfg.DATASET.SOURCE_DOMAINS)
            + list(cfg.DATASET.TARGET_DOMAINS)
        )
    )
    split_files = []
    for domain in domains:
        for split in ("train", "crossval"):
            path = split_dir / f"{domain}_{split}.txt"
            item = {
                "domain": domain,
                "split": split,
                "path": str(path),
                "exists": path.is_file(),
            }
            if path.is_file():
                item.update(
                    {
                        "rows": _line_count(path),
                        "sha256": _sha256(path),
                    }
                )
            split_files.append(item)

    audit_path = split_dir / "audit.json"
    audit_file = {
        "path": str(audit_path),
        "exists": audit_path.is_file(),
    }
    if audit_path.is_file():
        audit_file["sha256"] = _sha256(audit_path)

    return {
        "data_root": str(data_root),
        "split_profile": cfg.DATASET.SPLIT_PROFILE,
        "source_domains": list(cfg.DATASET.SOURCE_DOMAINS),
        "target_domains": list(cfg.DATASET.TARGET_DOMAINS),
        "dataset_size": {
            "train": int(dataset_size[0]),
            "validation": int(dataset_size[1]),
            "test": int(dataset_size[2]),
        },
        "split_files": split_files,
        "audit_file": audit_file,
    }


def start_run_record(output_dir, args, cfg, dataset_size):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    commit, dirty = _git_state(Path.cwd())
    dataset_size_record = {
        "train": int(dataset_size[0]),
        "validation": int(dataset_size[1]),
        "test": int(dataset_size[2]),
    }
    metadata = {
        "run_id": output_dir.name,
        "status": "running",
        "method": method_display_name(cfg),
        "algorithm_class": cfg.ALGORITHM,
        "gdrnet_components": enabled_gdrnet_components(cfg),
        "hcs_dcr": (
            {
                "beta": float(cfg.GDRNET.HCS_DCR.BETA),
                "tau": float(cfg.GDRNET.HCS_DCR.TAU),
                "min_weight": float(cfg.GDRNET.HCS_DCR.MIN_WEIGHT),
                "max_weight": float(cfg.GDRNET.HCS_DCR.MAX_WEIGHT),
                "lambda_max": float(cfg.GDRNET.HCS_DCR.LAMBDA_MAX),
                "warmup_ratio": float(
                    cfg.GDRNET.HCS_DCR.WARMUP_RATIO
                ),
            }
            if cfg.GDRNET.USE_HCS_DCR
            else None
        ),
        "rc_jdcr": (
            {
                "beta": float(cfg.GDRNET.RC_JDCR.BETA),
                "tau": float(cfg.GDRNET.RC_JDCR.TAU),
                "a_min": float(cfg.GDRNET.RC_JDCR.A_MIN),
                "residual_clip_log": float(
                    cfg.GDRNET.RC_JDCR.RESIDUAL_CLIP_LOG
                ),
                "min_weight": float(cfg.GDRNET.RC_JDCR.MIN_WEIGHT),
                "max_weight": float(cfg.GDRNET.RC_JDCR.MAX_WEIGHT),
                "warmup_start_ratio": float(
                    cfg.GDRNET.RC_JDCR.WARMUP_START_RATIO
                ),
                "full_weight_ratio": float(
                    cfg.GDRNET.RC_JDCR.FULL_WEIGHT_RATIO
                ),
            }
            if cfg.GDRNET.USE_RC_JDCR
            else None
        ),
        "proto": (
            {
                "weight": float(cfg.GDRNET.PROTO.WEIGHT),
                "momentum": float(cfg.GDRNET.PROTO.MOMENTUM),
                "temperature": float(cfg.GDRNET.PROTO.TEMPERATURE),
            }
            if cfg.GDRNET.USE_PROTO_CONTRAST
            else None
        ),
        "source_domains": list(cfg.DATASET.SOURCE_DOMAINS),
        "target_domains": list(cfg.DATASET.TARGET_DOMAINS),
        "seed": int(cfg.SEED),
        "split_profile": cfg.DATASET.SPLIT_PROFILE,
        "dg_mode": str(cfg.DG_MODE),
        "start_time": started.isoformat(),
        "start_time_epoch": started.timestamp(),
        "end_time": None,
        "duration_seconds": None,
        "epochs": int(cfg.EPOCHS),
        "batch_size": int(cfg.BATCH_SIZE),
        "num_workers": int(cfg.NUM_WORKERS),
        "validation_interval": int(cfg.VAL_EPOCH),
        "max_train_batches": int(cfg.MAX_TRAIN_BATCHES),
        "max_eval_batches": int(cfg.MAX_EVAL_BATCHES),
        "dataset_size": dataset_size_record,
        "output_directory": str(output_dir.resolve()),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_fingerprint": config_fingerprint(cfg),
        "resume_count": 0,
        "external_retinal_data": False,
    }
    _write_json(output_dir / "args.json", vars(args))
    (output_dir / "config_resolved.yaml").write_text(
        cfg.dump(), encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    (output_dir / "command_history.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.txt").write_text(
        _environment_text(), encoding="utf-8"
    )
    _write_json(
        output_dir / "data_manifest.json",
        _data_manifest(cfg, dataset_size),
    )
    _write_json(output_dir / "run_metadata.json", metadata)
    (output_dir / "running").touch()
    return metadata


def write_failure_context(output_dir, epoch, batch_index, labels, domains):
    label_values, label_counts = torch.unique(
        labels.detach().cpu(), return_counts=True
    )
    domain_values, domain_counts = torch.unique(
        domains.detach().cpu(), return_counts=True
    )
    _write_json(
        Path(output_dir) / "failure_batch.json",
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "batch_size": int(labels.shape[0]),
            "label_counts": dict(zip(label_values.tolist(), label_counts.tolist())),
            "domain_counts": dict(zip(domain_values.tolist(), domain_counts.tolist())),
        },
    )


def resume_run_record(output_dir, args, cfg):
    output_dir = Path(output_dir)
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_fingerprint") != config_fingerprint(cfg):
        raise ValueError("Run-record config fingerprint mismatch")
    command = shlex.join(sys.argv) + "\n"
    metadata.update(
        {
            "status": "running",
            "end_time": None,
            "duration_seconds": None,
            "resume_count": int(metadata.get("resume_count", 0)) + 1,
            "last_resume_time": _now().isoformat(),
        }
    )
    _write_json(metadata_path, metadata)
    (output_dir / "failed.json").unlink(missing_ok=True)
    (output_dir / "done").unlink(missing_ok=True)
    (output_dir / "running").touch()
    (output_dir / "command.txt").write_text(command, encoding="utf-8")
    with (output_dir / "command_history.txt").open("a", encoding="utf-8") as handle:
        handle.write(command)
    return metadata


def finish_run_record(output_dir, status, error=None):
    if status not in {"completed", "failed"}:
        raise ValueError(f"Unsupported run status: {status}")

    output_dir = Path(output_dir)
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ended = _now()
    started_epoch = float(metadata["start_time_epoch"])
    metadata.update(
        {
            "status": status,
            "end_time": ended.isoformat(),
            "duration_seconds": max(0.0, ended.timestamp() - started_epoch),
        }
    )
    _write_json(metadata_path, metadata)
    (output_dir / "running").unlink(missing_ok=True)

    if status == "completed":
        (output_dir / "failed.json").unlink(missing_ok=True)
        (output_dir / "done").touch()
    else:
        (output_dir / "done").unlink(missing_ok=True)
        failure = {
            "status": "failed",
            "failed_time": ended.isoformat(),
            "error_type": type(error).__name__ if error else None,
            "message": str(error) if error else None,
            "traceback": (
                "".join(
                    traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                )
                if error
                else None
            ),
        }
        _write_json(output_dir / "failed.json", failure)
