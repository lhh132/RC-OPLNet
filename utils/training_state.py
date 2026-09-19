import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def config_fingerprint(cfg):
    record = {
        "algorithm": cfg.ALGORITHM,
        "seed": int(cfg.SEED),
        "source_domains": list(cfg.DATASET.SOURCE_DOMAINS),
        "target_domains": list(cfg.DATASET.TARGET_DOMAINS),
        "split_profile": cfg.DATASET.SPLIT_PROFILE,
        "epochs": int(cfg.EPOCHS),
        "batch_size": int(cfg.BATCH_SIZE),
        "learning_rate": float(cfg.LEARNING_RATE),
        "weight_decay": float(cfg.WEIGHT_DECAY),
        # Frozen v1 checkpoint fields; preserve pre-cleanup resume compatibility.
        **{'dg_adr_fair': "ALLOW_EXTERNAL_RETINAL_DATA: false\nFOCAL_ALPHA: 0.25\nFOCAL_GAMMA: 2.0\nK: 5\nLOSS_ALPHA: 10.0\nMARGIN: 0.1\nSSL_CHECKPOINT: ''\nSYNTHETIC_ROOT: ''\nWARM_UP_EPOCHS: 0\nWEIGHT_LOSS_ALPHA: 1.0\n", 'gga': 'END_STEP: 200\nNEIGHBORHOOD_SIZE: 1.0e-05\nSEARCH_STEPS: 250\nSTART_STEP: 100\n'},
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_last_checkpoint(path, algorithm, scheduler, epoch, best_performance, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format_version": 1,
        "fingerprint": config_fingerprint(cfg),
        "algorithm": algorithm.state_dict(),
        "optimizer": algorithm.optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_performance": float(best_performance),
        "method_state": algorithm.resume_state()
        if hasattr(algorithm, "resume_state")
        else {},
        "rng": capture_rng_state(),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def load_last_checkpoint(path, algorithm, scheduler, cfg):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Last checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if payload.get("fingerprint") != config_fingerprint(cfg):
        raise ValueError("Checkpoint config fingerprint mismatch")
    algorithm.load_state_dict(payload["algorithm"])
    algorithm.optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if hasattr(algorithm, "load_resume_state"):
        algorithm.load_resume_state(payload.get("method_state", {}))
    restore_rng_state(payload["rng"])
    return int(payload["epoch"]), float(payload["best_performance"])
