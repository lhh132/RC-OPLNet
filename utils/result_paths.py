from pathlib import Path

from utils.method_names import method_display_name


def result_path(cfg):
    return (Path("result") / "fundusaug" / cfg.DATASET.SPLIT_PROFILE
            / method_display_name(cfg) / cfg.OUTPUT_PATH)
